"""Matrix transpose through shared memory, with coalesced reads and writes.

The task is output[j][i] = input[i][j]. `solve` wraps both buffers in views
with the same coordinates: A(i, j) is input[i][j], stored row-major, and
Bt(i, j) is output[j][i], which is the output buffer read column-major. The
kernel then only has to copy A(i, j) to Bt(i, j).

A CTA handles one 32 x 32 tile in two steps: global memory -> shared memory,
then shared memory -> global memory. The two steps let each side be accessed
along its own contiguous direction: A along j, Bt along i.

What changes relative to v1.py: the output copy assigns threads along i,
Bt's contiguous direction, so the 32 threads of a warp write 32 neighbouring
floats. In v1 they advanced along j and a warp's write was spread over 8
separate places.

Both copies keep their predicates on every tile. Copying full tiles without
them made no measurable difference on the platform.
"""

import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator

BM, BN = 32, 32
THREADS = 256


def _inside(thr_copy, part, limit):
    """One flag per copy of this thread: is the element inside the matrix?

    Partitioning an identity tensor the same way as the data gives each
    copy's coordinate (m, n) inside the tile. `limit` is how many rows and
    columns of this tile are inside the matrix. The shape is what a tiled copy
    expects for `pred`: (copies per thread value, repeats along m, along n).
    """
    crd = thr_copy.partition_S(cute.make_identity_tensor((BM, BN)))
    shape = (part.shape[0][1], cute.size(part, mode=[1]), cute.size(part, mode=[2]))
    pred = cute.make_rmem_tensor(shape, cutlass.Boolean)
    for v in range(shape[0]):
        for m in range(shape[1]):
            for n in range(shape[2]):
                pred[v, m, n] = cute.elem_less(crd[(0, v), m, n], limit)
    return pred


@cute.kernel
def _transpose(A: cute.Tensor, Bt: cute.Tensor, rows: cute.Int32, cols: cute.Int32):
    tid, _, _ = cute.arch.thread_idx()
    bn, bm, _ = cute.arch.block_idx()

    # The same tile of both views: element (m, n) of gA goes to (m, n) of gB.
    gA = cute.local_tile(A, (BM, BN), (bm, bn))
    gB = cute.local_tile(Bt, (BM, BN), (bm, bn))

    # A row of `s` holds BN values. The extra element per row puts a column
    # of the tile, which is what a warp reads below, into 32 different banks.
    s = SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout((BM, BN), stride=(BN + 1, 1))
    )

    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32
    )
    # In: thread ids advance along n, A's contiguous direction. A warp reads
    # one row of the tile; the (8, 32) patch repeats four times along m.
    copy_in = cute.make_tiled_copy_tv(
        atom,
        cute.make_layout((THREADS // BN, BN), stride=(BN, 1)),
        cute.make_layout((1, 1)),
    )
    # Out: thread ids advance along m, Bt's contiguous direction. A warp
    # writes one column of the tile; the (32, 8) patch repeats four times
    # along n.
    copy_out = cute.make_tiled_copy_tv(
        atom,
        cute.make_layout((BM, THREADS // BM), stride=(1, BM)),
        cute.make_layout((1, 1)),
    )

    thr_in = copy_in.get_slice(tid)
    thr_out = copy_out.get_slice(tid)
    in_src = thr_in.partition_S(gA)
    in_dst = thr_in.partition_D(s)
    out_src = thr_out.partition_S(s)
    out_dst = thr_out.partition_D(gB)

    # How many rows and columns of this tile are inside the matrix. A copy
    # whose flag is false is skipped, so nothing outside the matrix is read
    # or written. The elements of `s` that the first copy skips stay
    # uninitialised, and the second copy skips exactly the same ones.
    limit = (rows - cutlass.Int32(BM) * bm, cols - cutlass.Int32(BN) * bn)
    in_pred = _inside(thr_in, in_dst, limit)
    out_pred = _inside(thr_out, out_src, limit)
    cute.copy(copy_in, in_src, in_dst, pred=in_pred)
    cute.arch.sync_threads()
    cute.copy(copy_out, out_src, out_dst, pred=out_pred)


@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, rows: cute.Int32, cols: cute.Int32):
    # A: the input, rows x cols, row-major.
    A = cute.make_tensor(
        input.iterator, cute.make_layout((rows, cols), stride=(cols, 1))
    )
    # Bt: the output buffer in A's coordinates. The output is cols x rows and
    # row-major, (cols, rows):(rows, 1); swapping shape and stride together
    # gives the same memory as a column-major rows x cols matrix.
    Bt = cute.make_tensor(
        output.iterator, cute.make_layout((rows, cols), stride=(1, rows))
    )
    # x runs over the column tiles, y over the row tiles.
    _transpose(A, Bt, rows, cols).launch(
        grid=(cute.ceil_div(cols, BN), cute.ceil_div(rows, BM), 1),
        block=(THREADS, 1, 1),
    )
