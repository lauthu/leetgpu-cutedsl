import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator

BM, BN = 32, 32
THREADS = 256


def _inside(thr_copy, part, limit):
    """One flag per copy of this thread: is the element inside the matrix?

    FIX (added): the kernel always works on a full 32 x 32 tile, but the last
    tile of a row or column is usually partial, and matrices smaller than a
    tile are nothing but a partial tile. Without these flags the kernel reads
    and, worse, writes outside the matrices.

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

    gA = cute.local_tile(A, (BM, BN), (bm, bn))
    # FIX: Bt now has A's coordinate order (see `solve`), so its tile is taken
    # exactly like A's: element (m, n) of gB is where element (m, n) of gA
    # belongs. The transposition is in Bt's layout, not in the indices.
    # was: gB = cute.local_tile(Bt, (BN, BM), (bn, bm))
    gB = cute.local_tile(Bt, (BM, BN), (bm, bn))

    # FIX: a row of `s` holds BN values, so the padded row stride is BN + 1.
    # BM + 1 only worked because BM == BN.
    # was: cutlass.Float32, cute.make_layout((BM, BN), stride=(BM + 1, 1))
    s = SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout((BM, BN), stride=(BN + 1, 1))
    )

    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32
    )

    copy_in = cute.make_tiled_copy_tv(
        atom, cute.make_layout((8, 32), stride=(32, 1)), cute.make_layout((1, 1))
    )

    # Unchanged, and correct with the new Bt. For later, when performance
    # matters: Bt is contiguous along its first mode, so this thread order no
    # longer makes neighbouring threads write neighbouring memory.
    copy_out = cute.make_tiled_copy_tv(
        atom, cute.make_layout((32, 8), stride=(8, 1)), cute.make_layout((1, 1))
    )

    thr_in = copy_in.get_slice(tid)
    thr_out = copy_out.get_slice(tid)

    # FIX (added): how much of this tile is inside the matrix, and from that
    # the flags for both copies. Both use (m, n) in A's order, so one limit
    # serves both. `rows` and `cols` were passed in but never used before.
    limit = (rows - cutlass.Int32(BM) * bm, cols - cutlass.Int32(BN) * bn)
    in_src = thr_in.partition_S(gA)
    in_dst = thr_in.partition_D(s)
    out_src = thr_out.partition_S(s)
    out_dst = thr_out.partition_D(gB)
    in_pred = _inside(thr_in, in_dst, limit)
    out_pred = _inside(thr_out, out_src, limit)

    # FIX: both copies take a predicate. A copy whose flag is false is
    # skipped: nothing is read and nothing is written. The elements of `s`
    # that the first copy skips stay uninitialised, and the second copy skips
    # exactly the same ones.
    # was: cute.copy(copy_in, thr_in.partition_S(gA), thr_in.partition_D(s))
    cute.copy(copy_in, in_src, in_dst, pred=in_pred)
    cute.arch.sync_threads()
    # was: cute.copy(copy_out, thr_out.partition_S(s), thr_out.partition_D(gB))
    cute.copy(copy_out, out_src, out_dst, pred=out_pred)


@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, rows: cute.Int32, cols: cute.Int32):
    A = cute.make_tensor(
        input.iterator, cute.make_layout((rows, cols), stride=(cols, 1))
    )
    # FIX: the output is a cols x rows matrix stored by rows, so output[j][i]
    # is at offset j * rows + i and its own layout is (cols, rows):(rows, 1).
    # Swapping shape and stride together gives the transposed view,
    # (rows, cols):(1, rows): Bt(i, j) is output[j][i], in A's coordinate
    # order. Writing Bt(i, j) = A(i, j) is then the transpose.
    # The old layout was the transposed view of the *input's* layout: its
    # strides used cols where the output needs rows.
    # was: output.iterator, cute.make_layout((cols, rows), stride=(1, cols))
    Bt = cute.make_tensor(
        output.iterator, cute.make_layout((rows, cols), stride=(1, rows))
    )
    _transpose(A, Bt, rows, cols).launch(
        grid=(cute.ceil_div(cols, BN), cute.ceil_div(rows, BM), 1),
        block=(THREADS, 1, 1),
    )
