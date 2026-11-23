"""Matrix addition as one flat element-wise add.

Adding two matrices does not depend on rows and columns, so `solve` views
each N x N matrix as N * N consecutive float32 values. The kernel never sees
two dimensions.

Every thread owns VALUES consecutive elements. A thread whose elements are
all inside the data moves them together: one 128-bit access per matrix when
the buffers are 16-byte aligned. The thread that owns the last, partial group
handles it one element at a time.

What changes relative to v1.py: nothing in how a full group is accessed. The
flat view and the alignment promise come from one helper, so a matrix has one
name in `solve`. The choice between the two paths is made per thread, not per
block of 1024 elements, so there is no tile and matrices from 2 x 2 up reach
the vector path. All index arithmetic is Int32.
"""

import cutlass
import cutlass.cute as cute

THREADS = 256  # per block
VALUES = 4  # float32 values per thread
VALUE_BYTES = 4
VECTOR_BYTES = VALUES * VALUE_BYTES  # one 128-bit access


def _flat(x, count, align):
    """x's memory as a one-dimensional tensor of `count` elements.

    `align` is the alignment in bytes that the new pointer promises to the
    compiler. The address itself is not rounded or changed, so the caller must
    have checked it.
    """
    ptr = cute.make_ptr(
        x.element_type, x.iterator.toint(), x.iterator.memspace, assumed_align=align
    )
    return cute.make_tensor(ptr, cute.make_layout(count, stride=1))


@cute.kernel
def _add(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, count: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # This thread owns group number `group`: elements [first, first + VALUES).
    group = bidx * THREADS + tidx
    first = group * VALUES

    if first + VALUES <= count:
        # The whole group is inside the data. Tiling by VALUES gives it a
        # static shape, which a vectorized copy needs.
        tA = cute.local_tile(A, (VALUES,), (group,))
        tB = cute.local_tile(B, (VALUES,), (group,))
        tC = cute.local_tile(C, (VALUES,), (group,))
        rA = cute.make_rmem_tensor((VALUES,), A.element_type)
        rB = cute.make_rmem_tensor((VALUES,), B.element_type)
        rC = cute.make_rmem_tensor((VALUES,), C.element_type)

        # autovec_copy uses the widest access that the pointers' promised
        # alignment allows: 128 bits with 16 bytes, 32 bits with 4 bytes. That
        # makes this kernel correct for both views that `solve` can pass.
        cute.autovec_copy(tA, rA)
        cute.autovec_copy(tB, rB)
        rC.store(rA.load() + rB.load())
        cute.autovec_copy(rC, tC)
    else:
        # The last group when count is not a multiple of VALUES, and the
        # threads past the end of the data: one guarded element at a time.
        for k in cutlass.range_constexpr(VALUES):
            i = first + k
            if i < count:
                C[i] = A[i] + B[i]


def _launch(A, B, C, count, align):
    _add(
        _flat(A, count, align), _flat(B, count, align), _flat(C, count, align), count
    ).launch(
        grid=(cute.ceil_div(count, THREADS * VALUES), 1, 1),
        block=(THREADS, 1, 1),
    )


@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    # N <= 4096, so the count is at most 2**24. It fits Int32, the type of
    # thread and block indices, and the kernel needs no conversions.
    count = cutlass.Int32(N * N)

    # The platform's tensors promise only 4-byte alignment, which rules out
    # 128-bit accesses. Check the real addresses and promise 16 bytes only
    # when it is true. The other branch is the same kernel compiled for
    # 4-byte accesses; it costs a second compilation and no run time.
    addresses = A.iterator.toint() | B.iterator.toint() | C.iterator.toint()
    if (addresses & (VECTOR_BYTES - 1)) == 0:
        _launch(A, B, C, count, VECTOR_BYTES)
    else:
        _launch(A, B, C, count, VALUE_BYTES)
