"""Matrix addition as one flat element-wise add, in its shortest form.

Adding two matrices does not depend on rows and columns, so `solve` views
each N x N matrix as N * N consecutive float32 values. The kernel never sees
two dimensions.

Every thread owns VALUES consecutive elements. A thread whose elements are
all inside the data loads them from A and B, adds them and stores the sum to
C, each as one vector. The thread that owns the last, partial group handles
it one element at a time.

What changes relative to v2_per_thread.py:

- No alignment handling. v2 checked the three addresses and promised the
  compiler 16-byte alignment, so that a group could move as one 128-bit
  access. Timed in one process on the platform's H100 at N = 4096, the same
  kernel on views that promise only 4 bytes was just as fast: the kernel is
  limited by memory bandwidth, not by the number of accesses. So the check,
  the second launch and the second compiled kernel are gone.
- A group is added with load() and store() on the global-memory tensors
  themselves. The register tensors and the three copies are gone.
"""

import cutlass
import cutlass.cute as cute

THREADS = 256  # per block
VALUES = 4  # float32 values per thread


def _flat(x, count):
    """x's memory as a one-dimensional tensor of `count` elements."""
    return cute.make_tensor(x.iterator, cute.make_layout(count, stride=1))


@cute.kernel
def _add(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, count: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # This thread owns group number `group`: elements [first, first + VALUES).
    group = bidx * THREADS + tidx
    first = group * VALUES

    if first + VALUES <= count:
        # The whole group is inside the data. Tiling by VALUES gives it a
        # static shape, which load() and store() need.
        tA = cute.local_tile(A, (VALUES,), (group,))
        tB = cute.local_tile(B, (VALUES,), (group,))
        tC = cute.local_tile(C, (VALUES,), (group,))
        tC.store(tA.load() + tB.load())
    else:
        # The last group when count is not a multiple of VALUES, and the
        # threads past the end of the data: one guarded element at a time.
        for k in cutlass.range_constexpr(VALUES):
            i = first + k
            if i < count:
                C[i] = A[i] + B[i]


@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    # N <= 4096, so the count is at most 2**24. It fits Int32, the type of
    # thread and block indices, and the kernel needs no conversions.
    count = cutlass.Int32(N * N)
    _add(_flat(A, count), _flat(B, count), _flat(C, count), count).launch(
        grid=(cute.ceil_div(count, THREADS * VALUES), 1, 1),
        block=(THREADS, 1, 1),
    )
