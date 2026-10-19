"""Initial one-element-per-thread solution for LeetGPU Vector Addition."""

import cutlass
import cutlass.cute as cute


@cute.kernel
def vector_add_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    N: cute.Uint32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    block_size, _, _ = cute.arch.block_dim()
    i = bidx * block_size + tidx
    if i < N:
        C[i] = A[i] + B[i]


@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    threads = 256
    vector_add_kernel(A, B, C, N).launch(
        grid=((N + threads - 1) // threads, 1, 1),
        block=(threads, 1, 1),
    )
