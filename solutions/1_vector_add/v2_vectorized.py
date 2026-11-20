"""CuTe DSL vector addition for LeetGPU's one-dimensional float32 tensors.

Full tiles use four contiguous values per thread and auto-vectorized copies.
The final partial tile uses guarded scalar accesses. GPU validation and timing
must be performed on a CUDA machine.
"""

import cutlass
import cutlass.cute as cute


@cute.jit
def _with_alignment_16(x: cute.Tensor):
    # Call only after checking the address. Unlike Pointer.align(), this does
    # not round or otherwise change the address.
    ptr = cute.make_ptr(
        x.element_type,
        x.iterator.toint(),
        x.iterator.memspace,
        assumed_align=16,
    )
    return cute.make_tensor(ptr, x.layout)


@cute.kernel
def vector_add_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    N: cute.Uint32,
    threads: cutlass.Constexpr,
    values: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tile_size = threads * values
    base = cute.Uint32(bidx) * tile_size

    # Uniform across the block: only complete tiles enter the vector path.
    if cute.Uint32(bidx) < N // tile_size:
        # (thread, value) -> values * thread + value within this tile.
        tv_layout = cute.make_layout((threads, values), stride=(values, 1))
        tile_A = cute.local_tile(A, (tile_size,), (bidx,))
        tile_B = cute.local_tile(B, (tile_size,), (bidx,))
        tile_C = cute.local_tile(C, (tile_size,), (bidx,))

        tA = cute.composition(tile_A, tv_layout)[tidx, None]
        tB = cute.composition(tile_B, tv_layout)[tidx, None]
        tC = cute.composition(tile_C, tv_layout)[tidx, None]

        rA = cute.make_rmem_tensor((values,), A.element_type)
        rB = cute.make_rmem_tensor((values,), B.element_type)
        cute.autovec_copy(tA, rA)
        cute.autovec_copy(tB, rB)
        rA.store(rA.load() + rB.load())
        cute.autovec_copy(rA, tC)
    else:
        # Keep adjacent lanes contiguous for the scalar tail.
        for k in cutlass.range_constexpr(values):
            i = base + cute.Uint32(tidx) + k * threads
            if i < N:
                C[i] = A[i] + B[i]


@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    threads = 256
    values = 4
    tile_size = threads * values

    if N > 0:
        blocks = (N - 1) // tile_size + 1
        # The caller may supply only 4-byte alignment metadata, even when its
        # allocations are actually 16-byte aligned. Prove the stronger fact
        # before exposing it to the vectorizing compiler.
        addresses = A.iterator.toint() | B.iterator.toint() | C.iterator.toint()
        if (addresses & 15) == 0:
            A16 = _with_alignment_16(A)
            B16 = _with_alignment_16(B)
            C16 = _with_alignment_16(C)
            vector_add_kernel(A16, B16, C16, N, threads, values).launch(
                grid=(blocks, 1, 1), block=(threads, 1, 1)
            )
        else:
            vector_add_kernel(A, B, C, N, threads, values).launch(
                grid=(blocks, 1, 1), block=(threads, 1, 1)
            )
