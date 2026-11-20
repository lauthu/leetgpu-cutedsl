import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator

BLOCK_SIZE = 256
M_TILE = 64
K_TILE = 64
N_TILE = 16


@cute.kernel
def _matmul(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    row0, col0 = by * M_TILE, bx * K_TILE

    # each thread works 4 waves. each tile processing (16, N_TILE) in A_TILE,
    # (N_TILE, 16) in B_TILE.
    lane_row, lane_col = tid // N_TILE, tid % N_TILE

    smem = SmemAllocator()
    sA = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((M_TILE, N_TILE), stride=(N_TILE, 1))
    )
    sB = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((N_TILE, K_TILE), stride=(K_TILE, 1))
    )
    atom = cute.make_mma_atom(cute.nvgpu.MmaUniversalOp(cutlass.Float32))

    a = cute.make_rmem_tensor((1, 4, 1), cutlass.Float32)
    b = cute.make_rmem_tensor((1, 4, 1), cutlass.Float32)
    acc = cute.make_rmem_tensor((1, 4, 4), cutlass.Float32)
    acc.fill(0.0)

    for tile in range(cute.ceil_div(N, N_TILE)):
        n0 = tile * N_TILE

        # collaboratively load A tile and B tile. per wave: load A - (16, 16),
        # load B - (4, 64)
        for i in cutlass.range_constexpr(4):
            p = tid + BLOCK_SIZE * i
            am, an = p // N_TILE, p % N_TILE
            bn, bk = p // K_TILE, p % K_TILE
            av = cutlass.Float32(0.0)
            bv = cutlass.Float32(0.0)
            if (row0 + am < M) & (n0 + an < N):
                av = A[row0 + am, n0 + an]
            if (n0 + bn < N) & (col0 + bk < K):
                bv = B[n0 + bn, col0 + bk]
            sA[am, an] = av
            sB[bn, bk] = bv
        cute.arch.sync_threads()

        for n in cutlass.range_constexpr(N_TILE):
            for i in cutlass.range_constexpr(4):
                a[0, i, 0] = sA[lane_row + N_TILE * i, n]
                b[0, i, 0] = sB[n, lane_col + N_TILE * i]
            cute.gemm(atom, acc, a, b, acc)
        cute.arch.sync_threads()

    for i in cutlass.range_constexpr(4):
        row = row0 + lane_row + N_TILE * i
        for j in cutlass.range_constexpr(4):
            col = col0 + lane_col + N_TILE * j
            if (row < M) & (col < K):
                C[row, col] = acc[0, i, j]


@cute.jit
def solve(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
):
    a = cute.make_tensor(A.iterator, cute.make_layout((M, N), stride=(N, 1)))
    b = cute.make_tensor(B.iterator, cute.make_layout((N, K), stride=(K, 1)))
    c = cute.make_tensor(C.iterator, cute.make_layout((M, K), stride=(K, 1)))
    # handle (64, 64) per CTA, with 256 threads.
    _matmul(a, b, c, M, N, K).launch(
        grid=(cute.ceil_div(K, K_TILE), cute.ceil_div(M, M_TILE), 1),
        block=(BLOCK_SIZE, 1, 1),
    )
