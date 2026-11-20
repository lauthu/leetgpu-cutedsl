"""Matrix multiplication on FP16 tensor cores, with FP32 accuracy.

`solve` picks one of two kernels, as in v4_aligned.py.

`_gemm_split` runs when M and K are multiples of 128, N is a multiple of 16
and N >= 48. It keeps v4's frame: a 128 x 128 tile of C per CTA, 256 threads,
three k-tiles in shared memory filled by cp.async, and the x axis of the grid
over the columns of C. The arithmetic changes. The H100 has no FP32 tensor
core instruction, so every FP32 multiply-add of v4 is replaced by FP16 MMAs:

- Each FP32 input is split into two FP16 values, a = hi + lo / 2048. hi is a
  rounded to FP16, and lo is the rounding error, scaled up so that FP16 keeps
  its leading bits too. Then
      a * b = hi_a * hi_b + (hi_a * lo_b + lo_a * hi_b) / 2048
  up to a term of relative size 2^-22, which is dropped.
- Each of the three products is a 16 x 8 x 16 MMA with FP16 inputs and an FP32
  result. One instruction does 2048 multiply-adds for a warp.
- The main product of every 16-step block starts from zero, and its result is
  added to an FP32 accumulator outside the MMA. An MMA that accumulates into
  its own result truncates the small products against the large running sum,
  and over 6144 steps that costs more accuracy than plain FP32 has. The two
  correction products may accumulate inside the MMA, because their sum is
  divided by 2048 afterwards. The method is from Ootomo and Yokota (2022),
  "Recovering single precision accuracy from Tensor Cores while surpassing
  the FP32 theoretical peak performance".

`_gemm_any` handles every other size with one thread per output, as in v4.

Naming is CuTe's: A is (m, k), B is (n, k), C is (m, n). The challenge calls
CuTe's k "N" and CuTe's n "K".
"""

import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator

THREADS = 256
# The CTA tile of the tensor-core kernel, in CuTe's (m, n, k) order. BK is the
# k extent of one MMA instruction, so a k-tile is one block of MMAs.
BM, BN, BK = 128, 128, 16
CTA_TILER = (BM, BN, BK)
# k-tiles held in shared memory. The pipeline needs at least 3, and at least
# as many k-tiles in the problem.
STAGES = 3
# One MMA instruction covers 16 x 8 of C over 16 steps of k and occupies a
# warp. 2 x 4 of them side by side use the 8 warps and cover 32 x 32 of C,
# and that pattern repeats 4 x 4 times over the CTA tile.
MMA_SHAPE = (16, 8, 16)
ATOMS = (2, 4, 1)
# Shared memory keeps m (or n) contiguous and pads the stride along k. With a
# stride of 128 words, the 32 threads of a warp loading one MMA operand would
# hit 8 of the 32 banks; with 132 they hit all 32, and so do the 32 cp.async
# writes of a warp.
PAD = 4
# a = hi + lo / SPLIT. FP16 keeps 11 significant bits, so the rounding error
# of hi is at most 2^-11 of a. Multiplying it by 2^11 brings it back to a's
# range, where FP16 keeps 11 more bits.
SPLIT = 2048.0


@cute.kernel
def _gemm_split(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    # x runs over the n-tiles (columns of C), y over the m-tiles (rows).
    tiler_coord = (bidy, bidx, None)

    # This CTA's tiles.
    # gA: (BM, BK, k)   gB: (BN, BK, k)   gC: (BM, BN)
    gA = cute.local_tile(mA, tiler=CTA_TILER, coord=tiler_coord, proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=CTA_TILER, coord=tiler_coord, proj=(None, 1, 1))
    gC = cute.local_tile(mC, tiler=CTA_TILER, coord=tiler_coord, proj=(1, 1, None))

    # Shared memory holds the inputs as they are, in FP32.
    # sA: (BM, BK, PIPE)   sB: (BN, BK, PIPE)
    smem = SmemAllocator()
    sA_layout = cute.make_layout(
        (BM, BK, STAGES), stride=(1, BM + PAD, BK * (BM + PAD))
    )
    sB_layout = cute.make_layout(
        (BN, BK, STAGES), stride=(1, BN + PAD, BK * (BN + PAD))
    )
    sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

    # This thread's share of the gmem -> smem copies.
    # tAgA: (CPY, CPY_M, CPY_K, k)   tAsA: (CPY, CPY_M, CPY_K, PIPE)
    # tBgB: (CPY, CPY_N, CPY_K, k)   tBsB: (CPY, CPY_N, CPY_K, PIPE)
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    # This thread's share of the MMAs. MMA is the group of values one thread
    # holds for one instruction: 8 of A, 4 of B, 4 of C.
    # tCsA: (MMA, MMA_M, MMA_K, PIPE)   tCsB: (MMA, MMA_N, MMA_K, PIPE)
    # tCgC: (MMA, MMA_M, MMA_N)
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    mma_m = cute.size(tCgC, mode=[1])
    mma_n = cute.size(tCgC, mode=[2])

    # Registers for one k-tile: the FP32 values as loaded, and their two FP16
    # halves, which are what the MMA takes. (MMA, MMA_M, MMA_K) for A and
    # (MMA, MMA_N, MMA_K) for B, with MMA_K = 1.
    # They are made from the shape alone, which gives a plain column-major
    # layout. Registers made with make_fragment_like keep the stride order of
    # the smem partition, and with those the MMA received its values in the
    # wrong order.
    a_shape = tCsA[None, None, None, 0].shape
    b_shape = tCsB[None, None, None, 0].shape
    rA = cute.make_rmem_tensor(a_shape, cutlass.Float32)
    rB = cute.make_rmem_tensor(b_shape, cutlass.Float32)
    rA_hi = cute.make_rmem_tensor(a_shape, cutlass.Float16)
    rA_lo = cute.make_rmem_tensor(a_shape, cutlass.Float16)
    rB_hi = cute.make_rmem_tensor(b_shape, cutlass.Float16)
    rB_lo = cute.make_rmem_tensor(b_shape, cutlass.Float16)
    # cute.gemm takes A as (MMA, M), B as (MMA, N) and its results as
    # (MMA, M, N). The loop below hands it one row of MMA_N instructions at a
    # time, so M is 1: the results are (MMA, 1, MMA_N), and the accumulators
    # keep that shape per row, (MMA, 1, MMA_N, MMA_M).
    row_shape = (tCgC.shape[0], 1, mma_n)
    main = cute.make_rmem_tensor(row_shape, cutlass.Float32)
    corr = cute.make_rmem_tensor(row_shape, cutlass.Float32)
    acc = cute.make_rmem_tensor((tCgC.shape[0], 1, mma_n, mma_m), cutlass.Float32)
    acc.fill(0.0)
    split = cutlass.Float32(SPLIT)
    unsplit = cutlass.Float32(1.0 / SPLIT)

    # Prologue: start copying the first STAGES - 1 k-tiles, one group each.
    for k_tile in cutlass.range_constexpr(STAGES - 1):
        cute.copy(
            tiled_copy_A, tAgA[None, None, None, k_tile], tAsA[None, None, None, k_tile]
        )
        cute.copy(
            tiled_copy_B, tBgB[None, None, None, k_tile], tBsB[None, None, None, k_tile]
        )
        cute.arch.cp_async_commit_group()

    k_tile_count = cute.size(tAgA, mode=[3])
    # Next k-tile to copy, the stage being read, the stage the next copy fills.
    gmem_pipe_read = cutlass.Int32(STAGES - 1)
    smem_pipe_read = cutlass.Int32(0)
    smem_pipe_write = cutlass.Int32(STAGES - 1)

    for _ in range(k_tile_count):
        # Wait until this k-tile has landed, for every thread. Passing the
        # barrier also means every thread is done with the previous k-tile,
        # so its stage can be refilled.
        cute.arch.cp_async_wait_group(STAGES - 2)
        cute.arch.sync_threads()
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, smem_pipe_write],
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, gmem_pipe_read],
            tBsB[None, None, None, smem_pipe_write],
        )
        cute.arch.cp_async_commit_group()

        # smem -> registers, then split every value into its FP16 halves.
        # a - hi is exact in FP32, and so is the scaling by a power of two.
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]
        cute.autovec_copy(tCsA_p[None, None, 0], rA[None, None, 0])
        cute.autovec_copy(tCsB_p[None, None, 0], rB[None, None, 0])
        va = rA.load()
        va_hi = va.to(cutlass.Float16)
        rA_hi.store(va_hi)
        rA_lo.store(((va - va_hi.to(cutlass.Float32)) * split).to(cutlass.Float16))
        vb = rB.load()
        vb_hi = vb.to(cutlass.Float16)
        rB_hi.store(vb_hi)
        rB_lo.store(((vb - vb_hi.to(cutlass.Float32)) * split).to(cutlass.Float16))

        # One row of instructions at a time: the three products, then add to
        # the accumulators in FP32. cute.gemm adds to the tensor it writes,
        # so `main` and `corr` are cleared first; with a separate zero tensor
        # as the starting value the result came out as NaN. The second
        # correction adds to the first.
        for im in cutlass.range_constexpr(mma_m):
            a_hi = rA_hi[None, im, None]
            a_lo = rA_lo[None, im, None]
            main.fill(0.0)
            corr.fill(0.0)
            cute.gemm(tiled_mma, main, a_hi, rB_hi[None, None, 0], main)
            cute.gemm(tiled_mma, corr, a_hi, rB_lo[None, None, 0], corr)
            cute.gemm(tiled_mma, corr, a_lo, rB_hi[None, None, 0], corr)
            acc_row = acc[None, None, None, im]
            acc_row.store(acc_row.load() + (main.load() + corr.load() * unsplit))

        smem_pipe_write = smem_pipe_read
        smem_pipe_read = smem_pipe_read + 1
        if smem_pipe_read == STAGES:
            smem_pipe_read = cutlass.Int32(0)
        # Past the last k-tile the loop keeps copying, to stay free of
        # branches. Those copies re-read k-tile 0 and are never used.
        gmem_pipe_read = (
            gmem_pipe_read + 1
            if gmem_pipe_read + 1 < k_tile_count
            else cutlass.Int32(0)
        )

    # No copy may still be in flight when the kernel ends.
    cute.arch.cp_async_wait_group(0)

    # Registers -> gmem, row by row: (MMA, MMA_N) of the accumulators to the
    # matching (MMA, MMA_N) of this thread's part of C.
    store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
    for row in cutlass.range_constexpr(mma_m):
        cute.copy(store_atom, acc[None, 0, None, row], tCgC[None, row, None])


@cute.kernel
def _gemm_any(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    # One thread per output: row bidy of C, 256 columns per CTA.
    row = bidy
    col = bidx * THREADS + tidx
    if col < mC.shape[1]:
        acc = cutlass.Float32(0.0)
        for kk in range(mA.shape[1]):
            acc = acc + mA[row, kk] * mB[col, kk]
        mC[row, col] = acc


@cute.jit
def solve(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
    K: cute.Int32,
):
    # Row-major inputs viewed in CuTe's order. B is stored as [N, K], which is
    # [k, n]; viewed as (n, k), its contiguous dimension is n.
    mA = cute.make_tensor(A.iterator, cute.make_layout((M, N), stride=(N, 1)))
    mB = cute.make_tensor(B.iterator, cute.make_layout((K, N), stride=(1, K)))
    mC = cute.make_tensor(C.iterator, cute.make_layout((M, K), stride=(K, 1)))

    # gmem -> smem with cp.async, one float per copy. Thread ids advance along
    # each matrix's contiguous dimension, so a warp reads neighbouring memory:
    # a (32, 8) patch of A's (m, k) tile and a (128, 2) patch of B's (n, k)
    # tile. Each patch repeats eight times over the (128, 16) CTA tile.
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        mA.element_type,
        num_bits_per_copy=mA.element_type.width,
    )
    thr_layout_A = cute.make_layout((THREADS // 8, 8), stride=(8, 1))
    thr_layout_B = cute.make_layout((BN, THREADS // BN), stride=(1, BN))
    val_layout = cute.make_layout((1, 1))
    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom, thr_layout_A, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom, thr_layout_B, val_layout)

    # The tensor-core MMA: FP16 inputs, FP32 result.
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, MMA_SHAPE),
        ATOMS,
    )

    # Every tile full, and at least STAGES k-tiles for the pipeline to fill.
    if (M % BM == 0) & (K % BN == 0) & (N % BK == 0) & (N >= BK * STAGES):
        _gemm_split(mA, mB, mC, tiled_copy_A, tiled_copy_B, tiled_mma).launch(
            grid=(cute.ceil_div(K, BN), cute.ceil_div(M, BM), 1),
            block=(THREADS, 1, 1),
        )
    else:
        _gemm_any(mA, mB, mC).launch(
            grid=(cute.ceil_div(K, THREADS), M, 1),
            block=(THREADS, 1, 1),
        )
