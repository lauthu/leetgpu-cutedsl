"""Matrix multiplication: larger tiles, vector reads and a cp.async pipeline.

This follows the SIMT GEMM recipe of CuTe's ampere/sgemm.py example. What
changes relative to v2_tiled.py, and what each change is meant to buy:

- A CTA computes a 128 x 128 tile of C and every thread owns 8 x 8 outputs
  (v2: 64 x 64 and 4 x 4). Per step of k a thread reads 16 values for 64
  multiply-adds, instead of 8 values for 16.
- The tiled MMA is permuted so that a thread's values of A, and of B, are
  runs of four neighbours in shared memory. autovec_copy can then read a run
  with one 128-bit load.
- Shared memory holds three k-tiles. cp.async fills the tile two steps ahead
  while the current one is multiplied, so threads do not wait for global
  memory, and there is one barrier per 512 multiply-adds of a thread (v2: 128).
- Shared memory for A is padded so that the 32 writes of a warp go to 32
  different banks.
- The x axis of the grid runs over the columns of C, as in v1.py.

Naming is CuTe's, as in v2: A is (m, k), B is (n, k), C is (m, n). The
challenge calls CuTe's k "N" and CuTe's n "K".
"""

import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator

THREADS = 256
# The CTA tile in CuTe's (m, n, k) order.
BM, BN, BK = 128, 128, 8
CTA_TILER = (BM, BN, BK)
# k-tiles held in shared memory. The pipeline below needs at least 3.
STAGES = 3
# A is written to shared memory by threads that are neighbours along k, and k
# is the strided direction there. With a stride of BM words, a warp's writes
# would share 4 of the 32 banks; BM + 4 spreads them over all 32 and keeps
# every column 16-byte aligned for the vector reads.
PAD_A = 4


@cute.kernel
def _gemm(
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

    # When k is not a multiple of BK, one k-tile is partial. Shifting A and B
    # back along k makes that tile the first one instead of the last, so only
    # k-tile 0 needs a bound on k.
    residue_k = mA.shape[1] - BK * gA.shape[2]
    gA = cute.domain_offset((0, residue_k, 0), gA)
    gB = cute.domain_offset((0, residue_k, 0), gB)

    # sA: (BM, BK, PIPE)   sB: (BN, BK, PIPE)
    # m and n are the contiguous dimensions: those are the runs the MMA reads.
    smem = SmemAllocator()
    sA_layout = cute.make_layout(
        (BM, BK, STAGES), stride=(1, BM + PAD_A, BK * (BM + PAD_A))
    )
    sB_layout = cute.make_layout((BN, BK, STAGES), stride=(1, BN, BK * BN))
    sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

    # This thread's share of the gmem -> smem copies. CPY is (atom_v, rest_v).
    # tAgA: (CPY, CPY_M, CPY_K, k)   tAsA: (CPY, CPY_M, CPY_K, PIPE)
    # tBgB: (CPY, CPY_N, CPY_K, k)   tBsB: (CPY, CPY_N, CPY_K, PIPE)
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    # Predicates: which of those copies fall inside the matrices. Partitioning
    # an identity tensor the same way as the data gives each copy's global
    # coordinate, here (m, k) for A and (n, k) for B.
    cA = cute.local_tile(
        cute.make_identity_tensor(mA.shape),
        tiler=CTA_TILER,
        coord=tiler_coord,
        proj=(1, None, 1),
    )
    cB = cute.local_tile(
        cute.make_identity_tensor(mB.shape),
        tiler=CTA_TILER,
        coord=tiler_coord,
        proj=(None, 1, 1),
    )
    tAcA = thr_copy_A.partition_S(cute.domain_offset((0, residue_k, 0), cA))
    tBcB = thr_copy_B.partition_S(cute.domain_offset((0, residue_k, 0), cB))

    # A predicate tensor is (rest_v, CPY_M, CPY_K) for A, (rest_v, CPY_N, CPY_K)
    # for B.
    pA_shape = (tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2]))
    pB_shape = (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2]))
    # Full k-tiles only need the bound on m (or n). Stride 0 along k makes every
    # k share one flag.
    tApA = cute.make_rmem_tensor(
        cute.make_layout(pA_shape, stride=(pA_shape[1], 1, 0)), cutlass.Boolean
    )
    tBpB = cute.make_rmem_tensor(
        cute.make_layout(pB_shape, stride=(pB_shape[1], 1, 0)), cutlass.Boolean
    )
    # k-tile 0 also needs k >= 0, so it has one flag per element.
    tApA_k0 = cute.make_rmem_tensor(pA_shape, cutlass.Boolean)
    tBpB_k0 = cute.make_rmem_tensor(pB_shape, cutlass.Boolean)

    for v in cutlass.range_constexpr(pA_shape[0]):
        for m in cutlass.range_constexpr(pA_shape[1]):
            tApA[v, m, 0] = cute.elem_less(tAcA[(0, v), m, 0, 0][0], mA.shape[0])
            for k in cutlass.range_constexpr(pA_shape[2]):
                crd = tAcA[(0, v), m, k, 0]
                tApA_k0[v, m, k] = cute.elem_less(
                    (crd[0], cutlass.Int32(-1)), (mA.shape[0], crd[1])
                )
    for v in cutlass.range_constexpr(pB_shape[0]):
        for n in cutlass.range_constexpr(pB_shape[1]):
            tBpB[v, n, 0] = cute.elem_less(tBcB[(0, v), n, 0, 0][0], mB.shape[0])
            for k in cutlass.range_constexpr(pB_shape[2]):
                crd = tBcB[(0, v), n, k, 0]
                tBpB_k0[v, n, k] = cute.elem_less(
                    (crd[0], cutlass.Int32(-1)), (mB.shape[0], crd[1])
                )

    # A copy whose predicate is false leaves its destination untouched. Start
    # every stage from zeros so that what k-tile 0 skips contributes nothing.
    for i in cutlass.range_constexpr(cute.size(tAsA)):
        tAsA[i] = 0.0
    for i in cutlass.range_constexpr(cute.size(tBsB)):
        tBsB[i] = 0.0

    # Prologue: start the copies of the first STAGES - 1 k-tiles. k-tile 0 is
    # the partial one and goes to stage 0.
    k_pipe_max = cute.size(tAsA, mode=[3])
    k_tile_count = cute.size(tAgA, mode=[3])
    gmem_pipe_read = cutlass.Int32(0)
    cute.copy(
        tiled_copy_A,
        tAgA[None, None, None, gmem_pipe_read],
        tAsA[None, None, None, 0],
        pred=tApA_k0,
    )
    cute.copy(
        tiled_copy_B,
        tBgB[None, None, None, gmem_pipe_read],
        tBsB[None, None, None, 0],
        pred=tBpB_k0,
    )
    cute.arch.cp_async_commit_group()
    gmem_pipe_read = (
        gmem_pipe_read + 1 if gmem_pipe_read + 1 < k_tile_count else cutlass.Int32(0)
    )
    for k_tile in range(1, k_pipe_max - 1):
        if k_tile < k_tile_count:
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, gmem_pipe_read],
                tAsA[None, None, None, k_tile],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, gmem_pipe_read],
                tBsB[None, None, None, k_tile],
                pred=tBpB,
            )
        gmem_pipe_read = (
            gmem_pipe_read + 1
            if gmem_pipe_read + 1 < k_tile_count
            else cutlass.Int32(0)
        )
        cute.arch.cp_async_commit_group()

    # With fewer k-tiles than stages, everything is already on its way. The
    # main loop still issues a copy per iteration, so switch those off.
    if k_tile_count < k_pipe_max:
        for cv in cutlass.range_constexpr(pA_shape[0]):
            for cm in cutlass.range_constexpr(pA_shape[1]):
                tApA[cv, cm, 0] = cutlass.Boolean(0)
        for cv in cutlass.range_constexpr(pB_shape[0]):
            for cn in cutlass.range_constexpr(pB_shape[1]):
                tBpB[cv, cn, 0] = cutlass.Boolean(0)

    # This thread's share of the multiply-accumulate.
    # tCsA: (MMA, MMA_M, MMA_K, PIPE)   tCsB: (MMA, MMA_N, MMA_K, PIPE)
    # tCgC: (MMA, MMA_M, MMA_N)
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    # Stage being read, and stage the next copy writes to.
    smem_pipe_read = cutlass.Int32(0)
    smem_pipe_write = cutlass.Int32(k_pipe_max - 1)
    tCsA_p = tCsA[None, None, None, smem_pipe_read]
    tCsB_p = tCsB[None, None, None, smem_pipe_read]

    # Wait for k-tile 0, then put its first k in registers.
    k_block_max = cute.size(tCrA, mode=[2])
    cute.arch.cp_async_wait_group(k_pipe_max - 2)
    cute.arch.sync_threads()
    cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
    cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

    # Main loop. Two pipelines run one step ahead of the arithmetic:
    # - registers: while k_block is multiplied, k_block + 1 is read from smem;
    # - smem: while one k-tile is multiplied, the next is already there and
    #   the one after it is being copied into the stage that was just freed.
    for _ in range(k_tile_count):
        for k_block in range(k_block_max, unroll_full=True):
            if k_block == k_block_max - 1:
                # The next read crosses into the next k-tile: wait until its
                # copies have landed, for every thread.
                tCsA_p = tCsA[None, None, None, smem_pipe_read]
                tCsB_p = tCsB[None, None, None, smem_pipe_read]
                cute.arch.cp_async_wait_group(k_pipe_max - 2)
                cute.arch.sync_threads()

            # smem -> registers for the next k.
            k_block_next = (k_block + 1) % k_block_max
            cute.autovec_copy(
                tCsA_p[None, None, k_block_next], tCrA[None, None, k_block_next]
            )
            cute.autovec_copy(
                tCsB_p[None, None, k_block_next], tCrB[None, None, k_block_next]
            )

            # gmem -> smem for a later k-tile. A goes before the arithmetic
            # and B after it, to spread the memory requests.
            if k_block == 0:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, smem_pipe_write],
                    pred=tApA,
                )

            # C += A * B for this k.
            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, k_block],
                tCrB[None, None, k_block],
                tCrC,
            )

            if k_block == 0:
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, smem_pipe_write],
                    pred=tBpB,
                )
                cute.arch.cp_async_commit_group()
                smem_pipe_write = smem_pipe_read
                smem_pipe_read = smem_pipe_read + 1
                if smem_pipe_read == k_pipe_max:
                    smem_pipe_read = cutlass.Int32(0)
                # Past the last k-tile the loop keeps copying, to stay free of
                # branches. Those copies are never used; they re-read k-tile 1,
                # not k-tile 0, whose shifted start may lie before the matrix.
                gmem_pipe_read = (
                    gmem_pipe_read + 1
                    if gmem_pipe_read + 1 < k_tile_count
                    else cutlass.Int32(1)
                )

    # No copy may still be in flight when the kernel ends.
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # Registers -> gmem, skipping the outputs that fall outside C.
    tCcC = thr_mma.partition_C(cute.make_identity_tensor(gC.shape))
    tCpC = cute.make_rmem_tensor(tCrC.layout, cutlass.Boolean)
    residue_m = mC.shape[0] - cutlass.Int32(BM) * bidy
    residue_n = mC.shape[1] - cutlass.Int32(BN) * bidx
    for i in cutlass.range_constexpr(cute.size(tCrC)):
        tCpC[i] = cute.elem_less(tCcC[i], (residue_m, residue_n))
    store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
    cute.copy(store_atom, tCrC, tCgC, pred=tCpC)


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
    # tile. Each patch repeats four times over the (128, 8) CTA tile.
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        mA.element_type,
        num_bits_per_copy=mA.element_type.width,
    )
    thr_layout_A = cute.make_layout((THREADS // BK, BK), stride=(BK, 1))
    thr_layout_B = cute.make_layout((BN, THREADS // BN), stride=(1, BN))
    val_layout = cute.make_layout((1, 1))
    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom, thr_layout_A, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom, thr_layout_B, val_layout)

    # Multiply-accumulate: 16 x 16 fused multiply-adds over (m, n), thread
    # 16 * m + n. The permutation gives each thread four neighbouring m and
    # four neighbouring n per 64 x 64, so its values in smem are contiguous
    # runs of four. Over the 128 x 128 tile that is 8 x 8 outputs per thread.
    atoms_layout = cute.make_layout((THREADS // 16, 16, 1), stride=(16, 1, 0))
    permutation_m = cute.make_layout((atoms_layout.shape[0], 4), stride=(4, 1))
    permutation_n = cute.make_layout((atoms_layout.shape[1], 4), stride=(4, 1))
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.MmaUniversalOp(cutlass.Float32),
        atoms_layout,
        permutation_mnk=(permutation_m, permutation_n, None),
    )

    _gemm(mA, mB, mC, tiled_copy_A, tiled_copy_B, tiled_mma).launch(
        grid=(cute.ceil_div(K, BN), cute.ceil_div(M, BM), 1),
        block=(THREADS, 1, 1),
    )
