"""Matrix multiplication written with CuTe's tiled copy and tiled MMA.

The algorithm is the one in v1.py: a CTA of 256 threads computes a 64 x 64 tile
of C, walks the summed dimension 16 elements at a time, stages each slice of A
and B in shared memory, and every thread accumulates 16 outputs in registers.
v1 computes every index by hand. Here layouts decide which thread touches
which element, and cute.copy and cute.gemm do the transfers and the arithmetic.

Naming. The challenge computes C[M, K] = A[M, N] @ B[N, K], so N is the summed
dimension. CuTe calls the summed dimension k and the columns of C n. `solve`
keeps the challenge's argument names; everything after its first three lines
uses CuTe's letters: A is (m, k), B is (n, k), C is (m, n).
"""

import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator

THREADS = 256
# The CTA tile in CuTe's (m, n, k) order: a 64 x 64 tile of C, and 16 elements
# of the summed dimension per step.
BM, BN, BK = 64, 64, 16
CTA_TILER = (BM, BN, BK)


@cute.kernel
def _gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    tiler_coord = (bidx, bidy, None)

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

    # sA: (BM, BK)   sB: (BN, BK)
    smem = SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, cute.make_layout((BM, BK)), 16)
    sB = smem.allocate_tensor(mB.element_type, cute.make_layout((BN, BK)), 16)

    # This thread's share of the gmem -> smem copies. CPY is (atom_v, rest_v).
    # tAgA: (CPY, CPY_M, CPY_K, k)   tAsA: (CPY, CPY_M, CPY_K)
    # tBgB: (CPY, CPY_N, CPY_K, k)   tBsB: (CPY, CPY_N, CPY_K)
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
    # from zeros so that what k-tile 0 skips contributes nothing; the rows and
    # columns outside the matrices then stay zero for the whole kernel.
    for i in cutlass.range_constexpr(cute.size(tAsA)):
        tAsA[i] = 0.0
    for i in cutlass.range_constexpr(cute.size(tBsB)):
        tBsB[i] = 0.0

    # This thread's share of the multiply-accumulate.
    # tCsA: (MMA, MMA_M, MMA_K)   tCsB: (MMA, MMA_N, MMA_K)
    # tCgC: (MMA, MMA_M, MMA_N)
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    k_tile_count = cute.size(tAgA, mode=[3])
    for k_tile in range(k_tile_count):
        # gmem -> smem
        if k_tile == 0:
            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile], tAsA, pred=tApA_k0)
            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile], tBsB, pred=tBpB_k0)
        else:
            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile], tAsA, pred=tApA)
            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile], tBsB, pred=tBpB)
        # Every thread reads elements that other threads wrote.
        cute.arch.sync_threads()

        # smem -> registers, then C += A * B, one k at a time.
        for k_block in cutlass.range_constexpr(BK):
            rA = tCrA[None, None, k_block]
            rB = tCrB[None, None, k_block]
            cute.autovec_copy(tCsA[None, None, k_block], rA)
            cute.autovec_copy(tCsB[None, None, k_block], rB)
            cute.gemm(tiled_mma, tCrC, rA, rB, tCrC)
        # The next k-tile overwrites smem only after everyone has read it.
        cute.arch.sync_threads()

    # Registers -> gmem, skipping the outputs that fall outside C.
    tCcC = thr_mma.partition_C(cute.make_identity_tensor(gC.shape))
    tCpC = cute.make_rmem_tensor(tCrC.layout, cutlass.Boolean)
    residue_m = mC.shape[0] - cutlass.Int32(BM) * bidx
    residue_n = mC.shape[1] - cutlass.Int32(BN) * bidy
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

    # gmem -> smem. The thread layout maps a patch of the CTA tile to thread
    # ids, with the ids advancing along the matrix's contiguous dimension:
    # (16, 16) of A's (m, k), and (64, 4) of B's (n, k). Each thread moves one
    # float per patch, and the patch repeats over the (64, 16) tile, so every
    # thread moves four floats of A and four of B per k-tile.
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type)
    thr_layout_A = cute.make_layout((THREADS // BK, BK), stride=(BK, 1))
    thr_layout_B = cute.make_layout((BN, THREADS // BN), stride=(1, BN))
    val_layout = cute.make_layout((1, 1))
    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom, thr_layout_A, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom, thr_layout_B, val_layout)

    # Multiply-accumulate. The universal MMA op is one fused multiply-add,
    # c += a * b. Laying 16 x 16 of them over (m, n) assigns thread 16 * m + n.
    # That 16 x 16 patch repeats over the 64 x 64 tile of C, so every thread
    # owns a 4 x 4 set of outputs that are 16 apart.
    atoms_layout = cute.make_layout((THREADS // 16, 16, 1), stride=(16, 1, 0))
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.MmaUniversalOp(cutlass.Float32), atoms_layout
    )

    grid_dim = *cute.ceil_div(mC.shape, (BM, BN)), 1
    _gemm(mA, mB, mC, tiled_copy_A, tiled_copy_B, tiled_mma).launch(
        grid=grid_dim, block=(THREADS, 1, 1)
    )
