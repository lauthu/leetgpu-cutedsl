"""Matrix multiplication on Hopper's warpgroup tensor-core instruction.

`solve` picks one of two kernels, as in v5_tensor_core.py.

`_gemm_warpgroup` runs when M and K are multiples of 128, N is a multiple of
32 and the base addresses of A and B are multiples of 16 bytes. It keeps v5's
idea: every FP32 input is split into two FP16 values, a = hi + lo, and
    a * b = hi_a * hi_b + hi_a * lo_b + lo_a * hi_b
up to a term of relative size 2^-22. What changes is who does the work:

- The products are warpgroup MMAs: 64 x 128 x 16 per instruction, issued by
  128 threads, reading both operands from shared memory. v5 used a warp
  instruction of 16 x 8 x 16 and loaded its operands into registers first.
- Because the instruction reads shared memory, the FP16 halves have to be
  there. Every thread loads its own FP32 values from global memory into
  registers, splits them once, and writes the halves into FP16 tiles. In v5
  every warp split the values it needed, so a value was split up to four
  times, and FP32 tiles went through shared memory first.
- A thread's values are runs of four neighbours in global memory and in the
  FP16 tiles: along k for A, along n for B. One 128-bit load reads a run.
- The instruction is asynchronous. A k-tile's three products are issued, and
  while they run the threads split the next k-tile into a second set of FP16
  tiles and load the one after it.
- lo is not scaled. v5 kept lo * 2048 and divided the corrections by 2048;
  here the corrections go into the same MMA result as the main product, so
  only one extra set of registers is needed. FP16 keeps 11 bits of lo while
  lo is at least 6.1e-5 in magnitude; below that its error is at most 3e-8.
- The MMA result restarts from zero every two k-tiles and is then added to an
  FP32 accumulator outside the MMA. v5 did that after every k-tile.

Shared memory written by ordinary stores is not visible to the instruction
until the writing thread has executed a proxy fence. Without it the results
were wrong on the platform.

`_gemm_any` handles every other case with one thread per output, as in v4.

Naming is CuTe's: A is (m, k), B is (n, k), C is (m, n). The challenge calls
CuTe's k "N" and CuTe's n "K".
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.utils import SmemAllocator

# A warpgroup is four warps. ATOMS is how many of them sit side by side along
# (m, n, k); each runs one instruction per product and k-tile.
WG_THREADS = 128
ATOMS = (2, 1, 1)
THREADS = WG_THREADS * ATOMS[0] * ATOMS[1]
# The CTA tile, in CuTe's (m, n, k) order. BK is the k extent of one
# instruction.
BM, BN, BK = 128, 128, 16
CTA_TILER = (BM, BN, BK)
# One instruction: m is always 64 for this instruction family.
MMA_SHAPE = (BM // ATOMS[0], BN // ATOMS[1], BK)
# FP16 k-tiles in shared memory: the one being multiplied and the one being
# prepared.
BUFFERS = 2
# A thread's values come in runs of RUN neighbours: one 128-bit load of FP32.
RUN = 4
# k-tiles per update of the FP32 accumulator. It is also the number of
# k-tiles a thread holds in registers, so that a load is issued a full
# k-tile before its values are used.
PAIR = 2


def _split(r32, r_hi, r_lo, dst_hi, dst_lo):
    """Split FP32 values in registers into hi and lo = a - hi, both FP16.

    a - hi is exact in FP32.
    """
    v = r32.load()
    v_hi = v.to(cutlass.Float16)
    r_hi.store(v_hi)
    r_lo.store((v - v_hi.to(cutlass.Float32)).to(cutlass.Float16))
    cute.autovec_copy(r_hi, dst_hi)
    cute.autovec_copy(r_lo, dst_lo)


@cute.kernel
def _gemm_warpgroup(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    mma_fresh: cute.TiledMma,
    mma_add: cute.TiledMma,
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

    # Shared memory holds only the FP16 halves, in the swizzled layouts the
    # instruction reads: A with k contiguous, B with n contiguous, which is
    # how each is stored in global memory. The swizzle goes on the pointer
    # and the layout stays plain, as the instruction's operands require.
    # sA_hi, sA_lo: (BM, BK, BUFFERS)   sB_hi, sB_lo: (BN, BK, BUFFERS)
    smem = SmemAllocator()
    a16 = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, CTA_TILER, cutlass.Float16, BUFFERS
    )
    b16 = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.COL_MAJOR, CTA_TILER, cutlass.Float16, BUFFERS
    )
    sA_hi = smem.allocate_tensor(cutlass.Float16, a16.outer, 1024, swizzle=a16.inner)
    sA_lo = smem.allocate_tensor(cutlass.Float16, a16.outer, 1024, swizzle=a16.inner)
    sB_hi = smem.allocate_tensor(cutlass.Float16, b16.outer, 1024, swizzle=b16.inner)
    sB_lo = smem.allocate_tensor(cutlass.Float16, b16.outer, 1024, swizzle=b16.inner)

    # This thread's values of every k-tile: where they are in global memory
    # and where their halves go in the FP16 tiles.
    # tAgA: (CPY, CPY_M, CPY_K, k)   tAsA_hi, tAsA_lo: (CPY, CPY_M, CPY_K, BUFFERS)
    # tBgB: (CPY, CPY_N, CPY_K, k)   tBsB_hi, tBsB_lo: (CPY, CPY_N, CPY_K, BUFFERS)
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA_hi = thr_copy_A.partition_D(sA_hi)
    tAsA_lo = thr_copy_A.partition_D(sA_lo)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB_hi = thr_copy_B.partition_D(sB_hi)
    tBsB_lo = thr_copy_B.partition_D(sB_lo)

    # Registers: PAIR k-tiles of FP32 values, and one k-tile of FP16 halves.
    load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=RUN * 32
    )
    a_shape = tAgA[None, None, None, 0].shape
    b_shape = tBgB[None, None, None, 0].shape
    rA = [cute.make_rmem_tensor(a_shape, cutlass.Float32) for _ in range(PAIR)]
    rB = [cute.make_rmem_tensor(b_shape, cutlass.Float32) for _ in range(PAIR)]
    rA_hi = cute.make_rmem_tensor(a_shape, cutlass.Float16)
    rA_lo = cute.make_rmem_tensor(a_shape, cutlass.Float16)
    rB_hi = cute.make_rmem_tensor(b_shape, cutlass.Float16)
    rB_lo = cute.make_rmem_tensor(b_shape, cutlass.Float16)

    # The MMA operands. The instruction takes shared memory as a descriptor
    # per warpgroup, so A and B are partitioned for the warpgroup's first
    # thread; C is partitioned per thread.
    # tCrA_*: (MMA, MMA_M, MMA_K, BUFFERS)   tCrB_*: (MMA, MMA_N, MMA_K, BUFFERS)
    # tCgC: (MMA, MMA_M, MMA_N)
    warp_group = cute.arch.make_warp_uniform(tidx // WG_THREADS)
    warp_group_layout = cute.make_layout(THREADS // WG_THREADS, stride=WG_THREADS)
    wg_mma = mma_add.get_slice(warp_group_layout(warp_group))
    tCrA_hi = mma_add.make_fragment_A(wg_mma.partition_A(sA_hi))
    tCrA_lo = mma_add.make_fragment_A(wg_mma.partition_A(sA_lo))
    tCrB_hi = mma_add.make_fragment_B(wg_mma.partition_B(sB_hi))
    tCrB_lo = mma_add.make_fragment_B(wg_mma.partition_B(sB_lo))
    tCgC = mma_add.get_slice(tidx).partition_C(gC)

    # acc: the FP32 sum. main: the MMA result of the current PAIR k-tiles.
    acc = cute.make_rmem_tensor(tCgC.shape, cutlass.Float32)
    main = cute.make_rmem_tensor(tCgC.shape, cutlass.Float32)
    acc.fill(0.0)
    main.fill(0.0)
    # mma_fresh overwrites its result, mma_add adds to it.
    mma_fresh.set(warpgroup.Field.ACCUMULATE, False)
    mma_add.set(warpgroup.Field.ACCUMULATE, True)

    k_tile_count = cute.size(tAgA, mode=[3])

    # Prologue: k-tile 0 goes into FP16 buffer 0, and the next PAIR k-tiles
    # into the registers. Past the last k-tile the index wraps to 0; those
    # values are loaded and split but never multiplied.
    cute.copy(load, tAgA[None, None, None, 0], rA[0])
    cute.copy(load, tBgB[None, None, None, 0], rB[0])
    _split(
        rA[0], rA_hi, rA_lo, tAsA_hi[None, None, None, 0], tAsA_lo[None, None, None, 0]
    )
    _split(
        rB[0], rB_hi, rB_lo, tBsB_hi[None, None, None, 0], tBsB_lo[None, None, None, 0]
    )
    cute.arch.fence_proxy("async.shared", space="cta")
    gmem_next = cutlass.Int32(1)
    if gmem_next >= k_tile_count:
        gmem_next = cutlass.Int32(0)
    for i in cutlass.range_constexpr(PAIR):
        cute.copy(load, tAgA[None, None, None, gmem_next], rA[(i + 1) % PAIR])
        cute.copy(load, tBgB[None, None, None, gmem_next], rB[(i + 1) % PAIR])
        gmem_next = gmem_next + 1
        if gmem_next >= k_tile_count:
            gmem_next = cutlass.Int32(0)
    # The FP16 buffer to multiply.
    buf = cutlass.Int32(0)

    for _ in range(k_tile_count // PAIR):
        for part in cutlass.range_constexpr(PAIR):
            # Every thread has written and fenced this k-tile's FP16 halves,
            # and every warpgroup has finished the previous k-tile's MMAs, so
            # the other FP16 buffer may be overwritten below.
            cute.arch.sync_threads()

            # Issue the three products. They run while the code below
            # executes.
            warpgroup.fence()
            a_hi = tCrA_hi[None, None, 0, buf]
            a_lo = tCrA_lo[None, None, 0, buf]
            b_hi = tCrB_hi[None, None, 0, buf]
            b_lo = tCrB_lo[None, None, 0, buf]
            if cutlass.const_expr(part == 0):
                cute.gemm(mma_fresh, main, a_hi, b_hi, main)
            else:
                cute.gemm(mma_add, main, a_hi, b_hi, main)
            cute.gemm(mma_add, main, a_hi, b_lo, main)
            cute.gemm(mma_add, main, a_lo, b_hi, main)
            warpgroup.commit_group()

            # Meanwhile: split the next k-tile, which is already in
            # registers, into the other FP16 buffer, and make the writes
            # visible to the instruction. Then load the k-tile PAIR ahead
            # into the registers that just became free.
            buf_next = cutlass.Int32(1) - buf
            _split(
                rA[(part + 1) % PAIR],
                rA_hi,
                rA_lo,
                tAsA_hi[None, None, None, buf_next],
                tAsA_lo[None, None, None, buf_next],
            )
            _split(
                rB[(part + 1) % PAIR],
                rB_hi,
                rB_lo,
                tBsB_hi[None, None, None, buf_next],
                tBsB_lo[None, None, None, buf_next],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.copy(load, tAgA[None, None, None, gmem_next], rA[(part + 1) % PAIR])
            cute.copy(load, tBgB[None, None, None, gmem_next], rB[(part + 1) % PAIR])

            # The results are in the registers only after the wait.
            warpgroup.wait_group(0)

            buf = buf_next
            gmem_next = gmem_next + 1
            if gmem_next >= k_tile_count:
                gmem_next = cutlass.Int32(0)

        acc.store(acc.load() + main.load())

    # Registers -> gmem.
    cute.autovec_copy(acc, tCgC)


@cute.kernel
def _gemm_any(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    # One thread per output: row bidy of C, THREADS columns per CTA.
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

    # Which values a thread owns. Its loads are 128-bit: RUN neighbouring
    # values along each matrix's contiguous dimension.
    load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=RUN * 32
    )
    # A: (64, 4) threads over (m, runs along k), repeated twice along m.
    tiled_copy_A = cute.make_tiled_copy_tv(
        load,
        cute.make_layout((THREADS // (BK // RUN), BK // RUN), stride=(BK // RUN, 1)),
        cute.make_layout((1, RUN)),
    )
    # B: (32, 8) threads over (runs along n, k), repeated twice along k.
    tiled_copy_B = cute.make_tiled_copy_tv(
        load,
        cute.make_layout((BN // RUN, THREADS // (BN // RUN)), stride=(1, BN // RUN)),
        cute.make_layout((RUN, 1)),
    )

    # The warpgroup MMA: FP16 operands read from shared memory, FP32 result
    # in registers. Two copies of it, because whether it overwrites or adds
    # to its result is a setting of the object.
    mma_op = warpgroup.MmaF16BF16Op(
        cutlass.Float16,
        cutlass.Float32,
        MMA_SHAPE,
        warpgroup.OperandSource.SMEM,
        warpgroup.OperandMajorMode.K,
        warpgroup.OperandMajorMode.MN,
    )
    mma_fresh = cute.make_tiled_mma(mma_op, ATOMS)
    mma_add = cute.make_tiled_mma(mma_op, ATOMS)

    # Every tile full, a whole number of PAIRs of k-tiles, and base addresses
    # that allow 128-bit loads.
    addresses = A.iterator.toint() | B.iterator.toint()
    if (M % BM == 0) & (K % BN == 0) & (N % (BK * PAIR) == 0) & ((addresses & 15) == 0):
        # Tell the compiler what was just checked, without changing any
        # value: the row strides are multiples of 16 elements and the base
        # addresses of 16 bytes, so every run starts on a 16-byte boundary.
        n_div = cute.assume(N, divby=BK)
        k_div = cute.assume(K, divby=BN)
        mA16 = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float32,
                A.iterator.toint(),
                A.iterator.memspace,
                assumed_align=16,
            ),
            cute.make_layout((M, N), stride=(n_div, 1)),
        )
        mB16 = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float32,
                B.iterator.toint(),
                B.iterator.memspace,
                assumed_align=16,
            ),
            cute.make_layout((K, N), stride=(1, k_div)),
        )
        _gemm_warpgroup(
            mA16, mB16, mC, tiled_copy_A, tiled_copy_B, mma_fresh, mma_add
        ).launch(
            grid=(cute.ceil_div(K, BN), cute.ceil_div(M, BM), 1),
            block=(THREADS, 1, 1),
        )
    else:
        _gemm_any(mA, mB, mC).launch(
            grid=(cute.ceil_div(K, THREADS), M, 1),
            block=(THREADS, 1, 1),
        )
