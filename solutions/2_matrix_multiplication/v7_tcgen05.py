"""Matrix multiplication on Blackwell: tcgen05 MMA with tensor memory.

This version is for B200. Its MMA instruction exists only on Blackwell, so on
other GPUs `solve` does not compile; v6_warpgroup.py is the Hopper version.

`solve` picks one of two kernels, as in v6.

`_gemm_tcgen05` runs when M and K are multiples of 128, N is a multiple of 64
and the base addresses of A and B are multiples of 16 bytes. The arithmetic
and the way data reaches shared memory are v6's: every FP32 input is split
into two FP16 values, a = hi + lo, by the thread that loaded it, and
    a * b = hi_a * hi_b + hi_a * lo_b + lo_a * hi_b
up to a term of relative size 2^-22. What Blackwell changes:

- One thread issues an MMA for the whole 128 x 128 tile, 16 steps of k per
  instruction. On Hopper a warpgroup of 128 threads issued 64 x 128.
- The result is not in registers. It accumulates in tensor memory, a store
  of the SM that only the MMA writes; threads read it with a dedicated load.
  With no accumulators to hold, a thread needs few registers, so the CTA
  uses 512 threads and each one loads and splits less.
- A k-tile is 64 steps of k: four instructions per product, twelve in all,
  between two barriers. v6 used 16. The barrier, the wait for the MMAs and
  the read of tensor memory cost the same per k-tile whatever its depth.
- After every k-tile the threads wait for its MMAs, read the result from
  tensor memory and add it to an FP32 accumulator in registers; the next
  k-tile starts from zero. Leaving the sum in tensor memory for the whole
  kernel was 4% of outputs outside tolerance on the platform: the MMA does
  not round its running sum the way an FP32 add does.

As in v6, lo is not scaled: FP16 keeps 11 bits of it while it is at least
6.1e-5 in magnitude, and below that its error is at most 3e-8. And as in v6,
shared memory written by ordinary stores needs a proxy fence before the
instruction reads it.

`_gemm_any` handles every other case with one thread per output, as in v4.

Naming is CuTe's: A is (m, k), B is (n, k), C is (m, n). The challenge calls
CuTe's k "N" and CuTe's n "K".
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils import SmemAllocator

THREADS = 512
# The CTA tile, in CuTe's (m, n, k) order.
BM, BN, BK = 128, 128, 64
CTA_TILER = (BM, BN, BK)
# One instruction covers the tile's m and n, and 16 steps of k.
MMA_K = 16
MMA_SHAPE = (BM, BN, MMA_K)
# FP16 k-tiles in shared memory: the one being multiplied and the one being
# prepared.
BUFFERS = 2
# A thread's values come in runs of RUN neighbours: one 128-bit load of FP32.
RUN = 4
# Tensor memory has 128 lanes, one per row of the tile. A warp reads 32 of
# them, so four warps read every row, and each group of four warps takes its
# own share of the columns.
TMEM_THREADS = 128
# One column of tensor memory per column of the tile.
TMEM_COLUMNS = BN


@cute.struct
class SharedStorage:
    # The two barriers of the accumulator hand-over, and where the address
    # of the allocated tensor memory is kept.
    acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


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
def _gemm_tcgen05(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    mma_fresh: cute.TiledMma,
    mma_add: cute.TiledMma,
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, bidz = cute.arch.block_idx()
    # x runs over the n-tiles (columns of C), y over the m-tiles (rows).
    tiler_coord = (bidy, bidx, None)

    # This CTA's tiles.
    # gA: (BM, BK, k)   gB: (BN, BK, k)   gC: (BM, BN)
    gA = cute.local_tile(mA, tiler=CTA_TILER, coord=tiler_coord, proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=CTA_TILER, coord=tiler_coord, proj=(None, 1, 1))
    gC = cute.local_tile(mC, tiler=CTA_TILER, coord=tiler_coord, proj=(1, 1, None))

    # Shared memory: the barriers, then the FP16 halves in the swizzled
    # layouts the instruction reads: A with k contiguous, B with n
    # contiguous, which is how each is stored in global memory. The swizzle
    # goes on the pointer and the layout stays plain.
    # sA_hi, sA_lo: (BM, BK, BUFFERS)   sB_hi, sB_lo: (BN, BK, BUFFERS)
    smem = SmemAllocator()
    storage = smem.allocate(SharedStorage)
    a_atom = tcgen05.make_smem_layout_atom(
        sm100_utils.get_smem_layout_atom_ab(
            tcgen05.OperandMajorMode.K, cutlass.Float16, (BM, BK)
        ),
        cutlass.Float16,
    )
    b_atom = tcgen05.make_smem_layout_atom(
        sm100_utils.get_smem_layout_atom_ab(
            tcgen05.OperandMajorMode.MN, cutlass.Float16, (BN, BK)
        ),
        cutlass.Float16,
    )
    a16 = cute.tile_to_shape(a_atom, (BM, BK, BUFFERS), (0, 1, 2))
    b16 = cute.tile_to_shape(b_atom, (BN, BK, BUFFERS), (1, 0, 2))
    sA_hi = smem.allocate_tensor(cutlass.Float16, a16.outer, 1024, swizzle=a16.inner)
    sA_lo = smem.allocate_tensor(cutlass.Float16, a16.outer, 1024, swizzle=a16.inner)
    sB_hi = smem.allocate_tensor(cutlass.Float16, b16.outer, 1024, swizzle=b16.inner)
    sB_lo = smem.allocate_tensor(cutlass.Float16, b16.outer, 1024, swizzle=b16.inner)

    # Tensor memory for the MMA result. Warp 0 allocates it; the barrier
    # lets every thread fetch its address afterwards.
    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf, barrier_for_retrieve=tmem_barrier
    )
    tmem.allocate(TMEM_COLUMNS)

    # The hand-over of the result: one thread issues the MMAs and commits,
    # every thread waits for them, reads, and releases.
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.acc_mbar.data_ptr(),
    ).make_participants()

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

    # Registers: one k-tile of FP32 values and of FP16 halves.
    load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=RUN * 32
    )
    a_shape = tAgA[None, None, None, 0].shape
    b_shape = tBgB[None, None, None, 0].shape
    rA = cute.make_rmem_tensor(a_shape, cutlass.Float32)
    rB = cute.make_rmem_tensor(b_shape, cutlass.Float32)
    rA_hi = cute.make_rmem_tensor(a_shape, cutlass.Float16)
    rA_lo = cute.make_rmem_tensor(a_shape, cutlass.Float16)
    rB_hi = cute.make_rmem_tensor(b_shape, cutlass.Float16)
    rB_lo = cute.make_rmem_tensor(b_shape, cutlass.Float16)

    # The MMA operands are descriptors of shared memory, and its result is a
    # tensor in tensor memory.
    # tCrA_*: (MMA, MMA_M, MMA_K, BUFFERS)   tCrB_*: (MMA, MMA_N, MMA_K, BUFFERS)
    # tCtAcc, tCgC: (MMA, MMA_M, MMA_N)
    thr_mma = mma_add.get_slice(0)
    tCrA_hi = mma_add.make_fragment_A(thr_mma.partition_A(sA_hi))
    tCrA_lo = mma_add.make_fragment_A(thr_mma.partition_A(sA_lo))
    tCrB_hi = mma_add.make_fragment_B(thr_mma.partition_B(sB_hi))
    tCrB_lo = mma_add.make_fragment_B(thr_mma.partition_B(sB_lo))
    tCgC = thr_mma.partition_C(gC)
    tCtAcc = mma_add.make_fragment_C(mma_add.partition_shape_C((BM, BN)))
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    # Tensor memory -> registers. The columns are cut into one share per
    # group of four warps; a thread reads its row of its group's share.
    # tDtC, tDgC: (TMEM_CPY, NUM_TMEM_CPY, shares)
    shares = THREADS // TMEM_THREADS
    share_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // shares),
    )
    tCtAcc_shares = cute.zipped_divide(tCtAcc, share_tiler)
    gC_shares = cute.zipped_divide(tCgC, share_tiler)
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition(BN // shares)), cutlass.Float32
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_shares[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx % TMEM_THREADS)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_shares)
    tDgC = tmem_thr_copy.partition_D(gC_shares)
    share = cute.arch.make_warp_uniform(tidx // TMEM_THREADS)
    # acc: the FP32 sum. part: one k-tile's result as read from tensor memory.
    acc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, cutlass.Float32)
    part = cute.make_rmem_tensor(tDgC[None, None, 0].shape, cutlass.Float32)
    acc.fill(0.0)

    # mma_fresh overwrites the result, mma_add adds to it.
    mma_fresh.set(tcgen05.Field.ACCUMULATE, False)
    mma_add.set(tcgen05.Field.ACCUMULATE, True)

    k_tile_count = cute.size(tAgA, mode=[3])

    # Prologue: k-tile 0 goes into FP16 buffer 0 and k-tile 1 into the
    # registers. Past the last k-tile the index wraps to 0; those values are
    # loaded and split but never multiplied.
    cute.copy(load, tAgA[None, None, None, 0], rA)
    cute.copy(load, tBgB[None, None, None, 0], rB)
    _split(rA, rA_hi, rA_lo, tAsA_hi[None, None, None, 0], tAsA_lo[None, None, None, 0])
    _split(rB, rB_hi, rB_lo, tBsB_hi[None, None, None, 0], tBsB_lo[None, None, None, 0])
    cute.arch.fence_proxy("async.shared", space="cta")
    gmem_next = cutlass.Int32(1)
    if gmem_next >= k_tile_count:
        gmem_next = cutlass.Int32(0)
    cute.copy(load, tAgA[None, None, None, gmem_next], rA)
    cute.copy(load, tBgB[None, None, None, gmem_next], rB)
    gmem_next = gmem_next + 1
    if gmem_next >= k_tile_count:
        gmem_next = cutlass.Int32(0)
    # The FP16 buffer to multiply.
    buf = cutlass.Int32(0)

    for _ in range(k_tile_count):
        # Every thread has written and fenced this k-tile's FP16 halves.
        cute.arch.sync_threads()

        if warp_idx == 0:
            # Passes once every thread has released the previous k-tile, so
            # tensor memory and the other FP16 buffer are free again.
            acc_empty = acc_producer.acquire_and_advance()
            for kb in cutlass.range_constexpr(BK // MMA_K):
                a_hi = tCrA_hi[None, None, kb, buf]
                a_lo = tCrA_lo[None, None, kb, buf]
                b_hi = tCrB_hi[None, None, kb, buf]
                b_lo = tCrB_lo[None, None, kb, buf]
                if cutlass.const_expr(kb == 0):
                    cute.gemm(mma_fresh, tCtAcc, a_hi, b_hi, tCtAcc)
                else:
                    cute.gemm(mma_add, tCtAcc, a_hi, b_hi, tCtAcc)
                cute.gemm(mma_add, tCtAcc, a_hi, b_lo, tCtAcc)
                cute.gemm(mma_add, tCtAcc, a_lo, b_hi, tCtAcc)
            # Signals the waiting threads when these MMAs have finished.
            acc_empty.commit()

        # While the MMAs run: split the next k-tile, which is already in
        # registers, into the other FP16 buffer, and make the writes visible
        # to the instruction. Then load the k-tile after it.
        buf_next = cutlass.Int32(1) - buf
        _split(
            rA,
            rA_hi,
            rA_lo,
            tAsA_hi[None, None, None, buf_next],
            tAsA_lo[None, None, None, buf_next],
        )
        _split(
            rB,
            rB_hi,
            rB_lo,
            tBsB_hi[None, None, None, buf_next],
            tBsB_lo[None, None, None, buf_next],
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.copy(load, tAgA[None, None, None, gmem_next], rA)
        cute.copy(load, tBgB[None, None, None, gmem_next], rB)

        # Wait for this k-tile's MMAs, then move their result from tensor
        # memory into the FP32 sum.
        acc_full = acc_consumer.wait_and_advance()
        cute.copy(tmem_tiled_copy, tDtC[None, None, share], part)
        acc.store(acc.load() + part.load())
        acc_full.release()

        buf = buf_next
        gmem_next = gmem_next + 1
        if gmem_next >= k_tile_count:
            gmem_next = cutlass.Int32(0)

    # Warp 0 will not allocate again.
    tmem.relinquish_alloc_permit()

    # Registers -> gmem.
    cute.autovec_copy(acc, tDgC[None, None, share])

    # Tensor memory is freed once every thread is done with it.
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


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
    # A: (32, 16) threads over (m, runs along k), repeated four times along m.
    tiled_copy_A = cute.make_tiled_copy_tv(
        load,
        cute.make_layout((THREADS // (BK // RUN), BK // RUN), stride=(BK // RUN, 1)),
        cute.make_layout((1, RUN)),
    )
    # B: (32, 16) threads over (runs along n, k), repeated four times along k.
    tiled_copy_B = cute.make_tiled_copy_tv(
        load,
        cute.make_layout((BN // RUN, THREADS // (BN // RUN)), stride=(1, BN // RUN)),
        cute.make_layout((RUN, 1)),
    )

    # The tcgen05 MMA: FP16 operands read from shared memory, FP32 result in
    # tensor memory. Two copies of it, because whether it overwrites or adds
    # to its result is a setting of the object.
    mma_op = tcgen05.MmaF16BF16Op(
        cutlass.Float16,
        cutlass.Float32,
        MMA_SHAPE,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
    )
    mma_fresh = cute.make_tiled_mma(mma_op)
    mma_add = cute.make_tiled_mma(mma_op)

    # Every tile full, and base addresses that allow 128-bit loads.
    addresses = A.iterator.toint() | B.iterator.toint()
    if (M % BM == 0) & (K % BN == 0) & (N % BK == 0) & ((addresses & 15) == 0):
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
        _gemm_tcgen05(
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
