import torch
from functools import partial
import math
import operator
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.utils import LayoutEnum, StaticPersistentTileScheduler
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import make_ptr
import cutlass.pipeline as pipeline

@cute.jit
def tile_scheduler_get_next(cur_id, group_size, group_size_m, max_m):
    group_id = cur_id // group_size
    id_in_group = cur_id % group_size
    first_pid_m = group_id * group_size_m
    real_group_m = min(max_m - first_pid_m, group_size_m)
    row = id_in_group % real_group_m
    col = id_in_group // real_group_m
    return first_pid_m+row, col

@cute.kernel
def gemm_kernel(
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    c_smem_layout: cute.ComposedLayout,
    g_mnk: cute.Shape,
    b_mnk: cute.Shape,
    NUM_SM: cutlass.Constexpr,
    shared_storage: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
    
    cta_rank_in_cluster = cute.arch.make_warp_uniform(
        cute.arch.block_idx_in_cluster()
    )
    M, N, K = g_mnk
    BM, BN, BK = b_mnk
    # for tile swizzle
    tiles_m = (M+BM-1) // BM
    tiles_n = (N+BN-1) // BN
    num_tiles = tiles_m * tiles_n
    print(f"tiles_m={tiles_m}")
    print(f"tiles_n={tiles_n}")
    print(f"num_tiles={num_tiles}")
    group_size_m = 8
    group_size = group_size_m * tiles_n
    
    stages = cute.product(a_smem_layout.shape[2])
    gA = cute.local_tile(
        mA, (BM, BK), (None,None)
    ) # (128,64,ktiles):(1@1,1@0,64@0)
    gB = cute.local_tile(
        mB, (BN, BK), (None,None)
    ) # (256,64,ktiles):(1@0,1@1,64@1)>
    gC = cute.local_tile(
        mC, (BM, BN), (None,None)
    ) # (128,256):(4096,1)
    print(f"gA={gA}")
    print(f"gB={gB}")
    print(f"gC={gC}")

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage)
    sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
    sC = storage.sC.get_tensor(c_smem_layout.outer, swizzle=c_smem_layout.inner)
    sA_tma = cute.group_modes(sA, 0, 2)
    gA_tma = cute.group_modes(gA, 0, 2)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        cta_coord=0,
        cta_layout=cute.make_layout((1,)),
        smem_tensor=sA_tma,
        gmem_tensor=gA_tma,
    )
    sB_tma = cute.group_modes(sB, 0, 2)
    gB_tma = cute.group_modes(gB, 0, 2)
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        cta_coord=cta_rank_in_cluster,
        cta_layout=cute.make_layout((2,)),
        smem_tensor=sB_tma,
        gmem_tensor=gB_tma,
    )
    tma_sC, tma_gC = cute.nvgpu.cpasync.tma_partition(
        tma_atom_c,
        cta_coord=0,
        cta_layout=cute.make_layout((1,)),
        smem_tensor=cute.group_modes(sC, 0, 2),
        gmem_tensor=cute.group_modes(gC, 0, 2),
    )
    barrier_ptr = storage.barrier_ptr.data_ptr()
    empty_ptr = barrier_ptr+stages
    with cute.arch.elect_one():
        for i in cutlass.range_constexpr(stages):
            cute.arch.mbarrier_init(barrier_ptr+i, cnt=1)
            cute.arch.mbarrier_init(empty_ptr+i, cnt=2*2) # two consumer
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()
    
    tma_copy_bytes = cute.size_in_bytes(Float16, cute.slice_(a_smem_layout, (None, None, 0))) \
        + cute.size_in_bytes(Float16, cute.slice_(b_smem_layout, (None, None, 0)))
    cute.arch.cluster_wait()
    is_producer = warp_idx < 4
    if is_producer:
        cute.arch.warpgroup_reg_dealloc(24)
        if warp_idx == 0:
            phase = 1
            smem_k = 0
            tile_id = bx
            while tile_id < num_tiles:
                bm, bn = tile_scheduler_get_next(tile_id, group_size, group_size_m, tiles_m)
                tile_id += NUM_SM
                # cute.printf("tile_id={} bm={} bn={}", tile_id, bm, bn)
                for k in range(gA.shape[3]):
                    if smem_k == stages:
                        smem_k = 0
                        phase ^= 1
                    cute.arch.mbarrier_wait(empty_ptr+smem_k, phase)
                    cur_barrier = barrier_ptr+smem_k
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(cur_barrier, tma_copy_bytes)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, bm, k)],
                        tAsA[(None, smem_k)],
                        tma_bar_ptr=cur_barrier,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, bn, k)],
                        tBsB[(None, smem_k)],
                        tma_bar_ptr=cur_barrier,
                        mcast_mask=cutlass.Int16(3),
                    )
                    # if tx == 0 and bx == 0 and by == 0:
                    #     cute.printf("PRODUCER: TMA copy issued. k={}", k)
                    smem_k += 1
    else:
        cute.arch.warpgroup_reg_alloc(240)
        tid_in_wg = tx % 128
        thr_mma = tiled_mma.get_slice(tx-128)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCsC = thr_mma.partition_C(sC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc_shape = tiled_mma.partition_shape_C(
            (BM, BN)
        )
        tCrC = cute.make_fragment(acc_shape, Float32)
        print(f"tCgC={tCgC}")
        print(f"tCsC={tCsC}")
        print(f"tCrC={tCrC}")
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        phase = 0
        smem_k = 0
        # for tile_id in cutlass.range(bx, num_tiles, 78):
        tile_id = bx
        while tile_id < num_tiles:
            tCrC.fill(0.0)
            bm, bn = tile_scheduler_get_next(tile_id, group_size, group_size_m, tiles_m)
            # if bx==0 and tx==128:
            #     cute.printf("tile_id={}, group_size={}, group_size_m={} tiles_m={}", tile_id, group_size, group_size_m, tiles_m)
            #     cute.printf("tile_id={}, bm={}, bn={} num_tiles={}", tile_id, bm, bn, num_tiles)
            tile_id += NUM_SM
            for k in range(gA.shape[3]):
                if smem_k == 3:
                    smem_k = 0
                    phase ^= 1
                cur_barrier = barrier_ptr + smem_k
                cute.arch.mbarrier_wait(cur_barrier, phase)
                cur_stage_coord = (
                    None,
                    None,
                    None,      # tile_mma k
                    smem_k,    # pipeline stage
                )
                cute.nvgpu.warpgroup.fence()
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[cur_stage_coord],
                    tCrB[cur_stage_coord],
                    tCrC,
                )
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                if tid_in_wg < 2:
                    cute.arch.mbarrier_arrive(empty_ptr+smem_k, tid_in_wg)
                smem_k += 1

            tCrC_dtype = cute.make_fragment_like(tCrC, Float16)
            tCrC_dtype.store(tCrC.load().to(Float16))
            # cute.autovec_copy(tCrC_dtype, tCgC[(None,None,None, bm, bn)])
            cute.arch.cp_async_wait_group(0)
            r2s_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), Float16,
            )
            tiled_copy_c = cute.make_tiled_copy_C(r2s_atom, tiled_mma)
            thr_copy_c = tiled_copy_c.get_slice(tx-128)
            tXrC = thr_copy_c.retile(tCrC_dtype)
            tXsC = thr_copy_c.partition_D(sC)
            cute.copy(r2s_atom, tXrC, tXsC)
            # cute.autovec_copy(tCrC_dtype, tCsC)
            print(f"tma_gC={tma_gC}")
            print(f"tma_sC={tma_sC}")
            cute.arch.barrier(barrier_id=10, number_of_threads=256)
            if tx == 128:
                cute.copy(
                    tma_atom_c,
                    tma_sC,
                    tma_gC[(None, bm, bn)],
                )
                cute.arch.cp_async_commit_group()

        # if tx == 0 and bx==0 and by==0:
        #     cute.printf("tCrC={}", tCrC)
        #     cute.printf("tCrC_D={}", tCrC_dtype))
        # c_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        # cute.copy(c_copy_atom, tCrC_dtype, tCgC)


@cute.jit
def gemm_tn(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    k: cutlass.Constexpr,
    NUM_SM: cutlass.Constexpr,
):
    a_layout = cute.make_ordered_layout((m, k), order=(1, 0))
    b_layout = cute.make_ordered_layout((n, k), order=(0, 1))
    c_layout = cute.make_ordered_layout((m, n), order=(1, 0))
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    mB = cute.make_tensor(b_ptr, layout=b_layout)
    mC = cute.make_tensor(c_ptr, layout=c_layout)
    a_dtype = a_ptr.dtype
    b_dtype = b_ptr.dtype
    a_smem_layout_atom = warpgroup.make_smem_layout_atom(
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
        a_dtype,
    )
    b_smem_layout_atom = warpgroup.make_smem_layout_atom(
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128,
        b_dtype,
    )
    stages = 3
    BM, BN, BK = 128, 256, 64
    a_smem_layout = cute.tile_to_shape(
        a_smem_layout_atom,
        (BM, BK, stages),
        order=(0, 1, 2),
    )
    b_smem_layout = cute.tile_to_shape(
        b_smem_layout_atom,
        (BN, BK, stages),
        order=(1, 0, 2),
    )
    c_smem_layout = cute.tile_to_shape(
        a_smem_layout_atom,
        (BM, BN),
        order=(0, 1),
    )
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
        Float32,
        atom_layout_mnk=(2,1,1),
        tiler_mn=(64, BN),
    )
    buffer_align_bytes = 1024
    @cute.struct
    class SharedStorage:
        barrier_ptr: cute.struct.MemRange[
            cutlass.Int64, stages*2
        ]
        sA: cute.struct.Align[
            cute.struct.MemRange[
                a_dtype, cute.cosize(a_smem_layout)
            ],
            buffer_align_bytes,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[
                b_dtype, cute.cosize(b_smem_layout)
            ],
            buffer_align_bytes,
        ]
        sC: cute.struct.Align[
            cute.struct.MemRange[
                a_dtype, cute.cosize(c_smem_layout)
            ],
            buffer_align_bytes,
        ]
    tma_atom_A, tma_tensor_A = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mA,
        cute.slice_(a_smem_layout, (None,None,0)),
        (BM, BK),
        num_multicast=1,
    )
    tma_atom_B, tma_tensor_B = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(),
        mB,
        cute.slice_(b_smem_layout, (None,None,0)),
        (BN, BK),
        num_multicast=2,
    )
    tma_atom_C, tma_tensor_C = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        mC,
        c_smem_layout,
        (BM, BN),
        num_multicast=1,
    )
    gemm_kernel(
        tma_atom_A, tma_tensor_A,
        tma_atom_B, tma_tensor_B,
        tma_atom_C, tma_tensor_C,
        tiled_mma,
        a_smem_layout,
        b_smem_layout,
        c_smem_layout,
        (m, n, k),
        (BM, BN, BK),
        NUM_SM,
        SharedStorage,
    ).launch(
        # grid=[1, 1, 1],
        grid=[NUM_SM, 1, 1],
        block=[128*3, 1, 1],
        cluster=[2, 1, 1],
        # smem=get_smem_size_bytes(tiler_mn, num_warps),
        # no cluster for now
    )

props = torch.cuda.get_device_properties()
multi_processor_count = props.multi_processor_count
m, n, k = 4096, 4096, 4096
torch.manual_seed(22)
a = torch.randn(m, k, device="cuda", dtype=torch.float16) / 64
b = torch.randn(k, n, device="cuda", dtype=torch.float16)
c = torch.zeros(m, n, device="cuda", dtype=torch.float16)
a_ptr = make_ptr(
    cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
b_ptr = make_ptr(
    cutlass.Float16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
c_ptr = make_ptr(
    cutlass.Float16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
ref_c = a @ b

compiled_gemm = cute.compile(gemm_tn, a_ptr, b_ptr, c_ptr, m,n,k,multi_processor_count)
compiled_gemm(a_ptr, b_ptr, c_ptr)

# torch.set_printoptions(sci_mode=False, threshold=float('inf'), linewidth=999999)
torch.testing.assert_close(c, ref_c, atol=1e-2, rtol=1e-2)

# warm up
for i in range(32):
    ref_c = a @ b

def get_flops(func):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for i in range(32):
        ref_c = a @ b
    start_event.record()
    for _ in range(64):
        func()
    end_event.record()
    torch.cuda.synchronize()

    total_flop = m*n*k*2
    time_ms = start_event.elapsed_time(end_event)
    per_ms = time_ms / 64
    return total_flop / (per_ms / 1000)

print(f"cublas FLOPS: {get_flops(lambda: torch.mm(a, b)) / 1e12:.3f} TFLOPS")
import time
time.sleep(2)
print(f"cuteDSL FLOPS: {get_flops(lambda: compiled_gemm(a_ptr, b_ptr, c_ptr)) / 1e12:.3f} TFLOPS")
