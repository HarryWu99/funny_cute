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
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import make_ptr
import cutlass.pipeline as pipeline


@cute.kernel
def gemm_kernel(
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    shared_storage: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage)
    sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
    cta_tile_shape_mnk = (64, 64, 64)
    tile_coord_mnk = (by, bx, None)
    gA = cute.local_tile(
        mA, cta_tile_shape_mnk, tile_coord_mnk, proj=(1, None, 1)
    )
    gB = cute.local_tile(
        mB, cta_tile_shape_mnk, tile_coord_mnk, proj=(None, 1, 1)
    )
    sA_tma = cute.group_modes(sA, 0, 2)
    gA_tma = cute.group_modes(gA, 0, 2)
    print(f"after group sA={sA_tma} gA={gA_tma}")
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
        cta_coord=0,
        cta_layout=cute.make_layout((1,)),
        smem_tensor=sB_tma,
        gmem_tensor=gB_tma,
    )
    mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(mainloop_pipeline_array_ptr, cnt=1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    tma_copy_bytes = cute.size_in_bytes(Float16, cute.slice_(a_smem_layout, (None, None, 0))) \
        + cute.size_in_bytes(Float16, cute.slice_(b_smem_layout, (None, None, 0)))
    
    cute.arch.sync_threads()
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mainloop_pipeline_array_ptr, tma_copy_bytes)
        cute.copy(
            tma_atom_a,
            tAgA[(None, 0)],
            tAsA[(None, 0)],
            tma_bar_ptr=mainloop_pipeline_array_ptr,
        )
        cute.copy(
            tma_atom_b,
            tBgB[(None, 0)],
            tBsB[(None, 0)],
            tma_bar_ptr=mainloop_pipeline_array_ptr,
        )
        # todo tma_atom_b copy

    cute.arch.mbarrier_wait(mainloop_pipeline_array_ptr, 0)
    # if tx == 0:
    #     cute.printf("sA={}", tAsA)
    thr_mma = tiled_mma.get_slice(0)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    acc_shape = tiled_mma.partition_shape_C(
        cute.select(cta_tile_shape_mnk, mode=[0, 1])
    )
    rAcc = cute.make_fragment(acc_shape, Float32)
    rAcc.fill(0.0)
    num_k_blocks = cute.size(tCrA, mode=[2])
    print(f"tCrA={tCrA}")
    print(f"tCrB={tCrB}")
    cute.nvgpu.warpgroup.fence()
    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
        k_block_coord = (
            None,
            None,
            k_block_idx,
            0,
        )
        tCrA_1 = tCrA[k_block_coord]
        tCrB_1 = tCrB[k_block_coord]
        print(f"tCrA_1={tCrA_1}")
        print(f"tCrB_1={tCrB_1}")
        cute.gemm(
            tiled_mma,
            rAcc,
            tCrA_1,
            tCrB_1,
            rAcc,
        )
    cute.nvgpu.warpgroup.commit_group()
    cute.nvgpu.warpgroup.wait_group(0)
    if tx == 0 and bx==0 and by==0:
        cute.printf("rAcc={}", rAcc)


@cute.jit
def gemm_tn(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    k: cutlass.Constexpr,
):
    a_layout = cute.make_ordered_layout((m, k), order=(1, 0))
    b_layout = cute.make_ordered_layout((n, k), order=(0, 1))
    c_layout = cute.make_ordered_layout((m, n), order=(1, 0))
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    mB = cute.make_tensor(b_ptr, layout=b_layout)
    mC = cute.make_tensor(c_ptr, layout=c_layout)
    a_dtype = Float16
    b_dtype = Float16
    # only get K/MN major here
    a_layout_enum = LayoutEnum.from_tensor(mA)
    b_layout_enum = LayoutEnum.from_tensor(mB)
    a_smem_layout_atom = warpgroup.make_smem_layout_atom(
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
        a_dtype,
    )
    b_smem_layout_atom = warpgroup.make_smem_layout_atom(
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128,
        b_dtype,
    )
    stages = 3
    a_smem_layout = cute.tile_to_shape(
        a_smem_layout_atom,
        (64, 64, stages),
        order=(0, 1, 2),
    )
    b_smem_layout = cute.tile_to_shape(
        b_smem_layout_atom,
        (64, 64, stages),
        order=(1, 0, 2),
    )
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
        Float32,
        atom_layout_mnk=(1,1,1),
        tiler_mn=(64, 64),
    )
    buffer_align_bytes = 1024
    @cute.struct
    class SharedStorage:
        mainloop_pipeline_array_ptr: cute.struct.MemRange[
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
    tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    # tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp() for cluster
    tma_atom_A, tma_tensor_A = cute.nvgpu.cpasync.make_tiled_tma_atom(
        tma_op,
        mA,
        cute.slice_(a_smem_layout, (None,None,0)),
        (64, 64),
        num_multicast=1,
    )
    tma_atom_B, tma_tensor_B = cute.nvgpu.cpasync.make_tiled_tma_atom(
        tma_op,
        mB,
        cute.slice_(b_smem_layout, (None,None,0)),
        (64, 64),
        num_multicast=1,
    )
    gemm_kernel(
        tma_atom_A, tma_tensor_A,
        tma_atom_B, tma_tensor_B,
        mC,
        tiled_mma,
        a_smem_layout,
        b_smem_layout,
        SharedStorage,
    ).launch(
        # grid=[1, 1, 1],
        grid=[m//64, n//64, 1],
        block=[128, 1, 1],
        # smem=get_smem_size_bytes(tiler_mn, num_warps),
        # no cluster for now
    )

m, n, k = 4096, 4096, 4096
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16) / 64
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
# a@b
c = torch.zeros(4096, 4096, device="cuda", dtype=torch.float16)
a_ptr = make_ptr(
    cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
b_ptr = make_ptr(
    cutlass.Float16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
c_ptr = make_ptr(
    cutlass.Float16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
gemm_tn(a_ptr, b_ptr, c_ptr, m,n,k)
