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
def demokernel(
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    # a_smem_layout: cute.ComposedLayout,
    a_smem_layout: cute.Layout,
    shared_storage: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)

    bx, by, bz = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage)
    # outer is layout, inner is swizzle e.g S<3,4,3>
    # print(f"outer={a_smem_layout.outer} inner={a_smem_layout.inner}")
    # sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sA = storage.sA.get_tensor(a_smem_layout)
    # gA = cute.local_tile(
    #     mA, cta_tile_shape_mnk, tile_coord_mnk, proj=(1, None, 1)
    # )
    gA = mA[(None, None, 0)]
    print(f"sA={sA}")
    print(f"gA={gA}")
    gA_div = cute.flat_divide(gA, (64, 256))
    print(f"gA_div={gA_div}")
    sA_tma = cute.group_modes(sA, 0, 2)
    gA_tma = cute.group_modes(gA_div[(None,None,0,0)], 0, 2)
    # gA_tma = cute.group_modes(gA, 0, 2)
    print(f"sA_tma={sA_tma}")
    print(f"gA_tma={gA_tma}")
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        cta_coord=0,
        cta_layout=cute.make_layout((1,)),
        smem_tensor=sA_tma,
        gmem_tensor=gA_tma,
    )
    print(f"tAsA={tAsA} {cute.rank(tAsA)}")
    print(f"tAgA={tAgA} {cute.rank(tAgA)}")
    mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(mainloop_pipeline_array_ptr, cnt=1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    tma_copy_bytes = cute.size_in_bytes(Float16, a_smem_layout)
    print(f"tma_copy_bytes={tma_copy_bytes}")
    # cute.printf("tx={}", tx)

    cute.arch.sync_threads()
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mainloop_pipeline_array_ptr, tma_copy_bytes)
        cute.copy(
            tma_atom_a,
            tAgA,
            tAsA,
            tma_bar_ptr=mainloop_pipeline_array_ptr,
        )

    cute.arch.mbarrier_wait(mainloop_pipeline_array_ptr, 0)
    if tx == 0:
        # cute.printf("tAsA={}", tAsA)
        cute.print_tensor(sA)

@cute.jit
def demof(
    a_ptr: cute.Pointer,
    qh: cutlass.Constexpr,
    hd: cutlass.Constexpr,
    bs: cutlass.Constexpr,
):
    a_layout = cute.make_ordered_layout((qh, hd, bs), order=(1, 0, 2))
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    print(f"mA={mA}")
    a_dtype = Float16
    # only get K/MN major here
    # a_layout = LayoutEnum.from_tensor(mA)
    # a_smem_layout_atom = warpgroup.make_smem_layout_atom(
    #     cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
    #     a_dtype,
    # )
    # a_smem_layout = cute.tile_to_shape(
    #     a_smem_layout_atom,
    #     (128, 64, 1),
    #     order=(0, 1, 2),
    # )
    a_smem_layout = cute.make_ordered_layout((64, hd), order=(1, 0))
    print(f"a_smem_layout={a_smem_layout}")
    buffer_align_bytes = 1024
    @cute.struct
    class SharedStorage:
        mainloop_pipeline_array_ptr: cute.struct.MemRange[
            cutlass.Int64, 2
        ]
        sA: cute.struct.Align[
            cute.struct.MemRange[
                a_dtype, cute.cosize(a_smem_layout)
            ],
            buffer_align_bytes,
        ]
    tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    # tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp() for cluster
    tma_atom_A, tma_tensor_A = cute.nvgpu.cpasync.make_tiled_tma_atom(
        tma_op,
        mA,
        a_smem_layout,
        (64, hd),
        num_multicast=1,
    )
    demokernel(
        tma_atom_A, tma_tensor_A,
        a_smem_layout,
        SharedStorage,
    ).launch(
        grid=[1, 1, 1],
        block=[128, 1, 1],
        # smem=get_smem_size_bytes(tiler_mn, num_warps),
        # no cluster for now
    )

m, k = 256, 256
a = torch.randn(32, 256, 6, device="cuda", dtype=torch.float16)
a_ptr = make_ptr(
    cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
# a_ = from_dlpack(a, assumed_align=16)
# b_ = from_dlpack(b, assumed_align=16)
demof(a_ptr, 32, 256, 6)
