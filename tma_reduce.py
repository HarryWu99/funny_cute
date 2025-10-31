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

@cute.kernel
def tma_reduce_kernel(
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    a_smem_layout: cute.Layout,
    shared_storage: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
    
    BLOCK_SIZE = 64
    gA = cute.local_tile(
        mA, (BLOCK_SIZE, BLOCK_SIZE), (bx, by)
    )
    gB = cute.local_tile(
        mB, (BLOCK_SIZE, BLOCK_SIZE), (bx, by)
    )
    
    print(f"gA={gA}")
    print(f"gB={gB}")
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage)
    sA = storage.sA.get_tensor(a_smem_layout)
    sA_tma = cute.group_modes(sA, 0, 2)
    gA_tma = cute.group_modes(gA, 0, 2)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        cta_coord=0,
        cta_layout=cute.make_layout((1,)),
        smem_tensor=sA_tma,
        gmem_tensor=gA_tma,
    )
    tBsA, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        cta_coord=0,
        cta_layout=cute.make_layout((1,)),
        smem_tensor=cute.group_modes(sA, 0, 2),
        gmem_tensor=cute.group_modes(gB, 0, 2),
    )
    barrier_ptr = storage.barrier_ptr.data_ptr()
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(barrier_ptr, cnt=1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    tma_copy_bytes = cute.size_in_bytes(Float16, a_smem_layout)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barrier_ptr, tma_copy_bytes)
        cute.copy(
            tma_atom_a,
            tAgA,
            tAsA,
            tma_bar_ptr=barrier_ptr,
        )
    cute.arch.mbarrier_wait(barrier_ptr, 0)
    if warp_idx == 0:
        cute.copy(
            tma_atom_b,
            tBsA,
            tBgB,
        )
    

@cute.jit
def tma_reduce_launch(
    mA: cute.Tensor,
    mB: cute.Tensor
):
    m, n = mA.shape
    buffer_align_bytes = 1024
    a_dtype = Float16
    b_dtype = Float16
    a_smem_layout = cute.make_ordered_layout((64, 64), (1, 0))
    b_smem_layout = cute.make_ordered_layout((64, 64), (1, 0))
    @cute.struct
    class SharedStorage:
        barrier_ptr: cute.struct.MemRange[
            cutlass.Int64, 2
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
        a_smem_layout,
        (64, 64),
        num_multicast=1,
    )
    tma_atom_B, tma_tensor_B = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(),
        mB,
        b_smem_layout,
        (64, 64),
        num_multicast=1,
    )
    tma_reduce_kernel(
        tma_atom_A,
        tma_tensor_A,
        tma_atom_B,
        tma_tensor_B,
        a_smem_layout,
        SharedStorage,
    ).launch(
        grid=[m//64, n//64, 1],
        block=[1, 1, 1],
        # smem=get_smem_size_bytes(tiler_mn, num_warps),
        # no cluster for now
    )

m, n, k = 4096, 4096, 4096
torch.manual_seed(22)
a = torch.randn(m, n, device="cuda", dtype=torch.float16)
b = torch.randn(m, n, device="cuda", dtype=torch.float16)
ori_b = torch.empty_like(b)
ori_b.copy_(b)

a_, b_ = from_dlpack(a, assumed_align=32), from_dlpack(b, assumed_align=32)
compiled_add = cute.compile(tma_reduce_launch, a_, b_)
compiled_add(a_, b_)
# tma_reduce_launch(from_dlpack(a, assumed_align=32), from_dlpack(b, assumed_align=32))

torch.testing.assert_close(b, a+ori_b, rtol=1e-2, atol=1e-2)

for i in range(32):
    ref_c = a @ b

def get_bandwidth(func):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(64):
        func()
    end_event.record()
    torch.cuda.synchronize()

    total_bytes = m*n*2*2
    time_ms = start_event.elapsed_time(end_event)
    per_ms = time_ms / 64
    return total_bytes / (per_ms / 1000)

print(f"torch add: {get_bandwidth(lambda: b.add_(a)) / 1024**2:.3f} MB/s")
print(f"cuteDSL add: {get_bandwidth(lambda: compiled_add(a, b)) / 1024**2:.3f} MB/s")
