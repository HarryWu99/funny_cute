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

@cute.kernel
def mykernel(
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    a_smem_layout: cute.Layout, # or cute.ComposedLayout
    m: int, n: int,
    shared_storage: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
        
    is_producer = warp_idx < 4
    gA = cute.local_tile(
        mA, (64, 128), (bx, None)
    )
    print(f"gA={gA}")
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage)
    sA = storage.sA.get_tensor(a_smem_layout)
    barrier_ptr = storage.barrier_ptr.data_ptr()
    empty_ptr = barrier_ptr+3
    with cute.arch.elect_one():
        for i in cutlass.range_constexpr(3):
            cute.arch.mbarrier_init(barrier_ptr+i, cnt=1)
            cute.arch.mbarrier_init(empty_ptr+i, cnt=128)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    tma_copy_bytes = cute.size_in_bytes(Float16, cute.slice_(a_smem_layout, (None,None,0)))
    if is_producer:
        if warp_idx == 0:
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                tma_atom_a,
                cta_coord=0,
                cta_layout=cute.make_layout((1,)),
                smem_tensor=cute.group_modes(sA, 0, 2),
                gmem_tensor=cute.group_modes(gA, 0, 2),
            )
            print(f"tAsA={tAsA}")
            print(f"tAgA={tAgA}")
            phase = 1
            smem_k = 0
            for k in range(gA.shape[2]):
                if smem_k == 3:
                    smem_k = 0
                    phase ^= 1
                cute.arch.mbarrier_wait(empty_ptr+smem_k, phase)
                cur_barrier = barrier_ptr+smem_k
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(cur_barrier, tma_copy_bytes)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, k)],
                    tAsA[(None, smem_k)],
                    tma_bar_ptr=cur_barrier,
                )
                if tx == 0 and bx == 0 and by == 0:
                    cute.printf("PRODUCER: TMA copy issued. k={}", k)
                smem_k += 1
    else:
        if tx == 128 and bx == 0 and by == 0:
            cute.printf("CONSUMER: start waiting")
        phase = 0
        smem_k = 0
        for k in range(gA.shape[2]):
            if smem_k == 3:
                smem_k = 0
                phase ^= 1
            cur_barrier = barrier_ptr + smem_k
            cute.arch.mbarrier_wait(cur_barrier, phase)
            if tx == 128 and bx == 0 and by == 0:
                cute.printf("CONSUMER: k={} TMA load finished.\nTile in SMEM {}", k, sA[(None,None,smem_k)])
            cute.arch.mbarrier_arrive(empty_ptr+smem_k)
            smem_k += 1


@cute.jit
def launch_kernel(
    mA: cute.Tensor,
):
    a_dtype = Float16
    m, n = mA.shape
    tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    # tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp() for cluster
    stages = 3
    buffer_align_bytes = 1024
    a_smem_layout = cute.make_ordered_layout((64, 128, stages), order=(1, 0, 2))
    @cute.struct
    class SharedStorage:
        barrier_ptr: cute.struct.MemRange[
            cutlass.Int64, 2*stages
        ]
        sA: cute.struct.Align[
            cute.struct.MemRange[
                a_dtype, cute.cosize(a_smem_layout)
            ],
            buffer_align_bytes,
        ]
    tma_atom_A, tma_tensor_A = cute.nvgpu.cpasync.make_tiled_tma_atom(
        tma_op,
        mA,
        cute.slice_(a_smem_layout, (None,None,0)),
        (64, 128),
        num_multicast=1,
    )
    mykernel(
        tma_atom_A, tma_tensor_A,
        a_smem_layout,
        m, n,
        SharedStorage,
    ).launch(
        grid=[1, 1, 1],
        block=[256, 1, 1],
    )


m, n = 4096, 1024
a = torch.randn(m, n, device="cuda", dtype=torch.float16)
a_ptr = make_ptr(
    cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
a_ = from_dlpack(a, assumed_align=32)
# b_ = from_dlpack(b, assumed_align=16)
launch_kernel(a_)
