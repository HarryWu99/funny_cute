import torch
from functools import partial
import math
import operator
from typing import Callable, Optional, List, Tuple

import cutlass
from cutlass import const_expr, Float16, Float32, Int32, Int64, Boolean, Int8
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu.warpgroup import (
    OperandMajorMode,
    OperandSource,
    make_smem_layout_atom,
)
import cuda.bindings.driver as cuda

@cute.struct
class MLAParams:
    bs: Int32
    s_q: Int32
    pagesize: Int32

@cute.kernel
def kernel(
    params: MLAParams
):
    cute.printf("bs={}, s_q={}", params.bs, params.s_q)

class MLA_KVCACHE_FWD:
    def __init__(self):
        self.bs=1
        self.sq = 32
        self.pagesize = 64
    
    @cute.jit    
    def launch(
        self,
        optr: cute.Pointer,
    ):
        self.page_block_size = 64
        self.head_dim_k = 576
        self.head_dim_v = 512
        smem_layout_atom = make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
            Float16,
        )
        k_smem_layout = cute.tile_to_shape(
            smem_layout_atom,
            (self.page_block_size, self.head_dim_k),
            order=(0,1)
        )
        # transpose of k_smem_layout
        v_smem_layout = cute.composition(
            k_smem_layout,
            cute.make_ordered_layout((self.head_dim_v, self.page_block_size), order=(1, 0))
        )
        
        tiled_mma_pv_localP = sm90_utils.make_trivial_tiled_mma(
            Float16,
            Float16,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1,1,1),
            tiler_mn=(64, self.head_dim_v//2),
            a_source=OperandSource.RMEM,
        )
        print(f"k_smem_layout={k_smem_layout}")
        print(f"v_smem_layout={v_smem_layout}")
        # self.mla_fwd(optr).launch(
        #     grid=[1,1,1],
        #     block=[1, 1, 1],
        #     cluster=[1, 1, 1],
        # )
        tiled_mma_tmp = sm90_utils.make_trivial_tiled_mma(
            Float16,
            Float16,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1,1,1),
            tiler_mn=(64, 64),
            a_source=OperandSource.SMEM,
        )
        self.mla_combine(tiled_mma_tmp).launch(
            grid=[1,1,1],
            block=[256, 1, 1],
            cluster=[1, 1, 1],
        )

    @cute.kernel
    def mla_fwd(
        self,
        optr: cute.Pointer,
    ):
        tiled_mma_pv_localP = sm90_utils.make_trivial_tiled_mma(
            Float16,
            Float16,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1,1,1),
            tiler_mn=(64, 256),
            a_source=OperandSource.RMEM,
        )
        rO = tiled_mma_pv_localP.make_fragment_C(
            tiled_mma_pv_localP.partition_shape_C((64, 256)),
        )
        cute.printf("rO={}", rO)
        
        cur_phase = Int32(0)
        cur_phase_t = cute.make_rmem_tensor((1, ), Boolean)
        cur_phase_t.fill(0)
        # cur_phase_t[0] = True
        
        # cute.printf("cur_phase_t={}", cur_phase_t)
        # self.mutable_func(cur_phase_t)

        smem = cutlass.utils.SmemAllocator()
        barriers = smem.allocate_array(Int64, 2)
        cute.arch.mbarrier_init(barriers, cnt=1)
        cute.arch.mbarrier_init_fence()

        cute.arch.mbarrier_arrive_and_expect_tx(barriers, 0)
        cute.arch.mbarrier_wait(barriers, cur_phase_t[0])

        rP = cute.make_rmem_tensor((1, ), Float32)
        rP[0] = -float("inf")
        cute.printf("rP = {}", cute.exp2(rP[0] * 0.2 - 3))

        cute.printf("shunxu = {} {}", 32//4 *16, 15//4 *16)

        cute.printf("acc shape={}", tiled_mma_pv_localP.partition_shape_C((64, 256)))
        cute.printf("optr={}", optr)
        cute.printf("optr={}", optr + 2*4)
        # cute.printf("bs={}, s_q={}", self.bs, self.sq)

    @cute.jit
    def mutable_func(
        self,
        t,
    ):
        t[0] ^= True
        cute.printf("mutable_func: t={}", t[0])

    @cute.kernel
    def mla_combine(
        self,
        tiled_mma: cute.TiledMma
    ):
        tid, _, _ = cute.arch.thread_idx()
        if tid==0:
            cute.printf("combine, pagesize={}", self.pagesize)
        smem = cutlass.utils.SmemAllocator()
        smem_layout_atom = make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
            Float16,
        )
        layout = cute.tile_to_shape(smem_layout_atom, (64,128), order=(0,1))
        # sA = smem.allocate_tensor(Float16, cute.make_ordered_layout((64,128), order=(1,0)))
        # sB = smem.allocate_tensor(Float16, cute.make_ordered_layout((64,128), order=(1,0)))
        sA = smem.allocate_tensor(Float16, layout.outer, swizzle=layout.inner)
        sB = smem.allocate_tensor(Float16, layout.outer, swizzle=layout.inner)
        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        print(f"sA={sA} tCsA={tCsA}")
        tCrA = thr_mma.make_fragment_A(tCsA)
        tCrB = thr_mma.make_fragment_B(tCsB)
        tCrC = thr_mma.make_fragment_C(thr_mma.partition_shape_C((64, 64)))
        if tid < 128:
            self.inner_func(tiled_mma, tCrA, tCrB, tCrC)
            if tid==0:
                cute.printf("tCrC={}", tCrC)
        else:
            wgparams = (tiled_mma, tCrC)
            self.inner_func2(tiled_mma, tCrA, tCrB, tCrC)

    @cute.jit
    def inner_func(
        self,
        tiled_mma: cute.TiledMma,
        tCrA,
        tCrB,
        tCrC,
    ):
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
    
    @cute.jit
    def inner_func2(
        self,
        tiled_mma: cute.TiledMma,
        tCrA,
        tCrB,
        tCrC,
    ):
        cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

mla_obj = MLA_KVCACHE_FWD()
o = torch.randn(3,4)
mla_obj.launch(make_ptr(Float32, o.data_ptr(), cute.AddressSpace.gmem, assumed_align=16))

