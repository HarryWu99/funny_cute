import torch
import math
from typing import Callable, Optional, List, Tuple
from enum import IntEnum

import cutlass
from cutlass import const_expr, BFloat16, Float16, Float32, Int32, Int64, Boolean
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu.warpgroup import (
    OperandMajorMode,
    OperandSource,
    make_smem_layout_atom,
)

class NamedBarriers(IntEnum):
    sScale0Ready = 0
    sScale1Ready = 1
    sP0Ready = 2
    rO1sP0sV0RIssued = 3
    sMInitialized = 4

M_LOG2E = 1.4426950408889634074

@dsl_user_op
def sm90_bulk_copy_s2g(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: Int32,
    *,
    loc=None,
    ip=None,
):
    smem_ptr_i32 = smem_ptr.toint().ir_value()
    gmem_ptr_int = gmem_ptr.toint().ir_value()
    bytes_ir = store_bytes.ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [gmem_ptr_int, smem_ptr_i32, bytes_ir],
        f"cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        f"l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

class MLA_KVCACHE_FWD:
    def __init__(
        self,
        batch_size: int,
        seqlen_q_ori: int,
        q_seq_per_hk: int,
        num_heads_q: int,
        num_heads_k: int,
        num_blocks: int,
        page_block_size: int,
        num_q_heads_per_hk: int,
        is_causal: bool,
        head_size_k: int,
        head_size_v: int,
        softmax_scale: float,
    ):
        # Basic dimensions
        self.b = batch_size
        self.s_q = seqlen_q_ori
        self.q_seq_per_hk = q_seq_per_hk
        self.h_q = num_heads_q
        self.h_k = num_heads_k
        self.num_blocks = num_blocks
        self.q_head_per_hk = num_q_heads_per_hk
        
        # Attention parameters
        self.is_causal = is_causal
        self.d = head_size_k
        self.d_v = head_size_v
        self.scale_softmax = softmax_scale
        self.scale_softmax_log2 = softmax_scale * math.log2(math.e)
        
        self.page_block_size = page_block_size
        
        # Const
        self.block_size_m = 64
        self.PAGE_SIZE = 64
        self.head_dim_k = 576
        self.head_dim_v = 512

        self.max_init_val_sm = -1e30
        self.max_init_val = -1e33

    @cute.jit
    def launch(
        self,
        q_ptr: cute.Pointer,
        kcache_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        softmax_lse: cute.Tensor,
        seqlens_k: cute.Tensor,
        block_table: cute.Tensor,
        tile_scheduler_metadata: cute.Tensor,
        num_splits: cute.Tensor,
        softmax_lse_accum_ptr: cute.Pointer,
        out_accum_ptr: cute.Pointer,
    ):
        self.num_sm_parts = tile_scheduler_metadata.shape[0]
        self.total_num_splits = self.b + self.num_sm_parts
        q_dtype = self.q_dtype = q_ptr.dtype

        q = cute.make_tensor(
            q_ptr, cute.make_ordered_layout(
                (self.q_seq_per_hk, self.d, self.h_k, self.b),
                (2, 0, 1, 3)
            )
        )
        kcache = cute.make_tensor(
            kcache_ptr, cute.make_ordered_layout(
                (self.PAGE_SIZE, self.d, self.h_k, self.num_blocks),
                (2, 0, 1, 3)
            )
        )
        out = cute.make_tensor(
            out_ptr, cute.make_ordered_layout(
                (self.q_seq_per_hk, self.d_v, self.h_k, self.b),
                (2, 0, 1, 3)
            )
        )
        print(f"q={q}")
        print(f"kcache={kcache}")
        print(f"out={out}")
    
        smem_layout_atom = make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
            q_dtype,
        )

        tma_atom_Q, tma_tensor_Q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            q,
            cute.tile_to_shape(
                smem_layout_atom,
                (self.block_size_m, self.head_dim_k),
                order=(0, 1),
            ),
            (self.block_size_m, self.head_dim_k),
        )
        tma_atom_K, tma_tensor_K = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            kcache,
            cute.tile_to_shape(
                smem_layout_atom,
                (self.page_block_size, 64),
                order=(0, 1),
            ),
            (self.page_block_size, 64),
        )
        tma_atom_O, tma_tensor_O = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            out,
            cute.tile_to_shape(
                smem_layout_atom,
                (self.block_size_m, self.head_dim_v),
                order=(0, 1),
            ),
            (self.block_size_m, self.head_dim_v),
        )

        tiled_mma_qk_sq = sm90_utils.make_trivial_tiled_mma(
            q_dtype,
            q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1,1,1),
            tiler_mn=(self.block_size_m, self.page_block_size),
        )
        tiled_mma_qk_rq = sm90_utils.make_trivial_tiled_mma(
            q_dtype,
            q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1,1,1),
            tiler_mn=(self.block_size_m, self.page_block_size),
            a_source=OperandSource.RMEM,
        )
        tiled_mma_pv_localP = sm90_utils.make_trivial_tiled_mma(
            q_dtype,
            q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1,1,1),
            tiler_mn=(self.block_size_m, self.head_dim_v//2),
            a_source=OperandSource.RMEM,
        )
        tiled_mma_pv_remoteP = sm90_utils.make_trivial_tiled_mma(
            q_dtype,
            q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1,1,1),
            tiler_mn=(self.block_size_m, self.head_dim_v//2),
        )

        q_smem_layout = cute.tile_to_shape(
            smem_layout_atom,
            (self.block_size_m, self.head_dim_k),
            order=(0,1)
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
        # v_smem_layout = cute.tile_to_shape(
        #     make_smem_layout_atom(cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128, q_dtype),
        #     (self.head_dim_v, self.page_block_size),
        #     order=(1, 0)
        # )
        p0_smem_layout = cute.tile_to_shape(
            smem_layout_atom,
            (self.block_size_m, self.page_block_size),
            order=(0,1)
        )
        self.rP0_layout = tiled_mma_qk_sq.partition_shape_C((self.block_size_m, self.page_block_size))
        print(f"rP0_layout={self.rP0_layout}")

        buffer_align_bytes = 16
        @cute.struct
        class SharedStorage:
            smem_q: cute.struct.Align[cute.struct.MemRange[q_dtype, cute.cosize(q_smem_layout)],
                buffer_align_bytes,
            ]
            smem_k0: cute.struct.Align[
                cute.struct.MemRange[q_dtype, cute.cosize(k_smem_layout)],
                buffer_align_bytes,
            ]
            smem_k1: cute.struct.Align[
                cute.struct.MemRange[q_dtype, cute.cosize(k_smem_layout)],
                buffer_align_bytes,
            ]
            smem_P0: cute.struct.Align[
                cute.struct.MemRange[q_dtype, cute.cosize(p0_smem_layout)],
                buffer_align_bytes,
            ]
            smem_M: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, self.block_size_m
                ],
                buffer_align_bytes,
            ]
            sL_reduction_wksp: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, 2*self.block_size_m
                ],
                buffer_align_bytes,
            ]
            smem_sScale0: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, self.block_size_m
                ],
                buffer_align_bytes,
            ]
            smem_sScale1: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, self.block_size_m
                ],
                buffer_align_bytes,
            ]
            barriers_k0: cute.struct.MemRange[
                cutlass.Int64, self.head_dim_k//64
            ]
            barriers_k1: cute.struct.MemRange[
                cutlass.Int64, self.head_dim_k//64
            ]
            barrier_q: cute.struct.MemRange[
                cutlass.Int64, 1
            ]
        self.shared_storage = SharedStorage

        num_m_block = cute.ceil_div(self.q_seq_per_hk, self.block_size_m)
        self.split_kv_mla_kernel(
            tma_atom_Q, tma_tensor_Q,
            tma_atom_K, tma_tensor_K,
            tma_atom_O, tma_tensor_O,
            softmax_lse,
            seqlens_k,
            block_table,
            tile_scheduler_metadata,
            num_splits,
            softmax_lse_accum_ptr,
            out_accum_ptr,
            q_smem_layout,
            k_smem_layout,
            v_smem_layout,
            p0_smem_layout,
            tiled_mma_qk_sq,
            tiled_mma_qk_rq,
            tiled_mma_pv_localP,
            tiled_mma_pv_remoteP,
        ).launch(
            grid=[num_m_block, self.h_k, self.num_sm_parts],
            block=[256, 1, 1],
            cluster=[1, 1, 1],
        )
        # self.softmax_lse = softmax_lse
        
        # # Sequence lengths
        # self.seqlens_k = seqlens_k
        
        # # Block table parameters
        # self.block_table = block_table
        # self.block_table_batch_stride = block_table.stride(0)

    @cute.jit
    def get_tiled_mma_pv_localP(
        self,
    ):
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1,1,1),
            tiler_mn=(self.block_size_m, self.head_dim_v//2),
            a_source=OperandSource.RMEM,
        )
        return tiled_mma

    @cute.kernel
    def split_kv_mla_kernel(
        self,
        tma_atom_Q: cute.CopyAtom,
        mQ: cute.Tensor,    # (q_seq_per_hk, d, h_k, b)
        tma_atom_K: cute.CopyAtom,
        mK: cute.Tensor,    # (page_size, d, h_k, num_blocks)
        tma_atom_O: cute.CopyAtom,
        mO: cute.Tensor,    # (q_seq_per_hk, d_v, h_k, b)
        softmax_lse: cute.Tensor,             # (b, h_k, q_seq_per_hk)
        seqlens_k: cute.Tensor,               # (b,)
        block_table: cute.Tensor,             # (b, max_num_pages)
        tile_scheduler_metadata: cute.Tensor, # (num_sm_parts, meta)
        num_splits: cute.Tensor,              # (b+1,)
        softmax_lse_accum_ptr: cute.Pointer,
        oaccum_ptr: cute.Pointer,
        q_smem_layout: cute.ComposedLayout,
        k_smem_layout: cute.ComposedLayout,
        v_smem_layout: cute.ComposedLayout,
        p0_smem_layout: cute.ComposedLayout,
        tiled_mma_qk_sq: cute.TiledMma,
        tiled_mma_qk_rq: cute.TiledMma,
        tiled_mma_pv_localP: cute.TiledMma,
        tiled_mma_pv_remoteP: cute.TiledMma,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_O)

        m_block_idx = bx
        k_head_idx = by
        partition_idx = bz
        warpgroup_idx = tid // 128
        idx_in_warpgroup = tid % 128
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sQ = storage.smem_q.get_tensor(q_smem_layout.outer, swizzle=q_smem_layout.inner)
        sK0 = storage.smem_k0.get_tensor(k_smem_layout.outer, swizzle=k_smem_layout.inner)
        sK1 = storage.smem_k1.get_tensor(k_smem_layout.outer, swizzle=k_smem_layout.inner)
        sP0 = storage.smem_P0.get_tensor(p0_smem_layout.outer, swizzle=p0_smem_layout.inner)
        sP1 = cute.flat_divide(sQ, (self.block_size_m, self.page_block_size))[None,None,0,8]
        sM = storage.smem_M.get_tensor((self.block_size_m,))
        sL_reduction_wksp = storage.sL_reduction_wksp.get_tensor((2*self.block_size_m,))
        sScale0 = storage.smem_sScale0.get_tensor((self.block_size_m,))
        sScale1 = storage.smem_sScale1.get_tensor((self.block_size_m,))

        sO_addr = sK0.iterator
        print(f"sQ={sQ}")
        gK = mK[None,None,k_head_idx,None]
        print(f"mK={mK}")
        print(f"gK={gK}")
        barriers_k0 = storage.barriers_k0.data_ptr()
        barriers_k1 = storage.barriers_k1.data_ptr()
        barrier_q = storage.barrier_q.data_ptr()
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(barrier_q, 1)
            for i in cutlass.range_constexpr(9):
                cute.arch.mbarrier_init(barriers_k0+i, 1)
                cute.arch.mbarrier_init(barriers_k1+i, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        # cute.arch.cluster_arrive_relaxed()
        cur_phase_Q = 0
        cur_phase_K0 = cute.make_rmem_tensor((1, ), Boolean)
        cur_phase_K1 = cute.make_rmem_tensor((1, ), Boolean)
        cur_phase_K0.fill(0); cur_phase_K1.fill(0)
        begin_idx = tile_scheduler_metadata[partition_idx, 0]
        begin_seqlen = tile_scheduler_metadata[partition_idx, 1]
        end_idx = tile_scheduler_metadata[partition_idx, 2]
        end_seqlen = tile_scheduler_metadata[partition_idx, 3]
        begin_n_split_idx = tile_scheduler_metadata[partition_idx, 4]

        tiled_mma_qk_sq.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        tiled_mma_qk_rq.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        tiled_mma_pv_localP.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        tiled_mma_pv_remoteP.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

        if begin_idx >= self.b:
            end_idx = begin_idx-1
        else:
            self.launch_q_copy(
                tma_atom_Q, mQ, sQ, begin_idx, m_block_idx, k_head_idx, barrier_q
            )

        for batch_idx in cutlass.range(begin_idx, end_idx+1, 1, unroll=1):
            kBlockN = self.page_block_size
            n_split_idx = begin_n_split_idx if batch_idx == begin_idx else 0
            seqlen_k = seqlens_k[batch_idx]
            start_block_idx = begin_seqlen // kBlockN if batch_idx == begin_idx else 0
            end_block_idx = cute.ceil_div(end_seqlen, kBlockN) if batch_idx == end_idx else cute.ceil_div(seqlen_k, kBlockN)
            is_no_split = Boolean(start_block_idx == 0 and end_block_idx == cute.ceil_div(seqlen_k, kBlockN))
            rRightBorderForQSeq = cute.make_rmem_tensor((2, ), Int32)
            if self.is_causal:
                pass
            else:
                rRightBorderForQSeq[0] = seqlen_k
                rRightBorderForQSeq[1] = seqlen_k
            
            gO = mO[None, None, k_head_idx, batch_idx]
            gO = cute.local_tile(gO, (self.block_size_m, self.head_dim_v), (m_block_idx, 0))
            gSoftmaxLse = softmax_lse[batch_idx, k_head_idx, None]
            gSoftmaxLse = cute.local_tile(gSoftmaxLse, (self.block_size_m,), (m_block_idx,))
            cur_block_table = block_table[batch_idx, None]

            if warpgroup_idx == 0:
                self.launch_kv_tiles_copy_tma(0, 9, tma_atom_K, gK[None,None, cur_block_table[start_block_idx]], sK0, barriers_k0)
            if start_block_idx+1 < end_block_idx:
                if warpgroup_idx == 1:
                    self.launch_kv_tiles_copy_tma(4, 9, tma_atom_K, gK[None,None, cur_block_table[start_block_idx+1]], sK1, barriers_k1)
                    self.launch_kv_tiles_copy_tma(0, 4, tma_atom_K, gK[None,None, cur_block_table[start_block_idx+1]], sK1, barriers_k1)
            
            rO = tiled_mma_pv_localP.make_fragment_C(
                tiled_mma_pv_localP.partition_shape_C((self.block_size_m, self.head_dim_v//2))
            )
            rO.fill(0.0)
            rL = cute.make_rmem_tensor((2,), Float32)
            rL.fill(0.0)

            if tid < cute.size(sM):
                sM[tid] = self.max_init_val_sm

            cute.arch.mbarrier_wait(barrier_q, cur_phase_Q)
            cur_phase_Q ^= 1

            rQ8 = cute.make_rmem_tensor(((2,2,2), 1, 4), self.q_dtype)
            self.retrieve_rP_from_sP(rQ8, cute.local_tile(sQ, (64, 64), (0, 8)), idx_in_warpgroup, tiled_mma_pv_localP)

            rP0_layout = tiled_mma_qk_sq.partition_shape_C((self.block_size_m, self.page_block_size))
            if warpgroup_idx == 0:
                rP0 = cute.make_rmem_tensor(rP0_layout, Float32)
                for i in cutlass.range_constexpr(9):
                    if idx_in_warpgroup==0:
                        cute.arch.mbarrier_arrive_and_expect_tx(barriers_k0+i, 64*64*2)
                    cute.arch.mbarrier_wait(barriers_k0+i, Int32(cur_phase_K0[0]))
                cur_phase_K0[0] ^= True
                # Issue P0 = Q @ K0^T, wait
                self.warpgroup_cooperative_qkt_gemm_no_pipeline(sQ, sK0, rP0, tiled_mma_qk_sq, idx_in_warpgroup)
                # wait for the previous write to sM is finished
                cute.arch.barrier(barrier_id=NamedBarriers.sMInitialized, number_of_threads=128)
                cute.nvgpu.warpgroup.wait_group(0)
                
                wg0_params = (
                    tma_atom_K,
                    tiled_mma_pv_localP, tiled_mma_pv_remoteP,
                    tiled_mma_qk_sq, tiled_mma_qk_rq,
                    gK, sQ, sK0, sK1, sP0, sP1, sM,
                    sScale0, sScale1,
                    rQ8, rP0, rO, rL, rRightBorderForQSeq,
                    barriers_k0, barriers_k1, cur_phase_K0,
                    cur_block_table,
                    v_smem_layout, seqlen_k, end_block_idx, idx_in_warpgroup)

                block_idx = start_block_idx
                while block_idx < end_block_idx-2:
                    self.wg0_subroutine(False, False, block_idx, *wg0_params)
                    block_idx += 2

                if block_idx+1 < end_block_idx:
                    self.wg0_subroutine(False, True, block_idx, *wg0_params)
                elif block_idx < end_block_idx:
                    self.wg0_subroutine(True, False, block_idx, *wg0_params)
            else:
                # warpgroup 1
                rP1 = cute.make_rmem_tensor(rP0_layout, Float32)
                if start_block_idx+1 < end_block_idx:
                    # Issue rP1 = sQ @ sK1, wait
                    self.warpgroup_cooperative_qkt_gemm(1, sQ, sK1, rP1, rQ8, tiled_mma_qk_sq, tiled_mma_qk_rq, barriers_k1, cur_phase_K1, idx_in_warpgroup)
                    cute.nvgpu.warpgroup.wait_group(0)

                wg1_params = (
                    tma_atom_K,
                    tiled_mma_pv_localP, tiled_mma_pv_remoteP,
                    tiled_mma_qk_sq, tiled_mma_qk_rq,
                    gK, sQ, sK0, sK1, sP0, sP1, sM,
                    sScale0, sScale1,
                    rQ8, rP1, rO, rL, rRightBorderForQSeq,
                    barriers_k0, barriers_k1, cur_phase_K1,
                    cur_block_table,
                    v_smem_layout, seqlen_k, end_block_idx, idx_in_warpgroup
                )

                block_idx = start_block_idx
                while block_idx < end_block_idx-3:
                    self.wg1_subroutine(False, False, False, block_idx, *wg1_params)
                    block_idx += 2
                
                if block_idx+2 < end_block_idx:
                    self.wg1_subroutine(False, False, True, block_idx, *wg1_params)
                    block_idx += 2
                    self.wg1_subroutine(True, False, False, block_idx, *wg1_params)
                elif block_idx+1 < end_block_idx:
                    self.wg1_subroutine(False, True, False, block_idx, *wg1_params)
                elif block_idx < end_block_idx:
                    self.wg1_subroutine(True, False, False, block_idx, *wg1_params)
            
            if begin_idx >= self.b:
                pass
            else:
                rL[0] += cute.arch.shuffle_sync_bfly(rL[0], offset=1)
                rL[0] += cute.arch.shuffle_sync_bfly(rL[0], offset=2)
                rL[1] += cute.arch.shuffle_sync_bfly(rL[1], offset=1)
                rL[1] += cute.arch.shuffle_sync_bfly(rL[1], offset=2)

                my_row = self.get_AorC_row_idx(0, idx_in_warpgroup)
                if idx_in_warpgroup % 4 == 0:
                    sL_reduction_wksp[my_row + warpgroup_idx*64] = rL[0]
                    sL_reduction_wksp[my_row+8 + warpgroup_idx*64] = rL[1]
                cute.arch.sync_threads()
                if warpgroup_idx == 0:
                    rL[0] += sL_reduction_wksp[my_row + 64]
                    rL[1] += sL_reduction_wksp[my_row + 8 + 64]
                else:
                    if idx_in_warpgroup % 4 == 0:
                        sL_reduction_wksp[my_row] += rL[0]
                        sL_reduction_wksp[my_row + 8] += rL[1]
                    cute.arch.sync_warp()
                    rL[0] = sL_reduction_wksp[my_row]
                    rL[1] = sL_reduction_wksp[my_row + 8]
                # Prune out when rL is 0 or NaN
                rL[0] = 1.0 if rL[0] == 0.0 or rL[0] != rL[0] else rL[0]
                rL[1] = 1.0 if rL[1] == 0.0 or rL[1] != rL[1] else rL[1]
                
                if batch_idx+1 <= end_idx:
                    self.launch_q_copy(
                        tma_atom_Q, mQ, sQ, batch_idx+1, m_block_idx, k_head_idx, barrier_q
                    )
                else:
                    # cudaTriggerProgrammaticLaunchCompletion
                    pass

                num_valid_seq_q = min(self.q_seq_per_hk - m_block_idx * self.block_size_m, self.block_size_m)
                if is_no_split:
                    self.store_o(True, rO, gO, tma_atom_O, rL, sO_addr, tiled_mma_pv_localP, batch_idx, k_head_idx, m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup)
                    i = tid
                    if i < num_valid_seq_q:
                        cur_L = sL_reduction_wksp[i]
                        gSoftmaxLse[i] = float('inf') if cur_L == 0.0 or cur_L != cur_L else cute.log(cur_L) + sM[i] / Float32(M_LOG2E)
                    cute.arch.cp_async_bulk_wait_group(0)
                else:
                    # not use for now
                    split_idx = num_splits[batch_idx] + n_split_idx
                    cur_o_ptr = oaccum_ptr + ((split_idx*self.h_k + k_head_idx)*self.q_seq_per_hk + m_block_idx*self.block_size_m)*self.head_dim_v
                    cur_lse_ptr = softmax_lse_accum_ptr + (split_idx*self.h_k + k_head_idx)*self.q_seq_per_hk + m_block_idx*self.block_size_m
                    gOAccum = cute.make_tensor(
                        oaccum_ptr,
                        cute.make_ordered_layout((self.block_size_m, self.head_dim_v), order=(1, 0))
                    )
                    gLseAccum = cute.make_tensor(
                        oaccum_ptr,
                        (self.block_size_m,)
                    )
                    self.store_o(False, rO, gOAccum, tma_atom_O, rL, sO_addr, tiled_mma_pv_localP, batch_idx, k_head_idx, m_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup)
                    
                    i = tid
                    if i < num_valid_seq_q:
                        cur_L = sL_reduction_wksp[i]
                        gLseAccum[i] = -float('inf') if cur_L == 0.0 or cur_L != cur_L else cute.log2(cur_L) + sM[i]
                    cute.arch.cp_async_bulk_wait_group(0)

                if batch_idx != end_idx:
                    cute.arch.sync_threads()

    @cute.jit
    def launch_q_copy(
        self,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        sQ: cute.Tensor,
        batch_idx: Int32,
        m_block_idx: Int32,
        k_head_idx: Int32,
        barrier_q: cute.Pointer,
    ):
        warp_idx = cute.arch.warp_idx()
        if warp_idx==0:
            # (sq*qh, 576)
            gQ = mQ[None, None, k_head_idx, batch_idx]
            gQ = cute.flat_divide(gQ, (self.block_size_m, self.head_dim_k))[None, None, m_block_idx, 0]
            tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                cta_coord=0,
                cta_layout=cute.make_layout((1,)),
                smem_tensor=cute.group_modes(sQ, 0, 2),
                gmem_tensor=cute.group_modes(gQ, 0, 2),
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(barrier_q, self.block_size_m * self.head_dim_k*2)
            cute.copy(
                tma_atom_q,
                tQgQ,
                tQsQ,
                tma_bar_ptr=barrier_q,
                # cache_policy=cute.CacheEvictionPriority.EVICT_FIRST
            )

    @cute.jit
    def launch_kv_tiles_copy_tma(
        self,
        start_tile_idx: cutlass.Constexpr,
        end_tile_idx: cutlass.Constexpr,
        tma_atom_k: cute.CopyAtom,
        gK: cute.Tensor,
        sK: cute.Tensor,
        barriers_K: cute.Pointer,
    ):
        for i in cutlass.range_constexpr(start_tile_idx, end_tile_idx):
            warp_idx = cute.arch.warp_idx()
            if warp_idx % 4 == 0:
                # directly use tma_partition is not work. terminate called after throwing an instance of 'std::bad_variant_access' what():  Unexpected index
                # so use flat_divide first
                def tile_for_tma(ts: cute.Tensor):
                    return cute.group_modes(
                        cute.flat_divide(ts, (self.page_block_size, 64))[None,None, 0, i],
                        0, 2
                    )
                sK_tma = tile_for_tma(sK)
                gK_tma = tile_for_tma(gK)
                tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    cta_coord=0,
                    cta_layout=cute.make_layout((1,)),
                    smem_tensor=sK_tma,
                    gmem_tensor=gK_tma,
                )
                cute.copy(
                    tma_atom_k,
                    tKgK,
                    tKsK,
                    tma_bar_ptr=barriers_K+i
                )

        
    @cute.jit
    def retrieve_rP_from_sP(
        self,
        rPb: cute.Tensor,
        sP: cute.Tensor,
        idx_in_warpgroup: Int32,
        tiled_mma_pv_localP: cute.TiledMma,
    ):  
        # SM75_U32x4_LDSM_N
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), self.q_dtype,
        )
        s2r_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma_pv_localP)
        thr_copy = s2r_copy.get_slice(idx_in_warpgroup)
        thr_copy_sP = thr_copy.partition_S(sP)
        thr_copy_rPb = thr_copy.retile(rPb)
        cute.copy(copy_atom, thr_copy_sP, thr_copy_rPb)
    
    @cute.jit
    def get_AorC_row_idx(
        self,
        local_row_idx: Int32,
        idx_in_warpgroup: Int32,
    ):
        row_idx = (idx_in_warpgroup//32)*16 + local_row_idx*8 + (idx_in_warpgroup%32)//4
        return row_idx

    @cute.jit
    def warpgroup_cooperative_qkt_gemm_no_pipeline(
        self,
        sQ: cute.Tensor,
        sKV: cute.Tensor,
        rP: cute.Tensor,
        tiled_mma_qk_sq: cute.TiledMma,
        idx_in_warpgroup: Int32,
    ):
        thr_mma = tiled_mma_qk_sq.get_slice(idx_in_warpgroup)
        tPsQ = thr_mma.partition_A(sQ)
        tPsK = thr_mma.partition_B(sKV)
        tPrQ = thr_mma.make_fragment_A(tPsQ)
        tPrK = thr_mma.make_fragment_B(tPsK)
        self.run_gemm(tiled_mma_qk_sq, tPrQ, tPrK, rP, True, -1)

    @cute.jit
    def warpgroup_cooperative_qkt_gemm(
        self,
        PHASE_IDX: cutlass.Constexpr,
        sQ: cute.Tensor,
        sKV: cute.Tensor,
        rP: cute.Tensor,
        rQ8: cute.Tensor,
        tiled_mma_qk_sq: cute.TiledMma,
        tiled_mma_qk_rq: cute.TiledMma,
        barriers: cute.Pointer,
        cur_phase: cute.Tensor,
        idx_in_warpgroup: Int32,
    ):
        # (BLOCK_SIZE_M, 64, 9)
        sQ_tiled = cute.flat_divide(sQ, (self.block_size_m, 64))[None,None, 0, None]
        sKV_tiled = cute.flat_divide(sKV, (self.page_block_size, 64))[None,None, 0, None]
        thr_mma_sQ = tiled_mma_qk_sq.get_slice(idx_in_warpgroup)
        tPsQ = thr_mma_sQ.partition_A(sQ_tiled)
        tPsK = thr_mma_sQ.partition_B(sKV_tiled)

        def qkt_gemm_one_tile(tile_idx: cutlass.Constexpr):
            if const_expr(tile_idx != 8):
                self.qkt_gemm_one_tile_sQ(
                    tiled_mma_qk_sq,
                    tPsQ[None,None,None,tile_idx],
                    tPsK[None,None,None,tile_idx],
                    rP,
                    barriers + tile_idx,
                    cur_phase, idx_in_warpgroup
                )
            else:
                self.qkt_gemm_one_tile_rQ(
                    tiled_mma_qk_rq,
                    rQ8,
                    tPsK[None,None,None,tile_idx],
                    rP,
                    barriers + tile_idx,
                    cur_phase, idx_in_warpgroup
                )
        
        if const_expr(PHASE_IDX == 0):
            rP.fill(0)
            for i in cutlass.range_constexpr(4):
                qkt_gemm_one_tile(i)
        elif const_expr(PHASE_IDX == 1):
            # tiled_mma_qk_sq.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            # tiled_mma_qk_rq.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            rP.fill(0)
            for i in cutlass.range_constexpr(4, 4+9):
                qkt_gemm_one_tile(i%9)
            cur_phase[0] ^= True
        else:
            # tiled_mma_qk_sq.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            # tiled_mma_qk_rq.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            for i in cutlass.range_constexpr(4, 9):
                qkt_gemm_one_tile(i)
            cur_phase[0] ^= True
        

    @cute.jit
    def qkt_gemm_one_tile_sQ(
        self,
        tiled_mma: cute.TiledMma,
        tPsQ: cute.Tensor,
        tPsK: cute.Tensor,
        rP: cute.Tensor,
        barrier: cute.Pointer,
        cur_phase: cute.Tensor,
        idx_in_warpgroup: Int32,
    ):
        tPrQ = tiled_mma.make_fragment_A(tPsQ)
        tPrK = tiled_mma.make_fragment_B(tPsK)
        if idx_in_warpgroup == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 64*64*2)
        cute.arch.mbarrier_wait(barrier, cur_phase[0])
        cute.nvgpu.warpgroup.fence()
        # wgmma k is 16, sQ k is 64, so have 4 ktiles
        cute.gemm(tiled_mma, rP, tPrQ[None,None,0], tPrK[None,None,0], rP)
        cute.gemm(tiled_mma, rP, tPrQ[None,None,1], tPrK[None,None,1], rP)
        cute.gemm(tiled_mma, rP, tPrQ[None,None,2], tPrK[None,None,2], rP)
        cute.gemm(tiled_mma, rP, tPrQ[None,None,3], tPrK[None,None,3], rP)
        cute.nvgpu.warpgroup.commit_group()

    @cute.jit
    def qkt_gemm_one_tile_rQ(
        self,
        tiled_mma: cute.TiledMma,
        tPrQ: cute.Tensor,
        tPsK: cute.Tensor,
        rP: cute.Tensor,
        barrier: cute.Pointer,
        cur_phase: cute.Tensor,
        idx_in_warpgroup: Int32,
    ):
        tPrK = tiled_mma.make_fragment_B(tPsK)
        if idx_in_warpgroup == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 64*64*2)
        cute.arch.mbarrier_wait(barrier, cur_phase[0])
        cute.nvgpu.warpgroup.fence()
        # wgmma k is 16, sQ k is 64, so have 4 ktiles
        cute.gemm(tiled_mma, rP, tPrQ[None,None,0], tPrK[None,None,0], rP)
        cute.gemm(tiled_mma, rP, tPrQ[None,None,1], tPrK[None,None,1], rP)
        cute.gemm(tiled_mma, rP, tPrQ[None,None,2], tPrK[None,None,2], rP)
        cute.gemm(tiled_mma, rP, tPrQ[None,None,3], tPrK[None,None,3], rP)
        cute.nvgpu.warpgroup.commit_group()

    @cute.jit
    def wg0_subroutine(
        self,
        is_blk0_last: cutlass.Constexpr,
        is_blk1_last: cutlass.Constexpr,
        block_idx: Int32,
        tma_atom_k: cute.CopyAtom,
        tiled_mma_pv_localP: cute.TiledMma,
        tiled_mma_pv_remoteP: cute.TiledMma,
        tiled_mma_qk_sq: cute.TiledMma,
        tiled_mma_qk_rq: cute.TiledMma,
        gK: cute.Tensor,
        sQ: cute.Tensor,
        sK0: cute.Tensor,
        sK1: cute.Tensor,
        sP0: cute.Tensor,
        sP1: cute.Tensor,
        sM: cute.Tensor,
        sScale0: cute.Tensor,
        sScale1: cute.Tensor,
        rQ8: cute.Tensor,
        rP0: cute.Tensor,
        rO0: cute.Tensor,
        rL: cute.Tensor,
        rRightBorderForQSeq: cute.Tensor,
        barriers_K0: cute.Pointer,
        barriers_K1: cute.Pointer,
        cur_phase_K0: cute.Tensor,
        cur_block_table: cute.Tensor,
        v_smem_layout: cute.ComposedLayout,
        seqlen_k: Int32,
        end_block_idx: Int32,
        idx_in_warpgroup: Int32,
    ):
        start_token_idx = block_idx * self.page_block_size
        get_block_idx = lambda e: 0 if e >= end_block_idx else cur_block_table[e]
        nxt_block0_index = get_block_idx(block_idx+2)
        nxt_block1_index = get_block_idx(block_idx+3)

        sV0L = self.get_half_V(sK0, v_smem_layout, 0)
        sV1L = self.get_half_V(sK1, v_smem_layout, 0)
        rPb = cute.make_rmem_tensor(
            ((2,2,2), 1, 4),
            self.q_dtype
        )
        # Calc P0 = softmax(P0)
        self.wg0_bunch_0(
            is_blk0_last or is_blk1_last,
            rPb, rP0, rO0, sScale0, sM, rL, rRightBorderForQSeq, self.scale_softmax_log2, start_token_idx, idx_in_warpgroup
        )
        cute.arch.barrier_arrive(barrier_id=NamedBarriers.sScale0Ready, number_of_threads=256)
        # Issue rO0 += rPb @ sV0L
        if const_expr(is_blk0_last):
            self.fill_oob_V(sV0L, seqlen_k-start_token_idx, idx_in_warpgroup)
            cute.arch.fence_view_async_shared()
        self.warpgroup_cooperative_pv_gemm_localP(
            rPb, sV0L, rO0, tiled_mma_pv_localP, idx_in_warpgroup
        )
        # Wait for rO0, launch TMA for the next V0L
        cute.nvgpu.warpgroup.wait_group(0)
        # Wait for warpgroup 1, rescale P0, notify warpgroup 1
        cute.arch.barrier(barrier_id=NamedBarriers.sScale1Ready, number_of_threads=256)
        if const_expr(not is_blk0_last and not is_blk1_last):
            self.launch_kv_tiles_copy_tma(0, 4, tma_atom_k, gK[None,None,nxt_block0_index], sK0, barriers_K0)
        self.wg0_scale_rP0(
            sScale1, rP0, rPb, idx_in_warpgroup
        )
        self.save_rPb_to_sP(rPb, sP0, tiled_mma_qk_sq, idx_in_warpgroup)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(barrier_id=NamedBarriers.sP0Ready, number_of_threads=256)
        # Wait for warpgroup 1, rescale O0, issue rO0 += rPb @ sV1L
        if const_expr(not is_blk0_last):
            if const_expr(is_blk1_last):
                self.fill_oob_V(sV1L, seqlen_k - start_token_idx - self.page_block_size, idx_in_warpgroup)
                cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=NamedBarriers.rO1sP0sV0RIssued, number_of_threads=256)
            self.wg0_rescale_rO0(rO0, sScale1, rL, idx_in_warpgroup)
            self.warpgroup_cooperative_pv_gemm_remoteP(sP1, sV1L, rO0, tiled_mma_pv_remoteP, idx_in_warpgroup)

        # Issue P0 = Q @ K0^T
        # Since TMAs for these 4 tiles are launched right after rO0 += rPb @ sV0L finishes, they should have already finished. Therefore, we issue the first 4 tiles to fill the pipeline.
        if const_expr(not is_blk0_last and not is_blk1_last):
            self.warpgroup_cooperative_qkt_gemm(0, sQ, sK0, rP0, rQ8, tiled_mma_qk_sq, tiled_mma_qk_rq, barriers_K0, cur_phase_K0, idx_in_warpgroup)
        # Wait for rO0 += rPb @ sV1L, launch TMA
        if not is_blk0_last and not is_blk1_last and block_idx+3 < end_block_idx:
            cute.nvgpu.warpgroup.wait_group(4)
            self.launch_kv_tiles_copy_tma(0, 4, tma_atom_k, gK[None,None, nxt_block1_index], sK1, barriers_K1)

        # Issue P0 = Q @ K0^T
        if const_expr(not is_blk0_last and not is_blk1_last):
            self.warpgroup_cooperative_qkt_gemm(2, sQ, sK0, rP0, rQ8, tiled_mma_qk_sq, tiled_mma_qk_rq, barriers_K0, cur_phase_K0, idx_in_warpgroup)
        # Wait for P0 = Q @ K0^T
        cute.nvgpu.warpgroup.wait_group(0)

    @cute.jit
    def save_rPb_to_sP(
        self,
        rPb: cute.Tensor,
        sP: cute.Tensor,
        tiled_mma_qk_sq: cute.TiledMma,
        idx_in_warpgroup: Int32
    ):
        # SM90_U32x4_STSM_N
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), self.q_dtype,
        )
        r2s_copy = cute.make_tiled_copy_C(copy_atom, tiled_mma_qk_sq)
        thr_copy = r2s_copy.get_slice(idx_in_warpgroup)
        thr_copy_rPb = thr_copy.retile(rPb)
        thr_copy_sP = thr_copy.partition_D(sP)
        cute.copy(copy_atom, thr_copy_rPb, thr_copy_sP)

    @cute.jit
    def get_half_V(
        self,
        sK: cute.Tensor,
        v_smem_layout: cute.ComposedLayout,
        is_R: Int32,
    ):
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, v_smem_layout.inner, self.q_dtype),
            v_smem_layout.outer
        )
        ret = cute.flat_divide(sV, (self.head_dim_v//2, self.page_block_size))[None, None, is_R, 0]
        return ret

    @cute.jit
    def wg0_bunch_0(
        self,
        do_oob_filling: cutlass.Constexpr,
        rPb: cute.Tensor,
        rP0: cute.Tensor,
        rO0: cute.Tensor,
        sScale0: cute.Tensor,
        sM: cute.Tensor,
        rL: cute.Tensor,
        rRightBorderForQSeq: cute.Tensor,
        scale_softmax_log2: Float32,
        start_token_idx: Int32,
        idx_in_warpgroup: Int32,
    ):
        for local_row_idx in cutlass.range(2):
            row_idx = self.get_AorC_row_idx(local_row_idx, idx_in_warpgroup)
            cur_max = -float("inf")
            start_i = 2 if local_row_idx > 0 else 0
            for i in cutlass.range(start_i, cute.size(rP0), 4, unroll_full=True):
                if const_expr(do_oob_filling):
                    token_idx = start_token_idx + (i//4)*8 + idx_in_warpgroup%4*2
                    rP0[i] = rP0[i] if token_idx < rRightBorderForQSeq[local_row_idx] else -float("inf")
                    rP0[i+1] = rP0[i+1] if token_idx+1 < rRightBorderForQSeq[local_row_idx] else -float("inf")
                cur_max = max(cur_max, max(rP0[i], rP0[i+1]))
            
            cur_max = max(cur_max, cute.arch.shuffle_sync_bfly(cur_max, offset=1))
            cur_max = max(cur_max, cute.arch.shuffle_sync_bfly(cur_max, offset=2))

            cur_max *= scale_softmax_log2
            new_max = max(sM[row_idx], cur_max)
            scale_for_old = cute.exp2(sM[row_idx] - new_max)
            cute.arch.sync_warp()
            if idx_in_warpgroup % 4 == 0:
                sScale0[row_idx] = scale_for_old
                sM[row_idx] = new_max
            
            for i in cutlass.range(start_i, cute.size(rO0), 4, unroll_full=True):
                rO0[i] *= scale_for_old
                rO0[i+1] *= scale_for_old
            
            cur_sum = Float32(0)
            for i in cutlass.range(start_i, cute.size(rP0), 4, unroll_full=True):
                rP0[i] = cute.exp2(rP0[i] * scale_softmax_log2 - new_max)
                rP0[i+1] = cute.exp2(rP0[i+1] * scale_softmax_log2 - new_max)
                rPb[i] = self.q_dtype(rP0[i])
                rPb[i+1] = self.q_dtype(rP0[i+1])
                cur_sum += rP0[i] + rP0[i+1]
            rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum

    @cute.jit
    def wg1_bunch_0(
        self,
        is_blk0_last: cutlass.Constexpr,
        is_blk1_last: cutlass.Constexpr,
        is_blk2_last: cutlass.Constexpr,
        rP1b: cute.Tensor,
        sScale1: cute.Tensor,
        rO1: cute.Tensor,
        sM: cute.Tensor,
        rL: cute.Tensor,
        rRightBorderForQSeq: cute.Tensor,
        sScale0: cute.Tensor,
        rP1: cute.Tensor,
        scale_softmax_log2: Float32,
        start_token_idx: Int32,
        idx_in_warpgroup: Int32,
    ):
        for local_row_idx in cutlass.range(2):
            row_idx = self.get_AorC_row_idx(local_row_idx, idx_in_warpgroup)

            cur_max = -float("inf")
            start_i = 2 if local_row_idx > 0 else 0
            for i in cutlass.range(start_i, cute.size(rP1), 4, unroll_full=True):
                if const_expr(is_blk1_last or is_blk2_last):
                    token_idx = start_token_idx + (i//4)*8 + idx_in_warpgroup%4*2
                    rP1[i] = rP1[i] if token_idx < rRightBorderForQSeq[local_row_idx] else -float("inf")
                    rP1[i+1] = rP1[i+1] if token_idx+1 < rRightBorderForQSeq[local_row_idx] else -float("inf")
                elif const_expr(is_blk0_last):
                    rP1[i] = -float("inf")
                    rP1[i+1] = -float("inf")
                cur_max = max(cur_max, max(rP1[i], rP1[i+1]))

            cur_max = max(cur_max, cute.arch.shuffle_sync_bfly(cur_max, offset=1))
            cur_max = max(cur_max, cute.arch.shuffle_sync_bfly(cur_max, offset=2))
            cur_max *= scale_softmax_log2

            old_max = sM[row_idx]
            new_max = max(old_max, cur_max)
            scale_for_old = cute.exp2(old_max - new_max)
            cute.arch.sync_warp()
            if idx_in_warpgroup % 4 == 0:
                sM[row_idx] = new_max
                sScale1[row_idx] = scale_for_old

            cur_sum = Float32(0)
            if const_expr(not is_blk0_last):
                for i in cutlass.range(start_i, cute.size(rP1), 4, unroll_full=True):
                    rP1[i] = cute.exp2(rP1[i] * scale_softmax_log2 - new_max)
                    rP1[i+1] = cute.exp2(rP1[i+1] * scale_softmax_log2 - new_max)
                    rP1b[i] = self.q_dtype(rP1[i])
                    rP1b[i+1] = self.q_dtype(rP1[i+1])
                    cur_sum += rP1[i] + rP1[i+1]

            cur_scale_for_o1 = scale_for_old * sScale0[row_idx]
            for i in cutlass.range(start_i, cute.size(rO1), 4, unroll_full=True):
                rO1[i] *= cur_scale_for_o1
                rO1[i+1] *= cur_scale_for_o1

            rL[local_row_idx] = rL[local_row_idx] * cur_scale_for_o1 + cur_sum

    @cute.jit
    def warpgroup_cooperative_pv_gemm_remoteP(
        self,
        sP: cute.Tensor,
        sKV_half: cute.Tensor,
        rO: cute.Tensor,
        tiled_mma_pv_remoteP: cute.TiledMma,
        idx_in_warpgroup: Int32,
    ):
        # tiled_mma = self.get_tiled_mma_pv_localP()
        thr_mma = tiled_mma_pv_remoteP.get_slice(idx_in_warpgroup)
        mma_sP = thr_mma.partition_A(sP)
        mma_sKV = thr_mma.partition_B(sKV_half)
        mma_rP = thr_mma.make_fragment_A(mma_sP)
        mma_rKV = thr_mma.make_fragment_B(mma_sKV)
        self.run_gemm(tiled_mma_pv_remoteP, mma_rP, mma_rKV, rO, False, -1)
    
    @cute.jit
    def warpgroup_cooperative_pv_gemm_localP(
        self,
        rP: cute.Tensor, # ((2, 2, 8), 1, 1)
        sKV_half: cute.Tensor, # (HEAD_DIM_V/2, PAGE_BLOCK_SIZE)
        rO: cute.Tensor,
        tiled_mma_pv_localP: cute.TiledMma,
        idx_in_warpgroup: Int32,
    ):
        thr_mma = tiled_mma_pv_localP.get_slice(idx_in_warpgroup)
        rP_retiled = cute.make_tensor(
            rP.iterator,
            cute.make_layout(
                ((2,2,2), 1, 4),
                stride=((1,2,4), 0, 8)
            )
        )
        mma_sKV = thr_mma.partition_B(sKV_half)
        mma_rKV = thr_mma.make_fragment_B(mma_sKV)
        self.run_gemm(tiled_mma_pv_localP, rP_retiled, mma_rKV, rO, False, -1)

    @cute.jit
    def run_gemm(
        self,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        tCrC: cute.Tensor,
        zero_init: cutlass.Constexpr,
        wg_wait: cutlass.Constexpr,
        need_commit: cutlass.Constexpr = True,
    ):
        cute.nvgpu.warpgroup.fence()
        if const_expr(zero_init):
            tCrC.fill(0.0)
            for k_block in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None,None,k_block],
                    tCrB[None,None,k_block],
                    tCrC,
                )
        else:
            for k_block in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None,None,k_block],
                    tCrB[None,None,k_block],
                    tCrC,
                )
        if const_expr(need_commit):
            cute.nvgpu.warpgroup.commit_group()
        if const_expr(wg_wait >= 0):
            cute.nvgpu.warpgroup.wait_group(wg_wait)
    
    @cute.jit
    def wg0_scale_rP0(
        self,
        sScale1: cute.Tensor,
        rP0: cute.Tensor,
        rPb: cute.Tensor,
        idx_in_warpgroup: Int32,
    ):
        for local_row_idx in cutlass.range(2, unroll_full=True):
            row_idx = self.get_AorC_row_idx(local_row_idx, idx_in_warpgroup)
            scale_factor = sScale1[row_idx]
            start_i = 2 if local_row_idx > 0 else 0
            for i in cutlass.range(start_i, cute.size(rP0), 4, unroll_full=True):
                rPb[i] = self.q_dtype(rP0[i] * scale_factor)
                rPb[i+1] = self.q_dtype(rP0[i+1] * scale_factor)

    @cute.jit
    def wg0_rescale_rO0(
        self,
        rO0: cute.Tensor,
        sScale1: cute.Tensor,
        rL: cute.Tensor,
        idx_in_warpgroup: Int32,
    ):
        for local_row_idx in cutlass.range(2, unroll_full=True):
            row_idx = self.get_AorC_row_idx(local_row_idx, idx_in_warpgroup)
            scale_factor = sScale1[row_idx]
            start_i = 2 if local_row_idx > 0 else 0
            for i in cutlass.range(start_i, cute.size(rO0), 4, unroll_full=True):
                rO0[i] *= scale_factor
                rO0[i+1] *= scale_factor
            rL[local_row_idx] *= scale_factor

    @cute.jit
    def fill_oob_V(
        self,
        sV: cute.Tensor,
        valid_window_size: Int32,
        idx_in_warpgroup: Int32,
    ):
        sV_int64 = cute.make_tensor(
            cute.recast_ptr(sV.iterator, dtype=Int64),
            cute.tile_to_shape(
                cute.make_ordered_layout((16, 8), order=(0, 1)),
                (64, self.page_block_size), order=(1, 0)
            )
        )
        valid_window_size = max(valid_window_size, 0)
        head_dim_size = cute.size(sV_int64, [0])
        for token_idx in cutlass.range(
            valid_window_size + (idx_in_warpgroup//head_dim_size),
            cute.size(sV, [1]),
            128 // head_dim_size
        ):
            sV_int64[idx_in_warpgroup%head_dim_size, token_idx] = 0

    @cute.jit
    def store_o(
        self,
        is_no_split: cutlass.Constexpr,
        rO: cute.Tensor,
        gOorAccum: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        rL: cute.Tensor,
        sO_addr: cute.Pointer,
        tiled_mma_pv_localP: cute.TiledMma,
        batch_idx: Int32,
        k_head_idx: Int32,
        m_block_idx: Int32,
        num_valid_seq_q: Int32,
        warpgroup_idx: Int32,
        idx_in_warpgroup: Int32,
    ):
        if const_expr(is_no_split):
            sO_layout = cute.tile_to_shape(
                make_smem_layout_atom(cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128, self.q_dtype),
                (self.block_size_m, self.head_dim_v),
                order=(0,1)
            )
            sO = cute.make_tensor(
                cute.recast_ptr(sO_addr, sO_layout.inner, self.q_dtype),
                sO_layout.outer
            )
            rOb = cute.make_rmem_tensor_like(rO, self.q_dtype)
            for idx in cutlass.range(cute.size(rO), unroll_full=True):
                rOb[idx] = self.q_dtype(rO[idx] / rL[Int32(idx % 4 >= 2)])
            
            sO_cur = cute.local_tile(sO, (64, 256), (0, warpgroup_idx))
            # gO_cur = cute.local_tile(gOorAccum, (64, 256), (0, warpgroup_idx))
            r2s_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), self.q_dtype,
            )
            print(f"sO_cur={sO_cur}")
            print(f"rOb={rOb}")
            r2s_copy = cute.make_tiled_copy_C(r2s_atom, tiled_mma_pv_localP)
            thr_copy = r2s_copy.get_slice(idx_in_warpgroup)
            thr_copy_rOb = thr_copy.retile(rOb)
            thr_copy_sO = thr_copy.partition_D(sO_cur)
            cute.copy(r2s_atom, thr_copy_rOb, thr_copy_sO)
            cute.arch.fence_view_async_shared()

            cute.arch.sync_threads()
            tid,_,_ = cute.arch.thread_idx()
            if tid // 32 == 0:
                # TMA s2g
                tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_O,
                    cta_coord=0,
                    cta_layout=cute.make_layout((1,)),
                    smem_tensor=cute.group_modes(sO, 0, 2),
                    gmem_tensor=cute.group_modes(gOorAccum, 0, 2),
                )
                print(f"sO={sO}, tOsO={tOsO}")
                print(f"gO_cur={gOorAccum}, tOgO={tOgO}")
                cute.copy(
                    tma_atom_O,
                    tOsO,
                    tOgO,
                )
                cute.arch.cp_async_bulk_commit_group()
        else:
            # Should save the result to OAccum
            sO = cute.make_tensor(
                cute.recast_ptr(sO_addr, dtype=Float32),
                cute.make_layout((64, 512), stride=(520, 1)) # pad to 520 avoid bank conflict
            )
            for idx in cutlass.range(0, cute.size(rO), 2, unroll_full=True):
                tmp = 8 if idx%4 >= 2 else 0
                row = (idx_in_warpgroup // 32) * 16 + (idx_in_warpgroup%32//4) + tmp
                col = warpgroup_idx * 256 + (idx_in_warpgroup%4) * 2 + idx//4*8
                sO[row, col] = rO[idx] / rL[Int32(idx%4 >= 2)]
                sO[row, col+1] = rO[idx+1] / rL[Int32(idx%4 >= 2)]
            cute.arch.fence_view_async_shared()

            cute.arch.sync_threads()
            row, _, _ = cute.arch.thread_idx()
            # if row < num_valid_seq_q:
            #     sm90_bulk_copy_s2g(sO[row].iterator, gOorAccum[row].iterator, self.head_dim_v * 4)
            #     cute.arch.cp_async_bulk_commit_group()
            
            # cute.arch.cp_async_bulk_wait_group


    @cute.jit
    def wg1_subroutine(
        self,
        is_blk0_last: cutlass.Constexpr,
        is_blk1_last: cutlass.Constexpr,
        is_blk2_last: cutlass.Constexpr,
        block_idx: Int32,
        tma_atom_k: cute.CopyAtom,
        tiled_mma_pv_localP: cute.TiledMma,
        tiled_mma_pv_remoteP: cute.TiledMma,
        tiled_mma_qk_sq: cute.TiledMma,
        tiled_mma_qk_rq: cute.TiledMma,
        gK: cute.Tensor,
        sQ: cute.Tensor,
        sK0: cute.Tensor,
        sK1: cute.Tensor,
        sP0: cute.Tensor,
        sP1: cute.Tensor,
        sM: cute.Tensor,
        sScale0: cute.Tensor,
        sScale1: cute.Tensor,
        rQ8: cute.Tensor,
        rP1: cute.Tensor,
        rO1: cute.Tensor,
        rL: cute.Tensor,
        rRightBorderForQSeq: cute.Tensor,
        barriers_K0: cute.Pointer,
        barriers_K1: cute.Pointer,
        cur_phase_K1: cute.Tensor,
        cur_block_table: cute.Tensor,
        v_smem_layout: cute.ComposedLayout,
        seqlen_k: Int32,
        end_block_idx: Int32,
        idx_in_warpgroup: Int32,
    ):
        start_token_idx = block_idx * self.page_block_size
        get_block_idx = lambda e: 0 if e >= end_block_idx else cur_block_table[e]
        nxt_block0_index = get_block_idx(block_idx+2)
        nxt_block1_index = get_block_idx(block_idx+3)

        rP1b = cute.make_rmem_tensor(
            ((2,2,2), 1, 4),
            self.q_dtype
        )
        sV0R = self.get_half_V(sK0, v_smem_layout, 1)
        sV1R = self.get_half_V(sK1, v_smem_layout, 1)

        # Wait for rP1 and warpgroup 0, run bunch 1, notify warpgroup 0
        cute.arch.barrier(barrier_id=NamedBarriers.sScale0Ready, number_of_threads=256)
        self.wg1_bunch_0(
            is_blk0_last,
            is_blk1_last,
            is_blk2_last,
            rP1b, sScale1, rO1, sM, rL, rRightBorderForQSeq, sScale0, rP1,
            self.scale_softmax_log2,
            start_token_idx + self.page_block_size,
            idx_in_warpgroup,
        )
        cute.arch.barrier_arrive(barrier_id=NamedBarriers.sScale1Ready, number_of_threads=256)

        # Save rP1b to sP1, and issue rO1 += rP1b @ sV1R
        if const_expr(not is_blk0_last):
            self.save_rPb_to_sP(rP1b, sP1, tiled_mma_qk_sq, idx_in_warpgroup)
        if const_expr(not is_blk0_last):
            if const_expr(is_blk1_last):
                self.fill_oob_V(sV1R, seqlen_k-start_token_idx-self.page_block_size, idx_in_warpgroup)
                cute.arch.fence_view_async_shared()
            self.warpgroup_cooperative_pv_gemm_localP(
                rP1b, sV1R, rO1, tiled_mma_pv_localP, idx_in_warpgroup
            )
            if const_expr(not is_blk1_last):
                # Make sP1 visible to the async proxy if no previous fence was issued.
                cute.arch.fence_view_async_shared()

        # Wait for sP0, issue rO1 += sP0 @ sV0R, notify warpgroup 0
        cute.arch.barrier(barrier_id=NamedBarriers.sP0Ready, number_of_threads=256)
        if const_expr(is_blk0_last):
            self.fill_oob_V(sV0R, seqlen_k-start_token_idx, idx_in_warpgroup)
            cute.arch.fence_view_async_shared()
        self.warpgroup_cooperative_pv_gemm_remoteP(
            sP0, sV0R, rO1, tiled_mma_pv_remoteP, idx_in_warpgroup
        )
        if const_expr(not is_blk0_last):
            cute.arch.barrier_arrive(barrier_id=NamedBarriers.rO1sP0sV0RIssued, number_of_threads=256)

        # Wait for rO1 += rP1b @ sV1R, launch TMA for the next V1R
        if const_expr(not is_blk0_last and not is_blk1_last and not is_blk2_last):
            cute.nvgpu.warpgroup.wait_group(1)
            self.launch_kv_tiles_copy_tma(4, 9, tma_atom_k, gK[None,None,nxt_block1_index], sK1, barriers_K1)

        # Wait for rO1 += sP0 @ sV0R, launch TMA for the next V0R
        if const_expr(not is_blk0_last and not is_blk1_last):
            cute.nvgpu.warpgroup.wait_group(0)
            self.launch_kv_tiles_copy_tma(4, 9, tma_atom_k, gK[None,None,nxt_block0_index], sK0, barriers_K0)

        # Issue rP1 = sQ @ sK1, wait
        if const_expr(not is_blk0_last and not is_blk1_last and not is_blk2_last):
            cur_phase_K1 = self.warpgroup_cooperative_qkt_gemm(
                1, sQ, sK1, rP1, rQ8, tiled_mma_qk_sq, tiled_mma_qk_rq, barriers_K1, cur_phase_K1, idx_in_warpgroup
            )

        # Keep this wait outside conditionals to preserve WGMMA pipeline behavior.
        cute.nvgpu.warpgroup.wait_group(0)

    @cute.kernel
    def mla_combine(self):
        pass

def mla_kvcache_fwd(
    q: torch.Tensor,                              # bs x s_q x h_q x hd
    kcache: torch.Tensor,                         # num_blocks x page_size x h_k x hd
    head_size_v: int,
    seqlens_k: torch.Tensor,                      # bs
    block_table: torch.Tensor,                    # bs x max_num_pages_per_seq
    softmax_scale: float,
    is_causal: bool,
    tile_scheduler_metadata: torch.Tensor,        # num_sm_parts x TileSchedulerMetaDataSize
    num_splits: torch.Tensor                      # batch_size + 1
) -> List[torch.Tensor]:
    """
    Multi-Head Attention forward pass with KV cache and MLA (Multi-Latent Attention)
    
    Args:
        q: Query tensor [batch_size, seqlen_q, num_heads, head_size]
        kcache: Key cache tensor [num_blocks, page_block_size, num_heads_k, head_size]
        head_size_v: Value head dimension
        seqlens_k: Sequence lengths for keys [batch_size]
        block_table: Block table for paged attention [batch_size, max_num_blocks_per_seq]
        softmax_scale: Scaling factor for softmax
        is_causal: Whether to apply causal masking
        tile_scheduler_metadata: Metadata for tile scheduling [num_sm_parts, TileSchedulerMetaDataSize]
        num_splits: Number of splits per batch [batch_size + 1]
    
    Returns:
        List containing [output, softmax_lse]
    """
    
    # Check the architecture
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    device_props = torch.cuda.get_device_properties(q.device)
    is_sm90 = device_props.major == 9 and device_props.minor == 0
    assert is_sm90, "Only SM90 architecture is supported"
    
    # Check data types
    q_dtype = q.dtype
    assert q_dtype in [torch.bfloat16, torch.float16], \
        "Query must be bfloat16 or float16"
    assert kcache.dtype == q_dtype, \
        "Query and key must have the same dtype"
    assert seqlens_k.dtype == torch.int32, \
        "seqlens_k must have dtype int32"
    assert block_table.dtype == torch.int32, \
        "block_table must have dtype torch.int32"
    assert tile_scheduler_metadata.dtype == torch.int32, \
        "tile_scheduler_metadata must have dtype int32"
    assert num_splits.dtype == torch.int32, \
        "num_splits must have dtype int32"
    
    # Check device
    assert q.is_cuda, "q must be on CUDA device"
    assert kcache.is_cuda, "kcache must be on CUDA device"
    assert seqlens_k.is_cuda, "seqlens_k must be on CUDA device"
    assert block_table.is_cuda, "block_table must be on CUDA device"
    assert tile_scheduler_metadata.is_cuda, "tile_scheduler_metadata must be on CUDA device"
    assert num_splits.is_cuda, "num_splits must be on CUDA device"
    
    # Check layout
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert seqlens_k.is_contiguous(), "seqlens_k must be contiguous"
    assert block_table.stride(-1) == 1, "block_table must have contiguous last dimension"
    assert tile_scheduler_metadata.is_contiguous(), "tile_scheduler_metadata must be contiguous"
    assert num_splits.is_contiguous(), "num_splits must be contiguous"
    
    # Get dimensions
    batch_size, seqlen_q_ori, num_heads_q, head_size_k = q.shape
    assert head_size_k == 576, "Only head_size_k == 576 is supported"
    assert head_size_v == 512, "Only head_size_v == 512 is supported"
    
    max_num_blocks_per_seq = block_table.size(1)
    num_blocks = kcache.size(0)
    page_block_size = kcache.size(1)
    num_heads_k = kcache.size(2)
    
    assert batch_size > 0, "batch size must be positive"
    assert num_heads_q % num_heads_k == 0, \
        "Number of heads in key/value must divide number of heads in query"
    
    if seqlen_q_ori == 1:
        is_causal = False
    
    # Reshape query for grouped query attention
    num_q_heads_per_hk = num_heads_q // num_heads_k
    q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk
    num_heads = num_heads_k
    
    q = q.view(batch_size, seqlen_q_ori, num_heads_k, num_q_heads_per_hk, head_size_k) \
         .transpose(2, 3) \
         .reshape(batch_size, q_seq_per_hk, num_heads, head_size_k)
    
    # Verify shapes
    assert q.shape == (batch_size, q_seq_per_hk, num_heads, head_size_k)
    assert kcache.shape == (num_blocks, page_block_size, num_heads_k, head_size_k)
    assert seqlens_k.shape == (batch_size,)
    assert block_table.shape == (batch_size, max_num_blocks_per_seq)
    assert num_splits.shape == (batch_size + 1,), f"{num_splits.shape=}, {batch_size=}"

    # Create output tensors
    out = torch.empty(
        (batch_size, q_seq_per_hk, num_heads, head_size_v),
        dtype=q.dtype,
        device=q.device
    )
    softmax_lse = torch.empty(
        (batch_size, num_heads, q_seq_per_hk),
        dtype=torch.float32,
        device=q.device
    )

    # Create accumulation tensors
    num_sm_parts = tile_scheduler_metadata.size(0)
    total_num_splits = batch_size + num_sm_parts
    
    softmax_lse_accum = torch.empty(
        (total_num_splits, num_heads, q_seq_per_hk),
        dtype=torch.float32,
        device=q.device
    )
    out_accum = torch.empty(
        (total_num_splits, num_heads, q_seq_per_hk, head_size_v),
        dtype=torch.float32,
        device=q.device
    )
    cute_q_dtype = Float16 if q_dtype==torch.float16 else BFloat16

    # Prepare parameters dictionary
    mla_obj = MLA_KVCACHE_FWD(
        batch_size=batch_size,
        seqlen_q_ori=seqlen_q_ori,
        q_seq_per_hk=q_seq_per_hk,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_blocks=num_blocks,
        page_block_size=page_block_size,
        num_q_heads_per_hk=num_q_heads_per_hk,
        is_causal=is_causal,
        head_size_k=head_size_k,
        head_size_v=head_size_v,
        softmax_scale=softmax_scale,
    )
    def create_cute_tensor(pt: torch.Tensor, leading_dim, stride_order):
        cute_tensor = (
            from_dlpack(pt, assumed_align=16)
            .mark_layout_dynamic(leading_dim=leading_dim)
            .mark_compact_shape_dynamic(
                mode=leading_dim,
                stride_order=stride_order,
                divisibility=8
            )
        )
        return cute_tensor

    # q_ = create_cute_tensor(
    #     # [b, qh*sq, kh, hd] -> [qh*sq, hd, kh, b]
    #     q.permute(1, 3, 2, 0),
    #     1, (3, 0, 2, 1)
    # )
    # kcache_ = create_cute_tensor(
    #     # [num_pages, page_size, kh, hd] -> [page_size, hd, kh, num_pages]
    #     kcache.permute(1, 3, 2, 0),
    #     1, (3, 0, 2, 1)
    # )
    # out_ = create_cute_tensor(
    #     # [b, qh*sq, vh, vd] -> [qh*sq, vd, vh, b]
    #     out.permute(1, 3, 2, 0),
    #     1, (3, 0, 2, 1)
    # )
    q_ptr = make_ptr(
        cute_q_dtype, q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    kcache_ptr = make_ptr(
        cute_q_dtype, kcache.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    out_ptr = make_ptr(
        cute_q_dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    out_accum_ptr = make_ptr(
        Float32, out_accum.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    softmax_lse_accum_ptr = make_ptr(
        Float32, softmax_lse_accum.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )

    mla_obj.launch(
        q_ptr=q_ptr,
        kcache_ptr=kcache_ptr,
        out_ptr=out_ptr,
        softmax_lse=from_dlpack(softmax_lse),
        seqlens_k=from_dlpack(seqlens_k),
        block_table=from_dlpack(block_table),
        tile_scheduler_metadata=from_dlpack(tile_scheduler_metadata),
        num_splits=from_dlpack(num_splits),
        softmax_lse_accum_ptr=softmax_lse_accum_ptr,
        out_accum_ptr=out_accum_ptr
    )
 
    # Reshape output back to original format
    out = out.view(batch_size, seqlen_q_ori, num_q_heads_per_hk, num_heads_k, head_size_v) \
             .transpose(2, 3) \
             .reshape(batch_size, seqlen_q_ori, num_heads_q, head_size_v)
    
    softmax_lse = softmax_lse.view(batch_size, num_heads_k, seqlen_q_ori, num_q_heads_per_hk) \
                             .transpose(2, 3) \
                             .reshape(batch_size, num_heads_q, seqlen_q_ori)
    
    return [out, softmax_lse]

if __name__ == "__main__":
    torch.set_default_device(torch.cuda.current_device())
    torch.manual_seed(22)
    q = torch.randn(32, 1, 64, 576, device="cuda", dtype=torch.float16)
    kvcache = torch.randn(4096, 64, 1, 576, device="cuda", dtype=torch.float16)
    seqlens_k = torch.tensor([4000]*32, device="cuda", dtype=torch.int32)
    page_table = torch.randperm(2048, dtype=torch.int32, device="cuda")
    page_table = page_table.reshape(32, 64)

    tile_scheduler_metadata = torch.zeros(78, 8, device="cuda", dtype=torch.int32)
    tile_scheduler_metadata[:, 0] = torch.arange(78, dtype=torch.int32)
    tile_scheduler_metadata[:, 2] = torch.arange(78, dtype=torch.int32)
    tile_scheduler_metadata[:, 4] = torch.arange(78, dtype=torch.int32)
    tile_scheduler_metadata[:32, 3] = seqlens_k
    num_splits = torch.arange(33, device="cuda", dtype=torch.int32)

    from flash_mla import flash_mla_with_kvcache
    # t1, n1 = get_mla_metadata(
    #     seqlens_k,
    #     64,   # num_q_tokens_per_head_k (seqlen_q_ori * num_heads_q // num_heads_k)
    #     1,    # num_heads_k
    #     64,   # num_heads_q
    # )
    ref_o, ref_lse = flash_mla_with_kvcache(
        q, kvcache,
        page_table, seqlens_k,
        512,
        tile_scheduler_metadata, num_splits,
        192**-0.5, False
    )

    o, lse = mla_kvcache_fwd(
        q, kvcache, 512,
        seqlens_k,
        page_table,
        192**-0.5,
        False,
        tile_scheduler_metadata,
        num_splits,
    )

    def calc_cos(a, b):
        a = a.double()
        b = b.double()
        ret = 1 - 2 * (a * b).sum() / max((a * a + b * b).sum(), 1e-12)
        return ret.item()

    o_cos = calc_cos(ref_o, o)
    o_max_abs = (ref_o.float() - o.float()).abs().max().item()
    lse_cos = calc_cos(ref_lse, lse)
    lse_max_abs = (ref_lse.float() - lse.float()).abs().max().item()
    print(f"seqlen={seqlens_k} (is_no_split=True)")
    print(f"  lse_cos={lse_cos}, lse_max_abs={lse_max_abs}")
    print(f"  o_cos={o_cos}, o_max_abs={o_max_abs}")
