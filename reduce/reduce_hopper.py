# Reduce forward implementation adapted from the `quack` repository

import torch
from functools import partial
import math
import operator
from typing import Callable, Optional

import cutlass
from cutlass import const_expr, Float16, Float32, Int32
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda

@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)

@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: cute.Int32, *, loc=None, ip=None
) -> cutlass.Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

@dsl_user_op
def store_shared_remote(
    val: float | cutlass.Float32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), "val must be Float32, Int32, or Int64"
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

@cute.jit
def cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: Optional[cutlass.Int32] = None,
):
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_cta, warps_per_row, cluster_n = reduction_buffer.shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_cta * warps_per_row
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                num_warps * cluster_n * reduction_buffer.element_type.width // 8,
            )
    cute.arch.sync_threads()
    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, col_idx, cta_rank_in_cluster)),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < warps_per_row*cluster_n:
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, *cute.idx2crd(idx, (warps_per_row, cluster_n))])
    return warp_reduce(block_reduce_val, op)

@dsl_user_op
def domain_offset_i64(coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(
        flat_stride
    ), "Coordinate and stride must have the same length"
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)

@cute.kernel
def reduce_kernel(
    mX: cute.Tensor,
    mY: cute.Tensor,
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
    cluster_n: cutlass.Constexpr,
):
    warp_size = cute.arch.WARP_SIZE
    tid, _, _ = cute.arch.thread_idx()
    lane_id = tid % warp_size
    warp_id = tid // warp_size
    threads = cute.size(tv_layout, mode=[0])
    num_warp = threads // warp_size
    threads_per_row = tv_layout.shape[0][0]
    warps_per_row = max(threads_per_row // cute.arch.WARP_SIZE, 1)
    bx, by, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sX = smem.allocate_tensor(
        mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
    )
    reduction_buffer = smem.allocate_tensor(
        cute.Float32,
        cute.make_ordered_layout(
            (num_warp // warps_per_row, warps_per_row, cluster_n),
            order=(2, 0, 1)
        ),
        byte_alignment=16,
    )
    if const_expr(cluster_n > 1):
        mbar_ptr = smem.allocate_array(
            cutlass.Int64, num_elems=1
        )
        if tid == 0:
            cute.arch.mbarrier_init(mbar_ptr + tid, 1)
        cute.arch.mbarrier_init_fence()
        # Cluster arrive after barrier init
        cute.arch.cluster_arrive_relaxed()
    else:
        mbar_ptr = None
    
    # Error! local_tile int32 will out of bound
    # gX = cute.local_tile(mX, tiler_mn, (bx, by))
    gX = domain_offset_i64((bx*tiler_mn[0], 0), mX)
    gX = cute.local_tile(gX, tiler_mn, (0, by))
    tiled_g2s_copy = cute.make_tiled_copy(
        cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        ),
        tv_layout, tiler_mn
    )
    X_g2s_thr_copy = tiled_g2s_copy.get_slice(tid)
    tXgX = X_g2s_thr_copy.partition_S(gX)
    tXsX = X_g2s_thr_copy.partition_D(sX)
    print(f"tXgX={tXgX}")
    print(f"tXsX={tXsX}")
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=128
    )
    cute.copy(copy_atom, tXgX, tXsX)
    tXrX = cute.make_fragment_like(tXgX)
    cute.autovec_copy(tXsX, tXrX)

    x: cute.TensorSSA = tXrX.load().to(cute.Float32)
    init_val = 0.0
    val = x.reduce(cute.ReductionOp.ADD, init_val=init_val, reduction_profile=0)

    # warp_reduce(val, operator.add)
    val = cute.arch.warp_reduction(val, operator.add)

    row_idx, col_idx = warp_id // warps_per_row, warp_id % warps_per_row
    if const_expr(cluster_n == 1):
        if lane_id == 0:
            reduction_buffer[row_idx, col_idx, 0] = val
        cute.arch.sync_threads()

        val = init_val
        if lane_id < warps_per_row:
            val = reduction_buffer[row_idx, lane_id, 0]
        cute.arch.sync_threads()
        val = cute.arch.warp_reduction(val, operator.add)
    else:
        cute.arch.cluster_wait()
        val = cluster_reduce(val, operator.add, reduction_buffer, mbar_ptr)
        pass

    # gY = cute.flat_divide(mY, (tiler_mn[0],))
    # print(f"div gY={gY}")
    # gY = gY[None, bx]
    print(f"mY={mY}")
    gY = cute.local_tile(mY, (tiler_mn[0],), (bx, ))
    print(f"gY={gY}")
    if col_idx==0 and lane_id==0 and by==0:
        gY[row_idx] = cute.Float16(val)

def get_cluster_n(N):
    # 16bit dtype
    if N <= 16 * 1024:
        cluster_n = 1
    elif N <= 32 * 1024:
        cluster_n = 2
    elif N <= 64 * 1024:
        cluster_n = 4
    elif N <= 128 * 1024:
        cluster_n = 8
    else:
        cluster_n = 16
    return cluster_n

def get_threads_per_row(N):
    """Calculate the number of threads per row for the RMSNorm kernel."""
    if N <= 64:
        return 8
    elif N <= 128:
        return 16
    elif N <= 3072:
        return 32
    elif N <= 6144:
        return 64
    elif N <= 16384:
        return 128
    else:
        return 256

def get_num_threads(N):
    return 128 if N <= 16384 else 256

@cute.jit
def reduce_launcher(
    x: cute.Tensor,
    y: cute.Tensor,
):
    m, n = x.shape
    threads = get_num_threads(n)
    cluster_n = get_cluster_n(n)
    threads_per_row = get_threads_per_row(n)

    vecsize = 128 // x.element_type.width
    num_blocks_N = cute.ceil_div(n // vecsize, threads_per_row * cluster_n)

    # shape_dim_1 = max_num_per_sm // copy_elems
    cols_per_cta = threads // threads_per_row
    tv_layout = cute.make_ordered_layout(
        ((threads_per_row, cols_per_cta), (vecsize, num_blocks_N)),
        order=((2, 0), (1, 3)),
    )
    tiler_mn = (cols_per_cta, vecsize * num_blocks_N * threads_per_row)
    print(f"tv_layout={tv_layout}")
    print(f"tiler_mn={tiler_mn}")
    print(f"cluster_n={cluster_n}")
    grid = [cute.ceil_div(m, cols_per_cta), cluster_n, 1]
    print(f"grid={grid}")
    reduce_kernel(
        x, y, tiler_mn, tv_layout, cluster_n
    ).launch(
        grid=grid,
        block=[threads, 1, 1],
        cluster=[1, cluster_n, 1],
    )


torch.manual_seed(22)
for i in [2, 4, 16, 32, 64, 128]:
    M, N = 32*1024, i*1024
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    y = torch.zeros(M, device="cuda", dtype=torch.float16)
    x_ = from_dlpack(x, assumed_align=16)
    y_ = from_dlpack(y)

    compiled_reduce = cute.compile(reduce_launcher, x_, y_)
    compiled_reduce(x_, y_)

    print(y)
    print(x.sum(-1))

    def get_running_time(func):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for i in range(32):
            func()
        start_event.record()
        num_iter = 64
        for _ in range(num_iter):
            func()
        end_event.record()
        torch.cuda.synchronize()

        time_ms = start_event.elapsed_time(end_event)
        return time_ms / num_iter

    ref_y = torch.zeros(M, device="cuda", dtype=torch.float16)
    torch_time = get_running_time(lambda: torch.sum(x, dim=-1, out=ref_y))
    dsl_time = get_running_time(lambda: compiled_reduce(x_, y_))
    gbs = M*N * 2 / 1024**3
    print(f"{i},{gbs/(torch_time*10**-3):.3f},{gbs/(dsl_time*10**-3):.3f}")
    # print(f"torch bandwidth: {gbs/(torch_time*10**-3):.3f}")
    # print(f"dsl bandwidth: {gbs/(dsl_time*10**-3):.3f}")
