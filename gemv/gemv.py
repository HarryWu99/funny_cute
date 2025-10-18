# ref zartbot's blog 
import torch
import math
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.jit
def warp_reduce(
    val : cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE
):
    for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

@cute.kernel
def block_gemv_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()

    lane_id = cute.arch.lane_idx()
    warp_id = cute.arch.warp_idx()
    warp_num = bdimx // cute.arch.WARP_SIZE

    blk_coord = (None, (bidx,bidy))

    # logical coord -> address
    # blkA = gA[blk_coord]  # (TileM, TileN) -> physical address
    # print(f"blkA={blkA}")
    # tidfrgA = cute.composition(blkA, tv_layout)
    # print(f"tidfrgA={tidfrgA}")

    # thr_coord = (tidx, None)
    # thrA = tidfrgA[thr_coord]  # (V) -> physical address
    # a_vec = thrA.load()

    # b_tiler = cute.make_layout(tidfrgA.shape[1], stride=1)
    # blkB = cute.zipped_divide(gB, tiler=b_tiler)
    # print(f"gB={gB} b_tiler={b_tiler} blkB={blkB}")
    # b_vec = blkB[(None,tidx)].load()

    # thread_sum = 0.0
    # for i in cutlass.range(tidfrgA.shape[1]):
    #     thread_sum += a_vec[i] * b_vec[i]

    # warp_sum = warp_reduce(thread_sum,lambda x,y: x +y)
        
    # smem = cutlass.utils.SmemAllocator()
    # reduce_buffer = smem.allocate_tensor(
    #     element_type= cutlass.Float32,
    #     layout=cute.make_layout(shape=(16), stride=(1)),
    #     byte_alignment=16,
    # )

    # if lane_id == 0 :
    #     reduce_buffer[warp_id] = warp_sum
    
    # cute.arch.barrier()

    # sum = 0.0
    # if (warp_id == 0):       
    #     if (tidx < warp_num):
    #         warp_sum = reduce_buffer[tidx]
    #     else:
    #         warp_sum = 0.0
    #     sum = warp_reduce(warp_sum , lambda x,y : x+y)

    # if (tidx == 0):
    #     gC[bidx] = sum 
    
    
@cute.jit
def block_gemv(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    
    num_thread_per_block = 512
    num_elements_per_thread = mA.shape[1] // num_thread_per_block
    
    thr_layout = cute.make_layout((1, num_thread_per_block), stride=(num_thread_per_block, 1))
    val_layout = cute.make_layout((1, num_elements_per_thread), stride=( num_elements_per_thread, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}")
    print(f"TV Layout: {tv_layout}")

    # gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    print(f"Tiled Input Tensors:")
    # print(f"  gA: {mA.type}, shape {gA.shape[1][0]}")

    block_gemv_kernel(
        mA, mB, mC, tv_layout, tiler_mn
    ).launch(
        grid=[mA.shape[0], 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )
    
M, N = 4096, 4096

a = torch.randn(M, N, device="cuda", dtype=torch.float32)
b = torch.randn(N, device="cuda", dtype=torch.float32)
c = torch.zeros(M, device="cuda", dtype=torch.float32)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c)

# block_gemv_ = cute.compile(block_gemv, a_, b_, c_)
block_gemv(a_, b_, c_)

# verify correctness
torch.testing.assert_close(c,torch.mv(a, b),atol=1e-4, rtol=1.3e-6) 