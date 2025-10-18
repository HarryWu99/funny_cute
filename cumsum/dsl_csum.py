import torch
import math
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def naive_csum_kernel(
    gA: cute.Tensor,
    gO: cute.Tensor,
    tiler_mn: cute.Shape,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    m, n = gA.shape
    mA = cute.local_tile(
        gA, tiler_mn, (bidx,0)
    )
    mO = cute.local_tile(
        gO, tiler_mn, (bidx,0)
    )
    copy_atom_load_A = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(), gA.element_type, num_bits_per_copy=128
    )
    thr_copy_A = cute.make_tiled_copy(copy_atom_load_A, tv_layout, tiler_mn).get_slice(tidx)
    
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        element_type=gA.element_type,
        layout=cute.make_ordered_layout(tiler_mn, order=(1, 0)),
        byte_alignment=16,
    )
    tAgA = thr_copy_A.partition_S(mA)
    tAsA = thr_copy_A.partition_D(sA)
    cute.copy(copy_atom_load_A, tAgA, tAsA)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    # todo: predict
    num_threads_per_block = tv_layout.shape[0]
    iter_num = tv_layout.shape[1]
    log2n = int(math.log2(num_threads_per_block))
    cur_sum = cutlass.Int32(0)
    for j in cutlass.range_constexpr(iter_num):
        start_smem_idx = j * num_threads_per_block
        end_smem_idx = start_smem_idx + num_threads_per_block
        if tidx == 0:
            sA[start_smem_idx] += cur_sum
        cute.arch.sync_threads()       
        # up-sweep
        for i in cutlass.range_constexpr(log2n):
            stride = 1 << i
            idx = start_smem_idx + (tidx + 1) * 2 * stride - 1
            if idx < end_smem_idx:
                sA[idx] += sA[idx - stride]
            cute.arch.sync_threads()
            # if tidx == 0:
            #     cute.printf("i={} s={} {}", i, stride, sA)
        # # down-sweep
        for i in cutlass.range_constexpr(log2n):
            stride = 1 << (log2n - 1 - i)
            idx = start_smem_idx + (tidx + 1) * 2 * stride - 1
            if idx + stride < end_smem_idx:
                sA[idx+stride] += sA[idx]
            cute.arch.sync_threads()
            # if tidx == 0:
            #     cute.printf("i={} s={} {}", i, stride, gO)
        cur_sum = sA[end_smem_idx-1]
    # if tidx==0:
    #     sA_tile = cute.zipped_divide(sA, (1, 512))
    #     cute.printf("sA_tile={}", sA_tile)
    copy_atom_store_O = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=128
    )
    thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler_mn).get_slice(tidx)
    tOsO = thr_copy_O.partition_S(sA)
    tOgO = thr_copy_O.partition_D(mO)
    cute.copy(copy_atom_store_O, tOsO, tOgO)

    
@cute.jit
def naive_csum(
    mA: cute.Tensor,
    mO: cute.Tensor,
):
    num_threads_per_block = 512
    m, n = mA.shape
    num_elements_per_thread = math.ceil(n / num_threads_per_block)
    thr_layout = cute.make_ordered_layout((1, num_threads_per_block), order=(1, 0))
    # thr_layout = cute.make_layout((num_threads_per_block,), stride=(1,))
    val_layout = cute.make_ordered_layout((1, num_elements_per_thread), order=(1, 0))
    # val_layout = cute.make_layout((num_elements_per_thread,), stride=(1,))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"tiler_mn={tiler_mn}")
    print(f"tv_layout={tv_layout}")
    kernel = naive_csum_kernel(mA, mO, tiler_mn, tv_layout)
    kernel.launch(grid=(m, 1, 1),
                  block=(num_threads_per_block, 1, 1))

M, K = 3166, 2048

a = torch.ones(M, K, device="cuda", dtype=torch.int32)
# a = torch.arange(0, 2048, device="cuda", dtype=torch.int32).view(1, 2048)
o = torch.zeros(M, K, device="cuda", dtype=torch.int32)

a_ = from_dlpack(a, assumed_align=16)
o_ = from_dlpack(o, assumed_align=16)

# naive_csum(a_, o_)
# Compile kernel
naive_csum_ = cute.compile(naive_csum, a_, o_)
naive_csum_(a_, o_)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# warmup
for i in range(32):
    torch.cumsum(a, dim=1)

torch.cuda.synchronize()
start_event.record()
for i in range(32):
    torch.cumsum(a, dim=1)
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
print(f"torch elased {elapsed_time/32:.3f}ms")

torch.cuda.synchronize()
start_event.record()
for i in range(32):
    naive_csum_(a_, o_)
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
print(f"dsl elased {elapsed_time/32:.3f}ms")

# verify correctness
torch.testing.assert_close(o, torch.cumsum(a, dim=1).to(torch.int32), atol=1e-4, rtol=1.3e-6)
