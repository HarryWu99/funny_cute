import torch

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, BFloat16, Float16, Float32, Int32, Int64, Boolean
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

@cute.kernel
def mykernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(Float16, cute.make_ordered_layout((64,128), order=(1,0)))
    cute.autovec_copy(mA, sA)
    if tid < sA.shape[0]:
        sm90_bulk_copy_s2g(sA[tid,None].iterator, mB[tid,None].iterator, Int32(sA.shape[1] * 2))
        cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0)

@cute.jit
def launcher(
    mA: cute.Tensor,
    mB: cute.Tensor,
):
    mykernel(
        mA, mB
    ).launch(
        grid=[1,1,1],
        block=[128, 1, 1],
        cluster=[1, 1, 1],
    )

torch.manual_seed(22)
a = torch.randn(64, 128, dtype=torch.float16, device="cuda")
b = torch.zeros(64, 128, dtype=torch.float16, device="cuda")

a_ = from_dlpack(a, assumed_align=128)
b_ = from_dlpack(b, assumed_align=128)

launcher(a_, b_)

print(f"{a=}")
print(f"{b=}")
torch.allclose(a, b)
