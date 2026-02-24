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

@cute.kernel
def mykernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    tiled_mma: cute.TiledMma,
):
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    smem_layout_atom = make_smem_layout_atom(
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
        Float16,
    )
    layout = cute.tile_to_shape(smem_layout_atom, (64,128), order=(0,1))
    sA = smem.allocate_tensor(Float16, layout.outer, swizzle=layout.inner)
    sB = smem.allocate_tensor(Float16, layout.outer, swizzle=layout.inner)
    thr_mma = tiled_mma.get_slice(tid)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCrA = thr_mma.make_fragment_A(tCsA)
    tCrB = thr_mma.make_fragment_B(tCsB)
    tCrC = thr_mma.make_fragment_C(thr_mma.partition_shape_C((64, 64)))
    tCrC.fill(0)
    cute.autovec_copy(mA, sA)
    cute.autovec_copy(mB, sB)

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), Float16,
    )
    s2r_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
    thr_copy = s2r_copy.get_slice(tid)
    thr_copy_sA = thr_copy.partition_S(sA)
    thr_copy_rA = thr_copy.retile(tCrA)
    cute.copy(copy_atom, thr_copy_sA, thr_copy_rA)
    if tid==0:
        cute.printf("tCrA={}", tCrA)

    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
    cute.nvgpu.warpgroup.fence()
    cute.gemm(
        tiled_mma,
        tCrC,
        tCrA,
        tCrB,
        tCrC,
    )
    cute.nvgpu.warpgroup.commit_group()
    cute.nvgpu.warpgroup.wait_group(0)

    if tid==0:
        cute.printf("tCrC={}", tCrC)

@cute.jit
def launcher(
    mA: cute.Tensor,
    mB: cute.Tensor,
):
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        Float16,
        Float16,
        OperandMajorMode.K,
        OperandMajorMode.K,
        Float32,
        atom_layout_mnk=(1,1,1),
        tiler_mn=(64, 64),
        a_source=OperandSource.RMEM,
    )
    mykernel(
        mA, mB,
        tiled_mma,
    ).launch(
        grid=[1,1,1],
        block=[128, 1, 1],
        cluster=[1, 1, 1],
    )

torch.manual_seed(22)
a = torch.randn(64, 128, dtype=torch.float16, device="cuda")
b = torch.randn(64, 128, dtype=torch.float16, device="cuda")

a_ = from_dlpack(a, assumed_align=128)
b_ = from_dlpack(b, assumed_align=128)
launcher(a_, b_)

ref_c = a @ b.T

print(ref_c[0, :2])
print(ref_c[8, :2])
