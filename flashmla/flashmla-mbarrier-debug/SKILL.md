---
name: flashmla-mbarrier-debug
description: Diagnose and fix FlashMLA CuTe DSL hangs or incorrect outputs in flashmla/flashmla_dsl.py by tracing mbarrier/TMA producer-consumer dependencies against flashmla/splitkv_mla.cu. Use when execution stalls around qkt_gemm_one_tile_sQ mbarrier_wait, when "before wait" appears without "after wait", or when tensor o / calc_cos validation fails.
---

# FlashMLA Mbarrier Debug

## Scope

- Focus on `flashmla/flashmla_dsl.py`.
- Use `flashmla/splitkv_mla.cu` only as immutable reference.
- Keep these Python-vs-CUDA differences unchanged unless explicitly requested:
  - Preserve `if warpgroup_idx == 1` launch path.
  - Preserve `if warp_idx % 4 == 0` launch gate.

## Workflow

1. Reproduce with deterministic command
2. Map synchronization dependency chain
3. Add minimal `cute.printf` probes
4. Pinpoint mismatch and patch minimally
5. Re-run to verify hang removal and numeric quality
6. Fall back to IR/PTX artifact inspection if needed

## 1) Reproduce

Run with the environment's Python binary directly:

```bash
CUTE_DSL_KEEP_PTX=0 CUTE_DSL_KEEP_IR=0 \
/home/wuguanyu02/miniconda3/envs/fllm2/bin/python -u flashmla/flashmla_dsl.py
```

Treat `qkt_gemm_one_tile_sQ before wait` without matching `after wait` as deadlock signal.

## 2) Build Dependency Map

For each barrier used by consumer wait, identify:
- Which warpgroup/warp issues the TMA copy (`producer`)
- Which code path calls `mbarrier_wait` (`consumer`)
- Expected phase value transition (`cur_phase`)
- Barrier address and tile index mapping

Use these anchor points:
- Consumer path: `warpgroup_cooperative_qkt_gemm` -> `qkt_gemm_one_tile_sQ/rQ`
- Producer path: `launch_kv_tiles_copy_tma`
- Initial K0 wait path: first `for i in range_constexpr(9)` in warpgroup 0 branch
- Reference behavior: `splitkv_mla.cu` `launch_kv_tiles_copy_tma` and QKT pipeline

## 3) Instrument with Minimal `cute.printf`

Add only short probes:
- Producer launch: print `tid`, `warp_idx`, `start/end`, `i`, `barrier ptr`
- Consumer request: print `phase`, `tile_idx`, `barrier ptr`, `cur_phase`
- Wait boundaries: existing `before wait`/`after wait` prints

Do not spam all threads. Restrict to one lane per warpgroup, e.g. `tid==0` or `tid==128`.

## 4) Diagnose from Logs

Primary checks:
- Consumer barrier has corresponding producer launch before wait.
- No barrier index outside valid tile range.
- No unintended duplicate launch on same barrier phase.
- `cur_phase` type/value matches wait expectations.

Known pitfall seen in this repo:
- In `launch_kv_tiles_copy_tma`, using `range_constexpr(start, end+1)` with calls like `(4, 9)` and `(0, 4)` can create overlap/out-of-range effects in the 9-tile pipeline.
- This can break barrier state and stall `mbarrier_wait`.

## 5) Patch Strategy

Patch one cause at a time and re-run immediately.

Priority order:
1. Fix producer/consumer index or phase mismatch
2. Re-run and confirm wait no longer stalls
3. Remove temporary debug prints only after behavior is stable

Keep immutable constraints:
- Do not edit `flashmla/splitkv_mla.cu`.
- Keep the two fixed Python choices above unless user requests otherwise.

## 6) Verification Checklist

Required:
- Kernel run exits normally.
- `tensor o` is printed (if current main script prints it; otherwise add temporary print).
- `calc_cos` for both batches is below `1e-4`.

If the run still fails or hangs:
- Enable artifacts and rerun:

```bash
export CUTE_DSL_KEEP_PTX=1
export CUTE_DSL_KEEP_IR=1
/home/wuguanyu02/miniconda3/envs/fllm2/bin/python -u flashmla/flashmla_dsl.py
```

Then inspect generated IR/PTX around:
- `mbarrier_arrive_and_expect_tx`
- `mbarrier_wait`
- TMA copy issue order for each tile
