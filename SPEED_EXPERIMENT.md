# Milestone 4.4 — one NF4 lookup optimization

Known-good commit: `ed35c4b947a0076445ad4ed164b1d55caaf3b729`, retained on `codex/m43-nf4` and `codex/known-good-ed35c4b`. Experiment branch: `codex/m44-speed`. The user-verified T4 result is 89.8 s base, 86.5 s solution, 1.038x speedup, 1.158x memory ratio, 7.34% eligible kernel share, all 10 admission checks passing. Raw current run artifacts were not supplied. No new T4 measurement is claimed here.

## Selected change

Only `nested_nf4_dequant_kernel`'s constant lookup changes. The previous code iterated over 16 values, comparing each code and extending a dependent selection chain. The new code tests the four code bits once, selects adjacent pairs, then groups of 4, 8 and 16. It returns the same exact fp32 constant in four dependent stages. There is no approximate arithmetic, lookup tensor allocation or additional GPU work.

Scale reconstruction, separate fp32 multiply/add rounding (`enable_fp_fusion=False`), nibble order, output fp16 cast, masks, gradients and integration guards remain unchanged. Block size 1024, four warps and `N` specialization remain unchanged. CE, entropy, accuracy, rewind, native GEMMs, LoRA and starter SwiGLU are identical to the admitted commit. The validator is a separate post-run utility, never imported by `apply()`.

## Evidence and alternatives investigated

Controlled offline compilation with Triton 3.5.1, SM75, identical pointer signatures and compiler options, for N = 393216, 2359296 and 13762560 (the three unique supported weight sizes):

| Compiler property | Admitted linear selection | Balanced selection |
|---|---:|---:|
| PTX `setp` instructions in the whole kernel | 137 | 41 |
| PTX `selp` instructions in the whole kernel | 136 | 128 |
| `ptxas` registers/thread | 53 | 54 |
| Spill loads/stores; shared memory | 0 / 0; 0 bytes | 0 / 0; 0 bytes |
| Cubin bytes | 33768 | 29928 |

These are static compiler facts, not dynamic instruction counts or a throughput benchmark. The controlled compile did not add runtime pointer-alignment specializations. Kaggle's compiler version and actual JIT specialization can change generated code. Reduced predicate count and shorter dependence are the reason to test this change; the extra register and memory-bound workload mean a speedup is not assured.

**Launch configuration:** offline SM75 compilation of the admitted lookup used 62 registers at block512/warps4, 80 at block1024/warps2, 53 at block1024/warps4, and 62 at block1024/warps8, all without spills/shared memory. Lower occupancy or increased launch count could negate an apparent instruction benefit. No launch configuration is selected from register counts alone; the measured configuration is preserved.

**Compilation overhead:** `N: tl.constexpr` creates three size variants; the two MLP weight orientations share a size. A separate unsubmitted dynamic-N prototype compiled one generic size variant, but retained the old 137 predicates/136 selections. Its T4 runtime cost is unknown. Compilation uses cold per-run caches in the official harness and is included in scored wall time. The local compiler timings include CPU startup/order effects, so they are not treated as a T4 savings estimate. This experiment makes no specialization change. `nf4_cache_audit.json` will record actual cache metadata counts after each official solution run without changing compilation or exporting caches.

**Starter SwiGLU:** block1024 and the default four warps compile for SM75 using 36 registers forward and 58 backward, no spills/shared memory. The wrapper calls `.to(gate.dtype).contiguous()` on `up` and the incoming gradient. A local CPU storage/profiler probe confirms that matching contiguous tensors reuse storage (an `aten::to` call without `_to_copy`); strided tensors require a copy. Source-derived Qwen gate/up dtypes match, but full T4 gradient-stride evidence was not supplied. Removing these calls is not an established GPU-copy saving and could break unsupported layouts. The starter file stays byte-identical.

## Executed tests and limits

- Full solution suite: **85 passed, 4 skipped** under Triton's CPU interpreter. The four skips are the unchanged CUDA-only starter tests. Expected T4 suite: **89 tests, zero skips**.
- Existing numerical tests cover all 16 NF4 levels, all 256 nested scale codes, scale boundaries, tail blocks, q/v shapes and exact comparisons to bitsandbytes; native forward, input and bias gradients match exactly. The real reduced quantized PEFT/Qwen + AMP + Soup/TRL/rewind test compares all trainable gradients, loss, entropy, accuracy and rewind outputs.
- **79 official regressions passed** with the solution applied. Final modified kernel source compiles offline for all three supported weight sizes on SM75. No local GPU exists, so CUDA execution, speed and admission share for this change remain pending.
- Twelve synthetic CPU tests validate the promotion policy: rejects failed checks, incomplete repeats, wrong result/source hashes, slower/below-target speed, lost memory ratio, nonfinite ratios and a single repeat below 5%. Rollback restores the known-good commit and refuses to overwrite tracked edits or an unexpected HEAD. Synthetic fixture numbers are not benchmark measurements.
- An end-to-end utility smoke test triggers a real failed CUDA probe on this CPU host in a temporary Git repository, confirms candidate source/commit are archived before rollback, verifies ZIP integrity and checks both SHAs in `rollback.json`. It performs no GPU work.

## Promotion and rollback

Use the exact cells in `BENCHMARKS.md`. `--speed-gate` runs two paired official repeats, all ten checks and automatic ZIP export. It requires speedup >=1.05x (reports >=1.08x separately), memory ratio >=1.158x and minimum eligible share >=5%. Any failed gate restores only the clean Kaggle checkout to the full known-good SHA after exporting candidate evidence. The experimental remote branch is retained; no GitHub branch is rewritten by validation.

A faster original NF4 kernel can reduce eligible share. If it crosses below 5%, restore the admitted version; do not slow it deliberately or add work. Cross-session ratios have noise and the historical 1.038x/1.158x values are rounded. A marginal pass is not evidence of robust speedup on hidden rows. Keep `ed35c4b` as the submission candidate until the new paired results establish a passing improvement.
