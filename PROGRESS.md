# Progress

## Milestone 0 — audit and preparation

- Inspected README, `solution/`, `harness/`, `task/`, `tests/`, and the eight-cell Kaggle notebook.
- Documented execution, admission, and a first kernel investigation in `PROJECT.md`.
- Added reproducible Kaggle T4 baseline instructions and an empty measurement log in `BENCHMARKS.md`.
- No kernels or training settings changed. No Kaggle T4 measurement has been made in this milestone.

## Milestone 2 — GPU profiling and kernel selection

- Recorded user-verified Kaggle T4 results (repeats=1): base 79.91 s / 14,345 MiB / loss 2.04966; starter 83.37 s / 12,385 MiB / loss 2.04966. 8/10 checks and all 79 regression tests passed; only own-kernel checks 3 and 4 failed. Raw artifacts and run provenance have not yet been provided.
- Inspected the official profiler and own-kernel timer. Added exact commands and a small evidence export to `BENCHMARKS.md`, using a separate output directory to preserve the existing baseline.
- Actual per-operation GPU time remains unknown: no `profile.json` is available locally. The README's ~44% elementwise/casts, ~36% GEMM, ~7% softmax/cross-entropy, ~5% attention, and ~5% NF4 figures are organizer base-run references, not the starter's measured distribution. They cannot identify which operations dominate the supplied run or explain its 3.46 s wall-time difference.
- No kernel implementation or changes to protected source, notebook, or training settings.
- Verification: all 20 files under `solution/`, `harness/`, `task/`, `tests/`, and `templates/` match their pre-edit SHA-256 hashes; Bash/Python instruction cells passed syntax checks. No GPU profiling was executed locally; the required runtime/tooling is absent.

### Candidate assessment (hypotheses pending profile)

| Candidate | Potential benefit and required evidence | Forward/backward and integration risks | Decision |
|---|---|---|---|
| Fused Qwen2 RMSNorm | Fuse normalization reductions, scaling and casts; establish its actual call path and cost before coding. The broad elementwise category does not prove a large RMSNorm share. | Preserve epsilon, fp32 accumulation, intermediate rounding/output dtype, strides and masks. Frozen norm weights do not remove the need for input gradients to upstream LoRA layers; handle weight gradients if required. | Safest narrow first correctness experiment, but no evidence that it alone meets 5%. |
| Fused entropy/token-accuracy reductions | Avoid intermediates for required full-vocabulary metrics. Inspect installed TRL/Soup source and kernel attribution to confirm cost and whether metrics are detached. | Preserve shift/mask, token denominators, aggregation, and argmax tie handling. No custom backward is needed only if the existing metric path is detached. | Investigate alongside RMSNorm; share unknown. |
| Cross-entropy forward/backward with required loss/metric reductions | Full-vocabulary reductions are an explicit repository target; potentially broader useful work than normalization. Profile before choosing the scope. | Stable log-sum-exp, causal shift, ignored labels, reduction/gradient scaling, precision, per-row rewind losses, entropy and accuracy must match. Preserve logits needed by other consumers. Large vocabulary raises reduction/resource costs. | Stronger scope to investigate for coverage, with higher correctness/integration risk; not selected yet. |
| NF4 dequantization, GEMM or attention replacement | Each is reported in the organizer profile; measure the starter first. | Packed/double-quant scales and dtype fidelity for NF4; matrix-multiply accumulation/performance; attention masking and backward semantics. Existing library paths set the baseline. | Defer until a measured bottleneck justifies the larger scope. |

Recommendation: inspect the unchanged starter profile first. Keep RMSNorm as a narrow candidate, and prioritize attribution of vocabulary loss/metrics before committing to a kernel family. Generic CUDA kernel names may require follow-up inspection of the pinned installed source; the official export has no operator shapes or stacks. Do not treat a generic reduction as proof of RMSNorm time.

For admission, the **optimized** original kernels must together consume at least 5% of synchronized step-3+ wall time in every run. Multiple useful kernels and their backward paths can qualify together. The example's time is excluded. Fuse required work efficiently, then measure; if RMSNorm falls short, add another measured useful target rather than redundant work or artificial delay. All task/output and quality checks remain mandatory.

## Milestone 3 — original vocabulary CE experiment

- The user supplied a real T4 category profile: GEMM 39.82%, copies/dtype 18.77%, elementwise 16.30%, log-softmax forward/backward 6.49% (~55.7 ms/step), excluded example SwiGLU 5.47%, and NF4 4.82%. Prior step-3+ time is 50.566 s; 5% of that is 2.5283 s. Raw kernel attribution was not supplied, so the category's split between training CE, entropy, and rewind is unknown.
- Inspected SHA-verified published Soup 0.77.0, TRL 0.29.1, and Transformers 5.16.1 sources. `LOSS_PATH.md` traces labels, model loss, entropy, accuracy, and rewind with package locations and wheel hashes.
- Implemented `solution/vocab_ce.py`: original tiled stable CE forward/backward, guarded at `loss_utils.fixed_cross_entropy`, connected in `solution.apply()`. The existing SwiGLU source is unchanged. No RMSNorm fallback was needed because a narrow vocabulary integration exists.
- Executed the complete solution suite in Triton's CPU interpreter: **39 passed, 4 skipped** (the unchanged starter CUDA-only tests). Tests include full vocabulary, masking/reductions/scaled gradients, fp16 causal-loss casting, the real Soup/TRL loss call chain with exact metric/rewind equality, and a tiny real Qwen2 model's upstream projection gradient.
- All three new kernels compile offline for T4/SM75. **79 official regression tests passed** with `SOUP_APPLY_SOLUTION=1` in the isolated CPU environment. Harness source scanning recognizes three original candidates, excludes both starter kernels, and finds no banned imports, lookalikes, or parse errors.
- **Unverified:** CUDA numerical execution, new Kaggle full-run performance/quality, and admission checks 3/4. Interpreter/compilation/CPU regressions are not a GPU admission pass. Replacing training CE covers only part of the log-softmax category and may fall below 5% after optimization. No artificial work was added to increase its share.

## Next action

Run the exact T4 validation cell in `LOSS_PATH.md`: all 43 solution tests with CUDA available, paired full training with `--repeats 2`, then the unchanged admission checker and scorer. Return `runs_m3/result.json`, `runs_m3/checks.json`, per-solution `metrics.json`, and logs. Do not record a new benchmark or declare admission until those results exist.
