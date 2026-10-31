# Soup Kernel Challenge: project map

## Fixed task and flow

`task/task.yaml` fixes Qwen2.5-1.5B-Instruct SFT, 4-bit weights, LoRA r=16/alpha=32, batch 4, 512-token cap, one epoch, and seed 1. `task/prepare_data.py` pins the dataset revision and writes public training rows 0–255 and eval rows 40000–40255. The README states 64 training steps and hidden jury rows.

The Kaggle notebook installs pinned dependencies, clones the selected commit, prepares data, tests `solution/`, runs `harness/run_task.py`, then runs `harness/check_rules.py` and `harness/score.py`. `run_task.py` warms imports/model download before timing, creates a fresh per-run config, and launches separate base and solution `_train.py` processes. The solution process calls `solution.apply()` after CUDA initialization and before Soup creates its model/trainer. `apply()` preserves the shipped SwiGLU MLP patch and, from Milestone 3, installs a guarded original vocabulary CE forward/backward at `transformers.loss.loss_utils.fixed_cross_entropy`. The example does **not** count as an own kernel. See `LOSS_PATH.md` for the exact integration and verification limits.

`run_task.py` measures whole training-command wall time, including compilation, and peak `nvidia-smi` GPU memory. `_train.py` records trainer settings, steps, and CUDA-event time for `solution/` Triton kernels from step 3 onward. `eval_adapter.py` loads the saved adapter without `solution/` and computes answer-token loss. `run_task.py` writes `runs/result.json`; `check_rules.py` writes `runs/checks.json`. `score.py` accepts results only when all 10 checks pass and the checks file matches the result hash. From Milestone 4.2, `solution.apply()` also patches TRL's imported entropy helper for its detached metric on eligible CUDA logits; the user reports 1.445% eligible share at `b45a74b`. Milestone 4.3 added a guarded nested-NF4 dequantization fusion; the user-verified `ed35c4b` result passes all 10 checks with 7.34% eligible share, 1.038x speed and 1.158x memory ratio.

## Admission gates

All runs finish; no banned libraries or run-time installs; original own Triton kernels launch in every solution run and total at least 5% of training time after step 2; task settings, adapter structure, tokens, logged metrics, and rewind rows match; first-step loss gap ≤1%, mean curve gap ≤3%, eval loss ≤0.5% above base; time and memory ≤110% of base; clean single-T4 measurement; all 79 Soup regression tests pass. The jury also reviews code and README. See `harness/check_rules.py` for exact comparisons.

## Optimization targets

**Repository facts:** The README reports a prior base-Soup T4 profile: matrix multiplication ~36% of step GPU time, elementwise operations/casts ~44%, softmax/cross-entropy ~7%, attention ~5%, NF4 dequantization ~5%. These are organizer claims, not measurements from this checkout. The README says the shipped SwiGLU example cuts peak memory by ~2,000 MiB but is excluded from own-kernel checks. Soup computes entropy and token accuracy from full logits and writes per-row rewind losses, so loss-path changes must preserve these outputs.

**Hypotheses to measure:** Qwen2 RMSNorm may offer a narrow original fusion of reduction, scaling, and dtype conversion; its actual time share may be too small for the 5% gate. Full-vocabulary loss/metric fusion may save memory or launches, but is riskier because it affects training loss and required metrics. Attention, GEMM, and NF4 paths may be harder to improve safely on T4 because library kernels already handle them; this is not established without a profile.

**Current decision (Milestone 4.4):** Preserve admitted `ed35c4b` and test one change on `codex/m44-speed`: a balanced bit-selection NF4 codebook lookup. An offline SM75 compile reduces predicate instructions from 137 to 41 without spills; this is compiler evidence, not measured speed. All constants, fp32 rounding, launch settings, integration and SwiGLU stay unchanged. The exact Kaggle procedure in `BENCHMARKS.md` requires two paired repeats, all ten checks, >=1.05x speed, >=1.158x memory ratio and >=5% eligible share. Failed candidates are archived and the Kaggle checkout is restored to `ed35c4b`.
