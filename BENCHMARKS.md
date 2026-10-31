# Kaggle T4 measurements

## Exact starter baseline procedure

1. Push this starter to the **team's hackathon-platform repository** and use that repository's URL in Kaggle. Do not use the public starter repository as the submission repository. Do not change `task/`, `harness/`, `tests/`, or training settings.
2. In Kaggle, create a notebook and import `templates/kaggle_notebook.ipynb` using **File → Import Notebook**. Set **Accelerator: GPU T4 x2** and **Internet: on**. A phone-verified account is needed for Internet. For a private team repository, add a read-only `GITHUB_TOKEN` via **Add-ons → Secrets**; `HF_TOKEN` is optional.
3. In the notebook's first code cell, set `REPO_URL` to the team's clone URL and `COMMIT` to the exact pushed commit SHA. Keep the GPU idle; do not run other GPU work during measurement.
4. Select **Run All**. The notebook checks T4, installs the pinned dependencies, clones that commit afresh, runs `python task/prepare_data.py`, `python -m pytest solution -q`, `python harness/run_task.py`, `python harness/check_rules.py`, and `python harness/score.py runs/result.json`.
5. Confirm the printed commit and inspect `/kaggle/working/repo/runs/result.json`, `runs/checks.json`, and `runs/<run>/train.log`. The unmodified starter's example kernels are excluded from checks 3–4; their failure is expected until an original kernel is added. Record actual results below, including failed checks. Use the separate profiling directory below to preserve these baseline artifacts.

Only Kaggle T4 numbers count. The README's ~79–83 s and ~14,350 MiB are organizer reference figures, not entries in this log.

## Measurement log

| Date | Commit | Kaggle notebook/version | Base wall s | Solution wall s | Base peak MiB | Solution peak MiB | Base eval loss | Solution eval loss | Own kernel share | Checks | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Not supplied | Not supplied | Not supplied | 79.91 | 83.37 | 14345 | 12385 | 2.04966 | 2.04966 | Not supplied | 8/10; only 3 and 4 failed | User-verified Kaggle T4 starter run; repeats=1; 79 regression tests passed |
| Not supplied | `3e15f49` | Not supplied | 87.06 | 86.56 | 14345 | 12385 | 2.0497 | 2.0497 | 0 eligible CE launches | Not supplied | User-reported T4 summary; 2 base + 2 solution runs succeeded; 43/43 CUDA solution tests passed; solution metrics contained only excluded starter SwiGLU launches. Per-run values and summary aggregation were not supplied. |
| Not supplied | `02d49c5` | Not supplied | Not supplied | Not supplied | Not supplied | Not supplied | Not supplied | Not supplied | 0.893% | Not supplied | User-verified T4: 44 CUDA tests passed, original CE kernels executed; eligible kernel time 441.138 ms over 49.38 s synchronized training time. Full paired wall, memory, loss and per-run files were not supplied. |
| Not supplied | `b45a74b` | Not supplied | Not supplied | Not supplied | Not supplied | Not supplied | Not supplied | Not supplied | 1.445% | Not supplied | User-verified T4: original kernel time 677.431 ms; timed training 46.885 s. CUDA test count, raw per-run files, repeats, wall/memory/eval and other gates were not supplied. |
| Not supplied | `ed35c4b` | Not supplied | 89.8 | 86.5 | Not supplied | Not supplied | Not supplied | Not supplied | 7.34% | 10/10 PASS | User-verified Kaggle T4. Reported speedup 1.038x and memory ratio 1.158x. Raw per-run artifacts, repeat count, absolute memory/eval and notebook version were not supplied. |

Source: the user's Milestone 2 report; raw result/check files have not been inspected locally. The starter uses the shipped SwiGLU example, which earns no own-kernel credit. Derived from this pair: wall time +3.46 s (+4.33%), peak memory −1,960 MiB (−13.66%), speedup 0.9585x, memory ratio 1.1583x, reported loss delta 0%. One repeat does not establish timing variance. Run date, commit, notebook version, and numeric share remain unspecified.

## T4 profile supplied for Milestone 3

User-reported real T4 profile; raw per-kernel JSON, date, commit, and notebook version were not supplied.

| Category | GPU time share |
|---|---:|
| GEMM | 39.82% |
| Copies / dtype | 18.77% |
| Elementwise | 16.30% |
| Log-softmax forward/backward | 6.49% (~55.7 ms/step) |
| Starter SwiGLU (excluded from own-kernel share) | 5.47% |
| NF4 dequantization | 4.82% |

The listed categories total 91.67%; the remaining 8.33% was not categorized in the supplied report. Prior full-run synchronized steps 3+ totaled **50.566 s**, giving a reference 5% threshold of **2.5283 s**. The optimized run has its own denominator. The 6.49% category spans operations potentially used by training loss, entropy, and rewind; it is not an eligible-kernel measurement for the new CE implementation.

At the Milestone 3 commit, the original training CE kernels had **no T4 benchmark result**. The later `3e15f49` run above confirms they were not invoked. See `LOSS_PATH.md` for validation instructions. Local interpreter tests and offline SM75 compilation are verification results, not performance measurements.

Milestone 4.1 diagnosis: commit `3e15f49` completed two paired T4 runs but launched none of the original CE kernels in training. The published timing and loss therefore describe the starter path plus an inert CE patch, not CE performance. Local PEFT/Qwen2 tracing showed that the model forwards `output_attentions` and `output_hidden_states` to `fixed_cross_entropy`; the previous wrapper rejected all keyword arguments even though Transformers ignores these two. The narrow guard correction was measured for eligible share at `02d49c5`; its full paired wall, memory, and loss results were not supplied.

Milestone 4.2 arithmetic from the user-verified `02d49c5` measurement: `441.138 ms / (49.38 s × 1000) = 0.893%`. At that run's denominator, 5% would be 2,469 ms of eligible execution, leaving **2,027.862 ms**. This is a planning difference, not a forecast: the next solution changes both the numerator and the timed-loop denominator. The supplied prior category profile does not isolate entropy, accuracy, or rewind, and no new raw `profile.json` was provided. Do not attribute any specific milliseconds to them.

**Repository facts:** TRL's entropy helper materializes chunked full-vocabulary log-softmax values and reduces them without gradients; accuracy separately copies shifted logits and takes a vocabulary argmax; Soup rewind runs per-row detached CE and transfers row summaries to Python. **Hypotheses:** entropy likely contributes to the previous 6.49% log-softmax and 16.30% elementwise categories, while accuracy may contribute to copies/dtype (18.77%) and reductions. None of those contributions is individually measured. GEMM (39.82%) is the largest measured category, but replacing library matrix multiplication or the LM head would require more difficult gradient and performance validation. The narrow entropy kernel is a correctness-first experiment, not evidence that 5% will pass.

## Milestone 4.4: speed experiment and rollback

Known-good: **`ed35c4b947a0076445ad4ed164b1d55caaf3b729`**, all 10 admission checks passed on the user-verified T4 run, eligible share **7.34%**, base **89.8 s**, solution **86.5 s**, speedup **1.038x**, memory ratio **1.158x**. The rounded times imply 1.03815x. Speedup below 1.05x counts as a tie under the official scoring rule.

At an unchanged 89.8 s base, 1.05x requires at most **85.524 s** solution wall time (about **0.976 s** saved from 86.5 s); 1.08x requires **83.148 s** (about **3.352 s** saved). These are target calculations, not results. The experiment changes only the NF4 constant lookup to a balanced bit-selection tree; current launch configuration, compilation specialization and starter SwiGLU remain as measured. See `SPEED_EXPERIMENT.md` for offline compiler evidence and test results. No new T4 performance claim is made.

### Exact Kaggle cells

Use a fresh notebook/session on **GPU T4 x2**, Internet enabled, with both GPUs idle. Import the official template. Set its first code cell to:

```python
REPO_URL = "https://github.com/Sungatss/soup-kernel-experiment.git"
COMMIT = "SET_TO_THE_FULL_M44_SHA_REPORTED_WITH_DELIVERY"
```

Run the official dependency and clone cells, then data preparation. Use these cells in place of the template's separate test/benchmark/check/score cells. No package installation happens during measurements.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/repo
unset TRITON_INTERPRET
python task/prepare_data.py
git rev-parse HEAD
CUDA_VISIBLE_DEVICES=0 python solution/validate_kaggle.py --out runs_m44 --speed-gate
```

The expected CUDA suite is **89 tests with zero skips** (including 12 CPU-only validation-policy tests). The driver runs the complete solution suite, **two paired official repeats** in order base/solution/solution/base, `harness/check_rules.py` with all 10 checks and the 79 official regressions, then `harness/score.py`. Speed mode adds no profile or diagnostic training run. It checks exact result/source hashes, both repeats' original-kernel launches and the minimum eligible share. It records the actual NF4 compiler-cache metadata count after training exits; no compiler caches are reused or exported.

Acceptance requires all ten checks, speedup **>=1.05x**, memory ratio **>=1.158x**, and minimum eligible share **>=5%**. The preferred 1.08x target is reported separately. This deliberately restores the known-good version even for a small speed improvement that misses 1.05x. Historical ratios are a screening reference, not a new same-session rerun of `ed35c4b`; treat small changes as noisy. Do not promote a slower or failed run.

On failure, the driver first writes `/kaggle/working/runs_m44-evidence.zip` containing the **candidate** source, commit, checks and logs. It then switches only the clean Kaggle checkout to detached `ed35c4b` and appends `rollback.json` to the ZIP. It never changes a GitHub branch. If HEAD or tracked files changed during validation, it refuses to overwrite them and records manual rollback instructions. A killed process or full disk may prevent automatic export/rollback; inspect the retained run directory in that case.

Run this Python cell even if validation exits nonzero:

```python
import os
from pathlib import Path
from IPython.display import FileLink, display
archive = Path('/kaggle/working/runs_m44-evidence.zip')
assert archive.is_file(), 'Inspect the validation log and retained run directory'
os.chdir(archive.parent)
display(FileLink(archive.name))
```

Inspect `speed_decision.json`, `launch_audit.json`, `nf4_cache_audit.json`, `checks.json`, `result.json` and, after failure, `rollback.json`. Raw absolute timings, memory and losses remain in official artifacts. Do not rename the previous run's results as this experiment's evidence. Keep the ZIP even on rejection. Choose a fresh `--out` name for every rerun.

Manual rollback, only in a clean checkout after saving evidence:

```bash
cd /kaggle/working/repo
git checkout --detach ed35c4b947a0076445ad4ed164b1d55caaf3b729
```

No Milestone 4.4 Kaggle run has been executed locally. The published experimental branch must not replace the admitted implementation until this gate passes.

## Milestone 4.3: current Kaggle validation and automatic export

Verified `b45a74b` arithmetic: `677.431 / 46885 × 100 = 1.4449%`, consistent with the reported **1.445%**. At that denominator, 5% is **2,344.250 ms**, leaving **1,666.819 ms**. This is a planning gap, not predicted new performance. The prior profile is not a measurement of the new NF4 fusion. The later user-verified `ed35c4b` result is recorded above; the remainder of this section preserves the original M4.3 procedure.

Known-good full SHA: `b45a74b1ce18f5c2baf05e7f86394ebb56a86528`; preserved as `codex/known-good-b45a74b`. Use that SHA in a separate fresh notebook for rollback. Do not overwrite its prior run artifacts.

1. Use the official notebook on **GPU T4 x2**, Internet enabled. Set `REPO_URL = "https://github.com/Sungatss/soup-kernel-experiment.git"` and `COMMIT` to the full Milestone 4.3 SHA reported with delivery. Run its dependency, clone and data-preparation cells. Keep the official package pins and settings. The personal repository is for this experiment; submission still follows the challenge's repository rules.
2. Replace the notebook's execution of the test/benchmark/check/score cells with this single cell (do not run both sequences). It runs the same official commands, two paired repeats plus the official profile, and a separate diagnostic. It stops before benchmarking if numerical tests fail. It automatically creates a ZIP even if admission fails. No dependencies are installed during these commands.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/repo
unset TRITON_INTERPRET
git rev-parse HEAD
CUDA_VISIBLE_DEVICES=0 python solution/validate_kaggle.py --out runs_m43 --diagnose
```

The expected solution-suite count is **77 tests, zero skips on CUDA**. The numerical suite includes a real bitsandbytes 2,048-row forward fallback and backward, and a two-layer quantized PEFT/Qwen + AMP + Soup/TRL/rewind method-chain test. CPU interpreter results are not a substitute for this CUDA run. If a test or check fails, preserve the ZIP; do not relax tolerances or alter training settings to force a pass.

Underlying commands, all in fresh processes:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest solution -q -p no:cacheprovider
python harness/run_task.py --repeats 2 --profile --out runs_m43
python harness/check_rules.py --result runs_m43/result.json
python harness/score.py runs_m43/result.json
CUDA_VISIBLE_DEVICES=0 python -m solution.diagnose_projections --out runs_m43/diagnostic
```

The driver invokes these automatically; the list explains its operation. `check_rules.py` also runs all 79 official regressions with the solution applied. The driver audits positive launches, expected `solution/` file paths and timed execution for all six original kernels in **both** solution runs, including `nested_nf4_dequant_kernel`. `launch_audit.json` prints their sum and share; `checks.json` remains the authority for all admission gates and the minimum share. The starter kernels never count.

`runs_m43/solution_profile/profile.json` is the official kernel table (6 wait + 2 warmup + 16 active steps). The optional `runs_m43/diagnostic/` uses the same official profiling schedule with forward scopes and tensor-shape recording added only in that separate process. It writes:

- `projections.json`: actual model classes, frozen weight/quantization metadata, LoRA dtypes/scaling/dropout, first-call input/output and available gradient dtypes/strides, and AMP state for `lm_head`, `q_proj`, `v_proj`.
- `trace.json`: Chrome trace with `projection::<module path>` forward scopes, CPU operators and correlated CUDA kernels. Inspect forward `aten::linear`/`aten::mm`, their casts, and autograd `MmBackward`/`MatMul4BitBackward` for input-gradient costs. First-layer q/v may have no input gradient because their upstream base is frozen; later layers must propagate to upstream LoRA.
- `operators.json`: operator totals grouped by input shapes. Inclusive totals overlap; do not add them together or equate them to eligible time. Shape recording can retain tensors and change memory/timing. These diagnostic artifacts are **not admission measurements**.

If the output directory or ZIP exists, choose a fresh name (for example `--out runs_m43_retry1`). No old artifacts are deleted. For a manual diagnostic only after a successful download/preparation, the last command above is sufficient with a fresh output directory. No profiling package installation is needed.

3. Even if the Bash cell reports a nonzero exit, run this Python cell to download the automatically exported evidence:

```python
from pathlib import Path
from IPython.display import FileLink, display
import os
archive = Path('/kaggle/working/runs_m43-evidence.zip')
assert archive.is_file(), 'Inspect the driver log if the process was killed before export'
os.chdir(archive.parent)
display(FileLink(archive.name))
```

The ZIP includes commit/status/package versions, command exit codes, numerical/regression logs, paired result/checks/eval/metrics/configs, required rewind/trainer summaries, raw profiles and CSVs, optional projection diagnostics, submitted source and relevant documentation, plus file hashes. It excludes model/adapter weights, compiler caches, databases, and credentials. A killed process or exhausted disk cannot guarantee export; the run directory remains available in Kaggle output files. Failed checks are preserved and return a nonzero exit.

## Milestone 4.2: exact Kaggle validation and export

Push the new commit to the authorized repository, set the notebook's `COMMIT` to its full SHA, and run the provided dependency, clone, and data-preparation cells on a T4. Use a fresh checkout; the previously measured `02d49c5` checkout lacks the entropy kernels. Then run:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/repo
unset TRITON_INTERPRET
test ! -e runs_m42
git rev-parse HEAD
python - <<'PY'
from importlib.metadata import version
for package, expected in [('soup-cli', '0.77.0'), ('trl', '0.29.1'), ('transformers', '5.16.1')]:
    assert version(package) == expected, (package, version(package))
import solution
solution.apply()
from transformers.loss import loss_utils
from trl.trainer import sft_trainer
assert getattr(loss_utils.fixed_cross_entropy, '_soup_vocab_ce', False)
assert getattr(sft_trainer.entropy_from_logits, '_soup_vocab_entropy', False)
PY
CUDA_VISIBLE_DEVICES=0 python -m pytest solution -q
python harness/run_task.py --repeats 2 --out runs_m42
check_status=0
python harness/check_rules.py --result runs_m42/result.json || check_status=$?
python harness/score.py runs_m42/result.json
python - <<'PY'
import json
from pathlib import Path
expected = {
    'vocab_ce_partial_kernel': 'solution/vocab_ce.py',
    'vocab_ce_finish_kernel': 'solution/vocab_ce.py',
    'vocab_ce_backward_kernel': 'solution/vocab_ce.py',
    'vocab_entropy_partial_kernel': 'solution/entropy.py',
    'vocab_entropy_finish_kernel': 'solution/entropy.py',
}
for repeat in (1, 2):
    path = Path(f'runs_m42/solution_{repeat}/metrics.json')
    metrics = json.loads(path.read_text())
    for name, source in expected.items():
        assert metrics['triton_launches'].get(name, 0) > 0, (path, name, 'not launched')
        assert metrics['triton_files'].get(name) == source, (path, name, 'wrong source')
        assert name in metrics['own_kernel_ms'], (path, name, 'not timed')
    eligible_ms = sum(metrics['own_kernel_ms'][name] for name in expected)
    share = eligible_ms / (1000 * metrics['timed_loop_s'])
    print(path, f'eligible={eligible_ms:.3f} ms', f'share={100 * share:.3f}%')
PY
exit "$check_status"
```

The expected solution-suite count is **59 tests with no CUDA skips**. The checker is the authority for the minimum share across runs and all ten admission gates; the printout above is only an audit of the new launches. Keep failed outputs. If the new kernels launch but check 4 still fails, profile before choosing another operation:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/repo
test -f runs_m42/result.json
test ! -e runs_profile_m42
python harness/run_task.py --only solution --profile --out runs_profile_m42
```

That creates a separate full solution run and a 16-active-step diagnostic profile at `runs_profile_m42/solution_profile/profile.json`; do not treat it as another paired benchmark. Export available evidence with this Python notebook cell:

```python
from pathlib import Path
import os, subprocess, zipfile
from IPython.display import FileLink, display

repo = Path('/kaggle/working/repo')
roots = [repo / 'runs_m42', repo / 'runs_profile_m42']
assert roots[0].is_dir(), 'No paired run directory'
archive = Path('/kaggle/working/soup-m42-evidence.zip')
names = {'result.json', 'checks.json', 'metrics.json', 'profile.json', 'task.yaml'}
paths = [p for root in roots if root.is_dir() for p in root.rglob('*')
         if p.is_file() and (p.name in names or p.suffix == '.log')]
paths += list((repo / 'solution').glob('*.py'))
checkout = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True)
checkout += subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True)
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(paths):
        bundle.write(path, path.relative_to(repo))
    bundle.writestr('current_checkout.txt', checkout)
os.chdir('/kaggle/working')
display(FileLink(archive.name))
```

Download the ZIP or Kaggle output files. Choose a fresh ZIP filename if it exists. No model weights or caches are included.

## Milestone 2: exact Kaggle profiling and export

Use the existing successful T4 notebook session, with dependencies and public data already prepared. Append these cells after the baseline cells and run them individually. Do not rerun the clone cell: it deletes `/kaggle/working/repo`. No package installation or source/config edits are needed. Keep both GPUs otherwise idle; the harness selects GPU 0.

### 1. Run the official profiler

```bash
%%bash
set -euo pipefail
cd /kaggle/working/repo
test -f runs/result.json
test -f runs/checks.json
test ! -e runs_profile_m2
python harness/run_task.py --only solution --profile --out runs_profile_m2
```

This runs one full starter training/evaluation, followed by a separate profiled starter training run. The official profiler waits 6 steps, warms up for 2, records 16 active steps (steps 9–24), and stops the diagnostic run after step 24. This early stop is built into the official profiler; no training setting is edited. Its wall time/loss are not full-run benchmark results.

The output is `runs_profile_m2/solution_profile/profile.json`, with `active_steps`, `total_gpu_us`, and sorted `kernels` entries containing `name`, `time_us`, and `count`. The table sums CUDA event durations; it does not export CPU attribution, tensor shapes, call stacks, or a Chrome trace. Generic reduction/cast kernel names alone cannot establish RMSNorm's cost.

`--only solution` intentionally omits a new base run. Do not use this result for admission checks 1 and 5–8; keep the existing paired `runs/result.json` and `runs/checks.json`. To obtain a new full paired benchmark plus profile, omit `--only solution` in the command above. `--only base --profile` does **not** produce a base profile: `run_task.py` adds profiling only after a successful solution run.

If the profiling directory already exists, use a fresh directory name consistently in both cells. If the baseline files are absent, recover them from the saved notebook output or rerun the normal baseline first; do not label a new run with the earlier numbers.

### 2. Print the hotspot table and export the evidence

Run this as a Python notebook cell. It requires a successful profile and preserves full kernel names in a CSV inside the ZIP.

```python
from pathlib import Path
import csv, io, json, os, subprocess, zipfile
from IPython.display import FileLink, display

repo = Path('/kaggle/working/repo')
run = repo / 'runs_profile_m2'
result = json.loads((run / 'result.json').read_text())
assert result['profile_run']['exit_code'] == 0, 'Inspect solution_profile/train.log'
profile = json.loads((run / 'solution_profile/profile.json').read_text())
assert profile['active_steps'] == 16 and profile['total_gpu_us'] > 0
rows = sorted(profile['kernels'], key=lambda r: r['time_us'], reverse=True)
table = io.StringIO()
writer = csv.writer(table)
writer.writerow(['name', 'time_us', 'count', 'gpu_pct', 'ms_per_active_step'])
print(' GPU %   ms/step   launches  kernel')
for i, row in enumerate(rows):
    pct = 100 * row['time_us'] / profile['total_gpu_us']
    ms = row['time_us'] / (1000 * profile['active_steps'])
    writer.writerow([row['name'], row['time_us'], row['count'], pct, ms])
    if i < 30:
        print(f"{pct:6.2f} {ms:9.3f} {row['count']:10d}  {row['name']}")

paths = [repo / 'runs/result.json', repo / 'runs/checks.json', run / 'result.json']
paths += [run / side / name for side in ['solution_1', 'solution_profile']
          for name in ['metrics.json', 'train.log', 'task.yaml']]
paths += [run / 'solution_profile/profile.json']
assert all(p.is_file() for p in paths), 'Required evidence file is missing'
checkout = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True)
checkout += subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True)
archive = Path('/kaggle/working/soup-m2-profile.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    for path in paths:
        bundle.write(path, path.relative_to(repo))
    bundle.writestr('profile_kernels.csv', table.getvalue())
    bundle.writestr('current_checkout.txt', checkout)
os.chdir('/kaggle/working')
display(FileLink(archive.name))
```

Download `soup-m2-profile.zip` through the displayed link or Kaggle's output files. It contains the original paired result/checks, the new run's environment versions, diagnostic/full-run metrics and logs, generated configs, raw profile, CSV, and current checkout identity. It excludes model weights and compiler caches. The current checkout identity alone does not establish the earlier baseline's commit. The export uses exclusive creation; choose a new ZIP name if one already exists.

## Reading the 5% gate

`harness/check_rules.py:c4` computes, for each successful solution run:

```text
share = sum(eligible own_kernel_ms) / (1000 * metrics.timed_loop_s)
```

Eligibility requires original kernels under `solution/` that launched in every solution run. Both forward and backward launches can count; the starter example and generated Inductor kernels cannot. The denominator is the sum of synchronized step wall times from step 3 onward, not 83.37 s and not `profile.total_gpu_us`. The minimum share across runs must be at least 0.05. Full-run metrics are in `runs_profile_m2/solution_1/metrics.json`; profile-run metrics are diagnostic only.

A source operation's old GPU percentage does not predict the fused kernel's qualifying share: fusion reduces its runtime, and CPU/launch overhead also affects the denominator. Cover enough existing, required work using efficient original kernels; never duplicate work, add busy loops, disable optimization, or slow a kernel to reach the threshold. Measure the final share in complete paired runs and retain margin above 5% for different batches and jury rows.
