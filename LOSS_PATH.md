# Milestone 3: loss trace and implementation contract

## Sources inspected

Inspected the exact published wheels, verifying their SHA-256 against PyPI metadata before extraction. No third-party Triton kernel code was copied. The implementation uses the standard stable cross-entropy formula; these sources define integration semantics.

| Package | Published wheel SHA-256 |
|---|---|
| [soup-cli 0.77.0](https://pypi.org/project/soup-cli/0.77.0/) | `f1346e79cecb451211b714df06759b8b7dc9a6fd2fc086883297c35d1fe60ebb` |
| [trl 0.29.1](https://pypi.org/project/trl/0.29.1/) | `f9490cd1af93f3dce1cfdfe0fa1de108dbc46a2f6c2f90485cc2b4021c43eef3` |
| [transformers 5.16.1](https://pypi.org/project/transformers/5.16.1/) | `2f2d5b98a5ad3718713653734298fa620754ed683702a635ebb587df3ed29c7e` |

Locations below are package-relative, with line numbers from these wheels.

## Exact call path

1. **Labels:** Soup `data/sft_format.py:34` selects the assistant-only formatter for `train_on_responses_only`; `data/loss_mask.py:273` constructs masked labels. `trainer/sft.py:150` maps and checks causal targets, and `:175` derives `assistant_masks` from labels. TRL `trainer/sft_trainer.py:165–227` retains existing labels, pads them with `-100`, and applies completion/assistant masks. None of these functions is patched.
2. **Trainer:** Soup `trainer/sft.py:666` / `:1220` constructs `make_rewind_trainer_class(SFTTrainer)`. Its `trainer/rewind_hf.py:307` captures labels and calls TRL `SFTTrainer.compute_loss(..., return_outputs=True)`.
3. **Training CE:** TRL `trainer/sft_trainer.py:1234` delegates to Transformers `trainer.py:1974`, which calls the model and takes its loss. `models/qwen2/modeling_qwen2.py:463–469` computes full logits and invokes `self.loss_function`. `loss/loss_utils.py:49` (`ForCausalLMLoss`) casts logits to fp32, right-pads labels with `-100`, shifts by one, flattens, moves targets to the logits' device, then calls `fixed_cross_entropy` at `:32`. That helper uses a valid-target mean when `num_items_in_batch is None`, otherwise a sum divided by the supplied count. Transformers `trainer.py:2150` determines that count; the solution must use it as supplied, not substitute another count.
   TRL `SFTConfig.loss_type="nll"` is the default (`sft_config.py:236`); `SFTTrainer.__init__` leaves `compute_loss_func` unset for this value (`sft_trainer.py:915`). PEFT `PeftModelForCausalLM.forward` passes `output_attentions` and `output_hidden_states` to Qwen2, which forwards them through `ForCausalLMLoss` to `fixed_cross_entropy`. The latter accepts `**kwargs` and ignores them.
4. **Entropy:** TRL `trainer/sft_trainer.py:1238` runs under `no_grad`. Its imported `entropy_from_logits` (`trainer/utils.py:572`) flattens logits, processes chunks of 128 rows with `F.log_softmax`, and calculates `-(exp(logps) * logps).sum(-1)`. The trainer weights by **attention_mask**, including prompt tokens, and divides by its sum (or takes a mean for position IDs), then gathers the metric. This path is distinct from the supervised-loss mask. Milestone 4.2 replaces only the imported helper for eligible inputs; the trainer's mask/reduction remains native.
5. **Accuracy/tokens:** TRL `trainer/sft_trainer.py:1259–1320` accumulates attention-mask token totals. Accuracy uses `logits[..., :-1, :].contiguous().argmax(-1)` and `labels[..., 1:] != -100`; it gathers correct/valid counts and returns zero for no valid tokens. This path remains native.
6. **Rewind:** After TRL returns, Soup `trainer/rewind_hf.py:183` computes per-row CE under `no_grad`, using detached `logits[i, :-1, :]` in the logits' dtype and `labels[i, 1:]`. Only the token losses are converted to fp32; row sums are divided by `max(valid_count, 1)`. A fully masked row gives loss/count `(0, 0)`. `:350–366` records these values using the existing sampler FIFO and step/micro bookkeeping. This path remains native.

The reported 6.49% log-softmax category can include training CE, entropy, and rewind. The supplied aggregate does not split them. It cannot be assigned entirely to the new training-loss kernel.

With native fp16 AMP, the pinned Accelerate 1.14.0 `accelerator.py:1818` wraps model forward in autocast and `convert_outputs_to_fp32` (`utils/operations.py:889–942`). Thus low-precision model outputs are promoted before TRL/rewind consume them; the training loss already casts internally. The solution changes neither wrapper. Tests cover consumers with both fp16 and fp32 logits rather than assuming the raw LM-head dtype is the logged-output dtype.

## Chosen operation

`solution.apply()` keeps the starter MLP patch and installs an idempotent wrapper around **Transformers 5.16.1 `loss_utils.fixed_cross_entropy`**. The fast path requires contiguous CUDA fp32 `[N, 151936]`, contiguous int64 `[N]` labels on the same device, nonempty rows, and `ignore_index=-100`. Only the two ignored model-output keywords named above are allowed; other extra keywords call the original helper. Other unsupported inputs also fall back. Other Transformers versions leave the helper untouched. No global PyTorch CE/log-softmax patch, trainer replacement, or new trainable parameter is introduced.

`solution/vocab_ce.py` contains three original kernels:

- `vocab_ce_partial_kernel`: per-token vocabulary tiles of 2,048 values, with tile maximum and shifted exponential sum. Ignored rows skip vocabulary arithmetic.
- `vocab_ce_finish_kernel`: combine tile statistics, read the target logit, write token loss plus the row maximum and log-sum. Bounds assertions prevent invalid target reads.
- `vocab_ce_backward_kernel`: write `(softmax(x) - one_hot(target)) * upstream` tile by tile; ignored rows write zero. It neither mutates logits nor saves a full probability/log-softmax matrix.

For each valid row, `m = max(x)`, `s = sum(exp(x-m))`, and `loss = (m-x[target]) + log(s)`. Backward uses `exp((x-m)-log(s))`; storing `m` and `log(s)` separately avoids cancellation from a large common logit offset. Token/batch reductions and the supplied denominator are handled in PyTorch. Autograd preserves upstream casts and loss scaling. The all-ignored mean remains NaN with zero input gradients, matching native behavior for finite logits. Higher-order derivatives are intentionally unsupported (`once_differentiable`); this task uses first-order training.

## Verification and limits

- Local isolated `/tmp` environment: Python 3.12.14, PyTorch 2.9.1+cpu, Triton 3.5.1, and the pinned Soup/TRL/Transformers/PEFT dependencies. No packages were installed in a measured training run.
- `TRITON_INTERPRET=1 python -m pytest solution -q`: **39 passed, 4 skipped**. The four skipped tests are the unchanged CUDA-only starter SwiGLU tests. New kernels execute in Triton's CPU interpreter; CUDA dispatch is mocked only in interpreter integration tests, never in production code.
- Tests compare per-token loss and gradients against native PyTorch for ragged tiles, the full 151,936 vocabulary, large offsets, peaked/uniform logits, mixed/all-ignored masks, scalar/tensor denominators, signed non-contiguous upstream gradients, and loss scaling. Tests cover fallback/idempotence/version guards and fp16→fp32 causal-loss casting/shifting. The real Soup/TRL compute-loss methods preserve all tested metrics and rewind records exactly; a tiny real Qwen2 model verifies the gradient reaches `q_proj`.
- All three kernels compile offline for CUDA SM75 (T4); shared-memory metadata is 16 / 16 / 0 bytes for partial / finish / backward. Compilation is not device execution.
- `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q`: **79 passed** locally. This checks CPU/fake-model compatibility, not the CUDA fast path.
- The harness source scanner identifies all three kernels as original candidates, with no example lookalikes, prohibited imports, or parse errors. This does not establish that they launch or take 5% in training.
- T4 execution, numeric agreement under its compiler, new end-to-end time/memory/loss, and measured eligible share are **unverified**. Numerical testing covers finite logits; behavior of ignored rows containing NaN/Inf is not promised identical to PyTorch's internal log-softmax backward. The new CE is only a subset of the reported log-softmax work, and optimizing it shrinks its event time. Do not claim admission until the full Kaggle checks pass.

The paragraph above describes the Milestone 3 state. The user later verified CE launches at `02d49c5` and measured 441.138 ms / 49.38 s = 0.893%; the 5% gate still fails. Milestone 4.2's entropy addition has not yet been run on T4. See `BENCHMARKS.md` for the current validation cell.

## Milestone 4.1: why the kernels did not launch

The user-reported T4 runs at `3e15f49` completed with equal eval loss, but `own_kernel_ms`, `triton_launches`, and `triton_files` contained only the excluded starter SwiGLU kernels. A local Qwen2 + PEFT call with the pinned versions reached `fixed_cross_entropy` with fp32 contiguous `[N,151936]` logits and int64 contiguous labels, plus `output_attentions` and `output_hidden_states` in `kwargs`. The old `if not kwargs` guard therefore selected the original PyTorch helper. This explains why the CUDA unit tests passed while full training had no CE launches: the earlier tests used a bare Qwen2 model or synthetic model, neither of which passed the PEFT keywords.

The correction allows exactly these two ignored keyword names through the existing eligibility checks. It changes no kernel math, model output, task setting, metric, or rewind path. A new test runs the actual Soup rewind and TRL `compute_loss` methods with a PEFT-wrapped Qwen2 model at `loss_type="nll"`. It failed with zero dispatches before the correction and passes afterward, comparing native loss, LoRA gradient, entropy, accuracy, and rewind records. CPU interpreter coverage does not establish T4 execution or admission.

After the correction, the local solution suite reports **40 passed, 4 skipped** in Triton's CPU interpreter, and the 79 official regressions pass with the solution applied. The skipped tests are unchanged CUDA-only starter tests. The Kaggle cells below passed local Bash/Python syntax checks; they have not been run on the corrected commit.

## Exact Kaggle validation for Milestone 4.1

Push the solution files to the team's platform repository, pin that commit in the provided notebook, and run its dependency/clone/data setup cells on T4. Do not validate an older checkout. Then run this new Bash cell; it leaves prior `runs/` artifacts intact:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/repo
unset TRITON_INTERPRET
test ! -e runs_m41
git rev-parse HEAD
python - <<'PY'
from importlib.metadata import version
for package, expected in [('soup-cli', '0.77.0'), ('trl', '0.29.1'), ('transformers', '5.16.1')]:
    assert version(package) == expected, (package, version(package))
import solution
solution.apply()
from transformers.loss import loss_utils
assert getattr(loss_utils.fixed_cross_entropy, '_soup_vocab_ce', False)
PY
CUDA_VISIBLE_DEVICES=0 python -m pytest solution -q
python harness/run_task.py --repeats 2 --out runs_m41
check_status=0
python harness/check_rules.py --result runs_m41/result.json || check_status=$?
python harness/score.py runs_m41/result.json
python - <<'PY'
import json
from pathlib import Path
expected = {'vocab_ce_partial_kernel', 'vocab_ce_finish_kernel', 'vocab_ce_backward_kernel'}
for repeat in (1, 2):
    path = Path(f'runs_m41/solution_{repeat}/metrics.json')
    metrics = json.loads(path.read_text())
    for field in ('triton_launches', 'triton_files', 'own_kernel_ms'):
        missing = expected - metrics[field].keys()
        assert not missing, (str(path), field, sorted(missing))
    assert all(metrics['triton_launches'][name] > 0 for name in expected)
    assert all(metrics['triton_files'][name] == 'solution/vocab_ce.py' for name in expected)
    print(path, {name: metrics['own_kernel_ms'][name] for name in sorted(expected)})
PY
exit "$check_status"
```

All **44 solution tests** should pass with CUDA available and no skips; the CPU fallback/version tests intentionally remain CPU checks. The official checker runs the 79 regressions again. Read `runs_m41/checks.json`, `runs_m41/result.json`, and each `runs_m41/solution_*/metrics.json`: verify all three `vocab_ce_*` launch entries and `own_kernel_ms`; check 4 excludes the example and uses the minimum eligible share across runs. Download those JSON files and training logs from Kaggle output. If any test or admission gate fails, retain the logs and do not claim completion. No new speed, memory, or share figure has yet been recorded for the corrected integration.

Export evidence with this Python notebook cell, including after a failed validation. It preserves available JSON/logs/configs and the tested solution sources, without model weights. Missing results remain missing rather than being synthesized.

```python
from pathlib import Path
import os, subprocess, zipfile
from IPython.display import FileLink, display

repo = Path('/kaggle/working/repo')
run = repo / 'runs_m41'
assert run.is_dir(), 'No run directory; retain the notebook test output instead'
archive = Path('/kaggle/working/soup-m41-evidence.zip')
paths = [p for p in run.rglob('*') if p.is_file()
         and (p.name in {'result.json', 'checks.json', 'metrics.json', 'task.yaml'}
              or p.suffix == '.log')]
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

Download the ZIP and save the notebook output containing the CUDA test summary. Use a fresh ZIP filename if it already exists.
