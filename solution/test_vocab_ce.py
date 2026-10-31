"""Numerical tests: CUDA normally; TRITON_INTERPRET=1 explicitly runs on CPU.

Interpreter results test the kernel algorithms, not CUDA execution or T4 speed.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from solution import vocab_ce


@pytest.fixture
def kernel_device():
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA or explicit TRITON_INTERPRET=1")
    return torch.device("cuda")


def compare_gradients(actual, expected, x, ref, grad=None):
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5, equal_nan=True)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, ref.grad, rtol=5e-5, atol=2e-6, equal_nan=True)


@pytest.mark.parametrize("shape", [(1, 1), (5, 17), (4, 2048), (3, 2053), (3, 151936)])
@pytest.mark.parametrize("offset,scale", [(0.0, 1.0), (10000.0, 1.0), (-10000.0, 80.0)])
def test_token_losses_and_arbitrary_gradients(kernel_device, shape, offset, scale):
    torch.manual_seed(19)
    x = (torch.randn(shape, device=kernel_device) * scale + offset).requires_grad_()
    ref = x.detach().clone().requires_grad_()
    labels = torch.arange(shape[0], device=kernel_device, dtype=torch.int64) % shape[1]
    labels[-1] = shape[1] - 1
    if shape[0] > 1:
        labels[0] = -100
    # Non-contiguous upstream vector, including negative gradients.
    grad = torch.randn(shape[0] * 2, device=kernel_device)[::2] * 3
    actual = vocab_ce.token_cross_entropy(x, labels)
    expected = F.cross_entropy(ref, labels, reduction="none")
    compare_gradients(actual, expected, x, ref, grad)
    if shape[0] > 1:
        assert actual[0].item() == 0 and torch.count_nonzero(x.grad[0]).item() == 0


@pytest.mark.parametrize("divisor", [None, 11, "tensor"])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_reductions_and_scaled_gradients(kernel_device, divisor, all_ignored):
    torch.manual_seed(22)
    x = torch.randn((4, 2053), device=kernel_device, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    labels = torch.tensor([-100, 0, 2052, 7], device=kernel_device)
    if all_ignored:
        labels.fill_(-100)
    if divisor == "tensor":
        divisor = torch.tensor(11, device=kernel_device)
    actual = vocab_ce.fixed_vocab_cross_entropy(x, labels, divisor)
    expected = F.cross_entropy(ref, labels, reduction="mean" if divisor is None else "sum")
    if divisor is not None:
        expected = expected / divisor
    compare_gradients(actual * 128, expected * 128, x, ref)
    if all_ignored:
        assert torch.count_nonzero(x.grad).item() == 0


def test_uniform_logits_and_probability_gradient(kernel_device):
    x = torch.zeros((2, 151936), device=kernel_device, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    labels = torch.tensor([0, 151935], device=kernel_device)
    compare_gradients(vocab_ce.fixed_vocab_cross_entropy(x, labels),
                      F.cross_entropy(ref, labels), x, ref)
    torch.testing.assert_close(x.grad.sum(-1), torch.zeros(2, device=kernel_device), atol=2e-6, rtol=0)


def test_custom_ignore_index(kernel_device):
    x = torch.randn((3, 17), device=kernel_device, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    labels = torch.tensor([-7, 0, 16], device=kernel_device)
    compare_gradients(vocab_ce.fixed_vocab_cross_entropy(x, labels, ignore_index=-7),
                      F.cross_entropy(ref, labels, ignore_index=-7), x, ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("divisor", [None, 13])
def test_causal_loss_hook_preserves_cast_shift_and_divisor(kernel_device, monkeypatch, dtype, divisor):
    from transformers.loss import loss_utils

    original = loss_utils.fixed_cross_entropy
    monkeypatch.setattr(loss_utils, "fixed_cross_entropy", original)
    torch.manual_seed(37)
    x = torch.randn((2, 3, vocab_ce.VOCAB_SIZE), device=kernel_device, dtype=dtype).requires_grad_()
    ref = x.detach().clone().requires_grad_()
    labels = torch.tensor([[-100, 7, 151935], [-100, -100, 3]], device=kernel_device)
    expected = loss_utils.ForCausalLMLoss(ref, labels, vocab_ce.VOCAB_SIZE, divisor)
    assert vocab_ce.patch_transformers_cross_entropy()
    calls = []
    compute = vocab_ce.fixed_vocab_cross_entropy

    def observed(*args, **kwargs):
        calls.append((args[0].dtype, args[1].detach().clone()))
        return compute(*args, **kwargs)

    monkeypatch.setattr(vocab_ce, "fixed_vocab_cross_entropy", observed)
    if kernel_device.type == "cpu":
        # Exercise the real HF call chain with interpreted kernels; production
        # dispatch retains its CUDA guard and has no test/environment switch.
        monkeypatch.setattr(vocab_ce, "_eligible", lambda *args: True)
    actual = loss_utils.ForCausalLMLoss(x, labels, vocab_ce.VOCAB_SIZE, divisor)
    assert len(calls) == 1 and calls[0][0] == torch.float32
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:].reshape(-1)
    torch.testing.assert_close(calls[0][1], shifted)
    assert actual.dtype == torch.float32
    compare_gradients(actual, expected, x, ref)


def test_patch_idempotence_and_cpu_fallback(monkeypatch):
    from transformers.loss import loss_utils

    original = loss_utils.fixed_cross_entropy
    monkeypatch.setattr(loss_utils, "fixed_cross_entropy", original)
    assert vocab_ce.patch_transformers_cross_entropy()
    patched = loss_utils.fixed_cross_entropy
    assert vocab_ce.patch_transformers_cross_entropy()
    assert loss_utils.fixed_cross_entropy is patched
    monkeypatch.setattr(vocab_ce, "fixed_vocab_cross_entropy", lambda *a, **kw: pytest.fail("CPU dispatched"))
    x = torch.randn((3, vocab_ce.VOCAB_SIZE), requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    labels = torch.tensor([-100, 0, vocab_ce.VOCAB_SIZE - 1])
    compare_gradients(patched(x, labels), original(ref, labels), x, ref)


def test_version_guard(monkeypatch):
    from transformers.loss import loss_utils

    original = loss_utils.fixed_cross_entropy
    monkeypatch.setattr(vocab_ce, "version", lambda package: "different")
    assert not vocab_ce.patch_transformers_cross_entropy()
    assert loss_utils.fixed_cross_entropy is original


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_real_rewind_trl_call_chain(kernel_device, monkeypatch, dtype):
    """Exercise the installed trainer methods with a tiny synthetic-logit model.

    No tokenizer, downloaded model, optimizer, or data setting is substituted
    in a benchmark: this is a unit test of the actual loss/metric consumers.
    """
    from collections import defaultdict
    from types import SimpleNamespace
    from soup_cli.trainer.rewind_hf import attach_rewind_state, make_rewind_trainer_class
    from transformers.loss import loss_utils
    from transformers.modeling_outputs import CausalLMOutput
    from trl import SFTTrainer

    original = loss_utils.fixed_cross_entropy
    monkeypatch.setattr(loss_utils, "fixed_cross_entropy", original)
    torch.manual_seed(43)
    values = torch.randn((2, 4, vocab_ce.VOCAB_SIZE), device=kernel_device, dtype=dtype)
    # One fully unsupervised row verifies rewind's zero-token behavior;
    # entropy still uses attention_mask, not this supervision mask.
    labels = torch.tensor([[-100, 5, 151935, -100], [-100, -100, -100, -100]], device=kernel_device)
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], device=kernel_device)

    class LogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(values.clone())

        def forward(self, input_ids, attention_mask, labels, num_items_in_batch=None, use_cache=False):
            loss = loss_utils.ForCausalLMLoss(self.logits, labels, vocab_ce.VOCAB_SIZE, num_items_in_batch)
            return CausalLMOutput(loss=loss, logits=self.logits)

    def run():
        model = LogitModel()
        trainer = object.__new__(make_rewind_trainer_class(SFTTrainer))
        trainer.model = model
        trainer.args = SimpleNamespace(use_liger_kernel=False, average_tokens_across_devices=False)
        trainer.accelerator = SimpleNamespace(parallelism_config=None, gather_for_metrics=lambda x: x)
        trainer.label_smoother = trainer.compute_loss_func = None
        trainer.model_accepts_loss_kwargs = True
        trainer.num_virtual_tokens = 0
        trainer.aux_loss_enabled = False
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer._total_train_tokens = 0
        trainer.state = SimpleNamespace(global_step=0)
        records = []
        state = attach_rewind_state(trainer, SimpleNamespace(record_batch=lambda **kw: records.append(kw)))
        state.on_index(12)
        state.on_index(25)
        inputs = {"input_ids": torch.zeros_like(labels), "attention_mask": attention, "labels": labels}
        loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True, num_items_in_batch=2)
        loss.backward()
        assert not state.failed and state.dropped == 0
        assert len(records) == 1 and records[0]["rows"] == [12, 25]
        assert records[0]["row_tokens"] == [2, 0] and records[0]["row_loss"][1] == 0
        torch.testing.assert_close(outputs.logits, values, rtol=0, atol=0)
        return loss.detach(), model.logits.grad, dict(trainer._metrics["train"]), records

    expected = run()
    assert vocab_ce.patch_transformers_cross_entropy()
    calls = []
    compute = vocab_ce.fixed_vocab_cross_entropy

    def observed(*args, **kwargs):
        calls.append(True)
        return compute(*args, **kwargs)

    monkeypatch.setattr(vocab_ce, "fixed_vocab_cross_entropy", observed)
    if kernel_device.type == "cpu":
        monkeypatch.setattr(vocab_ce, "_eligible", lambda *args: True)
    actual = run()
    assert len(calls) == 1  # Rewind/entropy retain their separate native paths.
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=5e-5, atol=2e-6)
    assert actual[2] == expected[2]  # Entropy, accuracy, and token totals unchanged.
    assert actual[3] == expected[3]  # Row identities, token counts, and losses unchanged.


def test_tiny_qwen_model_gradient_path(kernel_device, monkeypatch):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.loss import loss_utils

    original = loss_utils.fixed_cross_entropy
    monkeypatch.setattr(loss_utils, "fixed_cross_entropy", original)
    torch.manual_seed(47)
    config = Qwen2Config(vocab_size=vocab_ce.VOCAB_SIZE, hidden_size=16,
                         intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1,
                         tie_word_embeddings=True, attention_dropout=0.0)
    model = Qwen2ForCausalLM(config).to(kernel_device).train()
    ids = torch.tensor([[5, 8, 11]], device=kernel_device)
    labels = torch.tensor([[-100, 8, 11]], device=kernel_device)
    expected = model(input_ids=ids, labels=labels, use_cache=False, num_items_in_batch=2)
    expected.loss.backward()
    projection = model.model.layers[0].self_attn.q_proj.weight
    expected_grad = projection.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    assert vocab_ce.patch_transformers_cross_entropy()
    calls = []
    compute = vocab_ce.fixed_vocab_cross_entropy

    def observed(*args, **kwargs):
        calls.append(True)
        return compute(*args, **kwargs)

    monkeypatch.setattr(vocab_ce, "fixed_vocab_cross_entropy", observed)
    if kernel_device.type == "cpu":
        monkeypatch.setattr(vocab_ce, "_eligible", lambda *args: True)
    actual = model(input_ids=ids, labels=labels, use_cache=False, num_items_in_batch=2)
    actual.loss.backward()
    assert len(calls) == 1
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(projection.grad, expected_grad, rtol=5e-5, atol=2e-6)
    assert torch.count_nonzero(projection.grad).item() > 0


def test_peft_rewind_nll_reaches_vocab_ce(kernel_device, monkeypatch):
    """Real PEFT/Qwen forward through Soup/TRL's NLL and metric path."""
    from collections import defaultdict
    from types import SimpleNamespace
    from peft import LoraConfig, get_peft_model
    from soup_cli.trainer.rewind_hf import attach_rewind_state, make_rewind_trainer_class
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.loss import loss_utils
    from trl import SFTTrainer

    original = loss_utils.fixed_cross_entropy
    ignored_kwargs = []

    def native_ce(*args, **kwargs):
        ignored_kwargs.append(set(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(loss_utils, "fixed_cross_entropy", native_ce)
    torch.manual_seed(53)
    config = Qwen2Config(vocab_size=vocab_ce.VOCAB_SIZE, hidden_size=16,
                         intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1,
                         tie_word_embeddings=True, attention_dropout=0.0)
    model = get_peft_model(Qwen2ForCausalLM(config),
                           LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                                      target_modules=["q_proj"], task_type="CAUSAL_LM"))
    model.to(kernel_device).train()
    projection = model.base_model.model.model.layers[0].self_attn.q_proj.lora_B["default"].weight
    labels = torch.tensor([[-100, 8, 11]], device=kernel_device)
    inputs = {"input_ids": torch.tensor([[5, 8, 11]], device=kernel_device),
              "attention_mask": torch.ones((1, 3), device=kernel_device),
              "labels": labels}
    trainer = object.__new__(make_rewind_trainer_class(SFTTrainer))
    trainer.model = model
    trainer.args = SimpleNamespace(use_liger_kernel=False, average_tokens_across_devices=False,
                                   loss_type="nll")
    trainer.accelerator = SimpleNamespace(parallelism_config=None, gather_for_metrics=lambda x: x)
    trainer.label_smoother = trainer.compute_loss_func = None
    trainer.model_accepts_loss_kwargs = True
    trainer.num_virtual_tokens = 0
    trainer.aux_loss_enabled = False
    trainer.state = SimpleNamespace(global_step=0)

    def run():
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer._total_train_tokens = 0
        records = []
        state = attach_rewind_state(trainer, SimpleNamespace(record_batch=lambda **kw: records.append(kw)))
        state.on_index(17)
        loss, outputs = trainer.compute_loss(model, dict(inputs), return_outputs=True,
                                             num_items_in_batch=2)
        loss.backward()
        assert not state.failed and state.dropped == 0
        assert records[0]["rows"] == [17] and records[0]["row_tokens"] == [2]
        assert outputs.logits.shape == (1, 3, vocab_ce.VOCAB_SIZE)
        return loss.detach(), projection.grad.detach().clone(), dict(trainer._metrics["train"]), records

    expected = run()
    assert ignored_kwargs == [{"output_attentions", "output_hidden_states"}]
    model.zero_grad(set_to_none=True)
    assert vocab_ce.patch_transformers_cross_entropy()
    calls = []
    compute = vocab_ce.fixed_vocab_cross_entropy

    def observed(*args, **kwargs):
        calls.append(True)
        return compute(*args, **kwargs)

    monkeypatch.setattr(vocab_ce, "fixed_vocab_cross_entropy", observed)
    if kernel_device.type == "cpu":
        monkeypatch.setattr(vocab_ce, "_eligible", lambda *args: True)
    actual = run()
    assert len(calls) == 1
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=5e-5, atol=2e-6)
    assert torch.count_nonzero(actual[1]).item() > 0
    assert actual[2:] == expected[2:]


@pytest.mark.parametrize("case", ["strided", "target_strided", "dtype", "vocab", "empty", "ignore", "kwargs"])
def test_unsupported_inputs_fall_back(kernel_device, monkeypatch, case):
    from transformers.loss import loss_utils

    original = loss_utils.fixed_cross_entropy
    monkeypatch.setattr(loss_utils, "fixed_cross_entropy", original)
    assert vocab_ce.patch_transformers_cross_entropy()
    monkeypatch.setattr(vocab_ce, "fixed_vocab_cross_entropy", lambda *a, **kw: pytest.fail("unsupported dispatch"))
    vocab = 31 if case == "vocab" else vocab_ce.VOCAB_SIZE
    rows = 0 if case == "empty" else 2
    x = torch.randn((rows, vocab * (2 if case == "strided" else 1)), device=kernel_device,
                    dtype=torch.float64 if case == "dtype" else torch.float32)
    if case == "strided":
        x = x[:, ::2]
    labels = torch.zeros(rows, dtype=torch.int64, device=kernel_device)
    if case == "target_strided":
        labels = torch.zeros(rows * 2, dtype=torch.int64, device=kernel_device)[::2]
    ignore = -7 if case == "ignore" else -100
    extra = {"unused_option": True} if case == "kwargs" else {}
    torch.testing.assert_close(loss_utils.fixed_cross_entropy(x, labels, ignore_index=ignore, **extra),
                               original(x, labels, ignore_index=ignore, **extra), equal_nan=True)
