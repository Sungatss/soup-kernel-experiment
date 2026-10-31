"""Numerical reference tests for the detached TRL entropy path."""

import os

import pytest
import torch
import torch.nn.functional as F

from solution import entropy


@pytest.fixture
def kernel_device():
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA or explicit TRITON_INTERPRET=1")
    return torch.device("cuda")


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 17), (2, 2, 2053), (1, 3, 151936)])
@pytest.mark.parametrize("offset,scale", [(0.0, 1.0), (10000.0, 1.0), (-10000.0, 80.0)])
def test_token_entropy_matches_log_softmax(kernel_device, shape, offset, scale):
    torch.manual_seed(61)
    logits = torch.randn(shape, device=kernel_device, dtype=torch.float32) * scale + offset
    with torch.no_grad():
        actual = entropy.token_entropy(logits)
        logps = F.log_softmax(logits, dim=-1)
        expected = -(torch.exp(logps) * logps).sum(-1)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)


def test_entropy_uniform_and_no_gradient(kernel_device):
    logits = torch.zeros((2, 151936), device=kernel_device, requires_grad=True)
    with torch.no_grad():
        actual = entropy.token_entropy(logits)
    torch.testing.assert_close(actual, torch.full_like(actual, torch.log(torch.tensor(151936.)).item()),
                               rtol=0, atol=1e-5)
    assert not actual.requires_grad
    with pytest.raises(ValueError, match="no_grad"):
        entropy.token_entropy(logits)


def test_trl_entropy_hook_and_cpu_fallback(kernel_device, monkeypatch):
    from trl.trainer import sft_trainer

    original = sft_trainer.entropy_from_logits
    monkeypatch.setattr(sft_trainer, "entropy_from_logits", original)
    assert entropy.patch_trl_entropy()
    patched = sft_trainer.entropy_from_logits
    assert entropy.patch_trl_entropy() and sft_trainer.entropy_from_logits is patched
    logits = torch.randn((1, 2, 151936), device=kernel_device)
    with torch.no_grad():
        expected = original(logits)
        if kernel_device.type == "cpu":
            monkeypatch.setattr(entropy, "_eligible", lambda *args: True)
        actual = patched(logits)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    if kernel_device.type == "cpu":
        monkeypatch.setattr(entropy, "_eligible", lambda *args: False)
        with torch.no_grad():
            torch.testing.assert_close(patched(logits), expected, rtol=0, atol=0)
    monkeypatch.setattr(entropy, "token_entropy", lambda *args: pytest.fail("unexpected fast path"))
    with torch.no_grad():
        short = logits[..., :17].contiguous()
        torch.testing.assert_close(patched(short), original(short), rtol=0, atol=0)
        torch.testing.assert_close(patched(logits, chunk_size=7), original(logits, chunk_size=7),
                                   rtol=0, atol=0)
        low_precision = short.half()
        torch.testing.assert_close(patched(low_precision), original(low_precision), rtol=0, atol=0)
    differentiable = (logits if kernel_device.type == "cuda" else short).detach().clone().requires_grad_()
    reference = differentiable.detach().clone().requires_grad_()
    patched(differentiable).sum().backward()
    original(reference).sum().backward()
    torch.testing.assert_close(differentiable.grad, reference.grad, rtol=0, atol=0)


def test_trl_entropy_version_guard(monkeypatch):
    from trl.trainer import sft_trainer

    original = sft_trainer.entropy_from_logits
    monkeypatch.setattr(entropy, "version", lambda package: "different")
    assert not entropy.patch_trl_entropy()
    assert sft_trainer.entropy_from_logits is original
