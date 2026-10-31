"""NF4 format, native bitsandbytes call paths, and upstream gradients."""

import os
from types import SimpleNamespace

import bitsandbytes as bnb
import pytest
import torch

from solution import nf4


@pytest.fixture
def kernel_device():
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA or explicit TRITON_INTERPRET=1")
    return torch.device("cuda")


def quantized(shape, device):
    torch.manual_seed(71)
    weight = torch.randn(shape, dtype=torch.float16, device=device) * 0.05
    return bnb.functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)


@pytest.mark.parametrize("shape", [(1, 64), (3, 64), (257, 64), (256, 1536), (1536, 1536)])
def test_nested_dequant_matches_bnb(kernel_device, shape):
    packed, state = quantized(shape, kernel_device)
    expected = bnb.functional.dequantize_4bit(packed, state)
    actual = nf4.dequantize_nested(packed, state.absmax, state.state2.absmax,
                                 state.state2.code, state.offset, shape)
    assert actual.dtype == torch.float16 and actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_all_nf4_and_nested_scale_codes(kernel_device):
    packed, state = quantized((257, 64), kernel_device)
    # All 16 nibble values; every scale-code index and a second scale group.
    packed.copy_((torch.arange(packed.numel(), device=kernel_device) % 256).byte().view_as(packed))
    state.absmax.copy_((torch.arange(257, device=kernel_device) % 256).byte())
    state.offset.fill_(0.75)
    expected = bnb.functional.dequantize_4bit(packed, state)
    actual = nf4.dequantize_nested(packed, state.absmax, state.state2.absmax,
                                 state.state2.code, state.offset, state.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def restore_hooks(monkeypatch, device):
    monkeypatch.setattr(bnb.functional, "dequantize_4bit", bnb.functional.dequantize_4bit)
    if device.type == "cuda":
        from bitsandbytes.backends.cuda import ops
        monkeypatch.setattr(ops, "_dequant_linear_fallback", ops._dequant_linear_fallback)
    else:
        monkeypatch.setattr(nf4, "_on_cuda", lambda tensor: True)


@pytest.mark.parametrize("rows", [5, 2048])
def test_real_bnb_forward_backward(kernel_device, monkeypatch, rows):
    """Actual MatMul4Bit; 2048 rows reaches CUDA's mandatory fallback branch."""
    packed, state = quantized((256, 1536), kernel_device)
    x = torch.randn((rows, 1536), device=kernel_device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(256, device=kernel_device, dtype=torch.float16, requires_grad=True)
    upstream = torch.randn((rows, 256), device=kernel_device, dtype=torch.float16) * 0.125
    expected = bnb.matmul_4bit(x, packed, quant_state=state, bias=bias)
    expected.backward(upstream)
    dx, db = x.grad.clone(), bias.grad.clone()
    x.grad = bias.grad = None
    restore_hooks(monkeypatch, kernel_device)
    assert nf4.patch_bitsandbytes_nf4()
    calls = []
    original = nf4.dequantize_nested

    def observed(*args):
        calls.append(tuple(args[-1]))
        return original(*args)

    monkeypatch.setattr(nf4, "dequantize_nested", observed)
    actual = bnb.matmul_4bit(x, packed, quant_state=state, bias=bias)
    forward_calls = len(calls)
    actual.backward(upstream)
    assert len(calls) > forward_calls, "native backward did not reach fused dequant"
    if kernel_device.type == "cuda" and rows == 2048:
        assert forward_calls == 1, "real CUDA fallback did not reach fused dequant"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, dx, rtol=0, atol=0)
    torch.testing.assert_close(bias.grad, db, rtol=0, atol=0)


def test_forward_fallback_wrapper_and_transpose(kernel_device, monkeypatch):
    packed, state = quantized((256, 1536), kernel_device)
    x = torch.randn((2, 3, 1536), device=kernel_device, dtype=torch.float16)
    expected_weight = bnb.functional.dequantize_4bit(packed, state)
    expected = torch.nn.functional.linear(x, expected_weight)
    restore_hooks(monkeypatch, kernel_device)
    ops = SimpleNamespace(_dequant_linear_fallback=lambda *a, **kw: "fallback")
    nf4._patch_cuda_fallback(ops)
    patched = ops._dequant_linear_fallback
    nf4._patch_cuda_fallback(ops)
    assert ops._dequant_linear_fallback is patched
    args = (x, packed, state.shape, state.state2.absmax, 64, "nf4")
    kwargs = dict(absmax_8bit=state.absmax, absmax_code=state.state2.code, absmax_offset=state.offset)
    torch.testing.assert_close(patched(*args, **kwargs), expected, rtol=0, atol=0)
    assert patched(*args[:-1], "fp4", **kwargs) == "fallback"
    assert nf4.patch_bitsandbytes_nf4()
    patched_dequant = bnb.functional.dequantize_4bit
    assert nf4.patch_bitsandbytes_nf4() and bnb.functional.dequantize_4bit is patched_dequant
    torch.testing.assert_close(patched_dequant(packed.t(), state), expected_weight.t(), rtol=0, atol=0)


@pytest.mark.parametrize("case", ["shape", "dtype", "block", "nested", "scale_block", "out", "cpu"])
def test_unsupported_dequant_uses_original(kernel_device, monkeypatch, case):
    packed, state = quantized((256, 1536), kernel_device)
    restore_hooks(monkeypatch, kernel_device)
    calls = []
    monkeypatch.setattr(bnb.functional, "dequantize_4bit", lambda *a, **kw: calls.append(True))
    assert nf4.patch_bitsandbytes_nf4()
    if case == "shape":
        state.shape = (1536, 256)
    elif case == "dtype":
        state.dtype = torch.bfloat16
    elif case == "block":
        state.blocksize = 128
    elif case == "nested":
        state.nested = False
    elif case == "scale_block":
        state.state2.blocksize = 128
    elif case == "cpu":
        monkeypatch.setattr(nf4, "_on_cuda", lambda tensor: False)
    out = torch.empty(state.shape, device=kernel_device, dtype=torch.float16) if case == "out" else None
    bnb.functional.dequantize_4bit(packed, state, out=out)
    assert calls == [True]


def test_version_guard(monkeypatch):
    original = bnb.functional.dequantize_4bit
    monkeypatch.setattr(nf4, "version", lambda package: "different")
    assert not nf4.patch_bitsandbytes_nf4()
    assert bnb.functional.dequantize_4bit is original
