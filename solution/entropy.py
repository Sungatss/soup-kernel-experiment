"""Tiled full-vocabulary entropy for TRL's detached SFT metric."""

from functools import wraps
from importlib.metadata import version

import torch
import triton
import triton.language as tl


VOCAB_SIZE = 151936
TILE = 2048


@triton.jit
def vocab_entropy_partial_kernel(X, PM, PS, PQ, V: tl.constexpr,
                                 PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    valid = col < V
    values = tl.load(X + row * V + col, valid, other=-float("inf"))
    maximum = tl.max(values, 0)
    shifted = tl.where(valid, values - maximum, 0.0)
    exp_shifted = tl.where(valid, tl.exp(shifted), 0.0)
    offset = row * PARTS + part
    tl.store(PM + offset, maximum)
    tl.store(PS + offset, tl.sum(exp_shifted, 0))
    tl.store(PQ + offset, tl.sum(exp_shifted * shifted, 0))


@triton.jit
def vocab_entropy_finish_kernel(PM, PS, PQ, H, PARTS: tl.constexpr,
                                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    valid = part < PARTS
    maximums = tl.load(PM + row * PARTS + part, valid, other=-float("inf"))
    totals = tl.load(PS + row * PARTS + part, valid, other=0.0)
    moments = tl.load(PQ + row * PARTS + part, valid, other=0.0)
    maximum = tl.max(maximums, 0)
    offsets = tl.where(valid, maximums - maximum, 0.0)
    weights = tl.where(valid, tl.exp(offsets), 0.0)
    total = tl.sum(weights * totals, 0)
    moment = tl.sum(weights * (moments + offsets * totals), 0)
    tl.store(H + row, tl.log(total) - moment / total)


def token_entropy(logits):
    """Detached fp32 entropy per token; direct use is restricted to no_grad."""
    if (torch.is_grad_enabled() or logits.ndim < 2 or min(logits.shape) < 1
            or logits.dtype != torch.float32 or not logits.is_contiguous()):
        raise ValueError("expected contiguous fp32 logits with no_grad")
    vocab = logits.shape[-1]
    flat = logits.reshape(-1, vocab)
    rows = flat.shape[0]
    parts = triton.cdiv(vocab, TILE)
    partial_max = torch.empty((rows, parts), device=logits.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    partial_moment = torch.empty_like(partial_max)
    entropy = torch.empty((rows,), device=logits.device, dtype=torch.float32)
    vocab_entropy_partial_kernel[(rows, parts)](
        flat, partial_max, partial_sum, partial_moment, vocab, parts, TILE, num_warps=4,
    )
    vocab_entropy_finish_kernel[(rows,)](
        partial_max, partial_sum, partial_moment, entropy, parts,
        triton.next_power_of_2(parts), num_warps=4,
    )
    return entropy.reshape(logits.shape[:-1])


def _eligible(logits, chunk_size):
    return (isinstance(logits, torch.Tensor) and logits.is_cuda
            and not torch.is_grad_enabled() and logits.ndim == 3
            and logits.numel() > 0 and logits.shape[-1] == VOCAB_SIZE
            and logits.dtype == torch.float32 and logits.is_contiguous()
            and chunk_size == 128)


def patch_trl_entropy():
    """Replace only the entropy helper imported by TRL 0.29.1 SFTTrainer."""
    if version("trl") != "0.29.1":
        return False
    from trl.trainer import sft_trainer

    original = sft_trainer.entropy_from_logits
    if getattr(original, "_soup_vocab_entropy", False):
        return True

    @wraps(original)
    def entropy_from_logits(logits, chunk_size=128):
        if _eligible(logits, chunk_size):
            return token_entropy(logits)
        return original(logits, chunk_size)

    entropy_from_logits._soup_vocab_entropy = True
    sft_trainer.entropy_from_logits = entropy_from_logits
    return True
