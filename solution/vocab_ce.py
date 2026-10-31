"""Original tiled fp32 vocabulary cross-entropy, with a narrow Transformers hook.

Only fixed_cross_entropy's contiguous CUDA [tokens, 151936] fp32 / int64
case is replaced. ForCausalLMLoss still owns casting and causal label shifting;
TRL metrics and Soup rewind still consume the original logits. No full
log-softmax/probability tensor is materialized by these kernels.
"""

from functools import wraps
from importlib.metadata import version

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


VOCAB_SIZE = 151936
TILE = 2048
_IGNORED_MODEL_OUTPUT_KWARGS = frozenset({"output_attentions", "output_hidden_states"})


@triton.jit
def vocab_ce_partial_kernel(X, Y, PM, PS, V: tl.constexpr, PARTS: tl.constexpr,
                            IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    label = tl.load(Y + row)
    if label != IGNORE:
        col = part * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(X + row * V + col, mask=col < V, other=-float("inf"))
        maximum = tl.max(values, axis=0)
        total = tl.sum(tl.exp(values - maximum), axis=0)
        tl.store(PM + row * PARTS + part, maximum)
        tl.store(PS + row * PARTS + part, total)


@triton.jit
def vocab_ce_finish_kernel(X, Y, PM, PS, LOSS, STATS, V: tl.constexpr,
                           PARTS: tl.constexpr, IGNORE: tl.constexpr,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    label = tl.load(Y + row)
    maximum, log_total, loss = 0.0, 0.0, 0.0
    if label != IGNORE:
        valid_label = (label >= 0) & (label < V)
        tl.device_assert(valid_label, "vocabulary target out of bounds")
        part = tl.arange(0, BLOCK)
        maxima = tl.load(PM + row * PARTS + part, part < PARTS, other=-float("inf"))
        totals = tl.load(PS + row * PARTS + part, part < PARTS, other=0.0)
        maximum = tl.max(maxima, axis=0)
        total = tl.sum(totals * tl.exp(maxima - maximum), axis=0)
        log_total = tl.log(total)
        target_value = tl.load(X + row * V + label, valid_label, other=float("nan"))
        # Keep max and log(sum(exp(x-max))) separate to avoid cancellation
        # when logits have a large common offset.
        loss = (maximum - target_value) + log_total
    tl.store(LOSS + row, loss)
    tl.store(STATS + 2 * row, maximum)
    tl.store(STATS + 2 * row + 1, log_total)


@triton.jit
def vocab_ce_backward_kernel(X, Y, STATS, DLOSS, DX, V: tl.constexpr,
                             IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    label = tl.load(Y + row)
    dx = tl.full((BLOCK,), 0, tl.float32)
    if label != IGNORE:
        maximum = tl.load(STATS + 2 * row)
        log_total = tl.load(STATS + 2 * row + 1)
        values = tl.load(X + row * V + col, col < V, other=-float("inf"))
        probability = tl.exp((values - maximum) - log_total)
        upstream = tl.load(DLOSS + row)
        dx = (probability - (col == label).to(tl.float32)) * upstream
    # Ignored rows are explicitly zero even when upstream is NaN/Inf
    # (e.g. the native all-ignored mean-loss 0/0 case).
    tl.store(DX + row * V + col, dx, col < V)


class _TokenCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, target, ignore_index):
        rows, vocab = source.shape
        parts = triton.cdiv(vocab, TILE)
        partial_max = torch.empty((rows, parts), device=source.device, dtype=torch.float32)
        partial_sum = torch.empty_like(partial_max)
        loss = torch.empty((rows,), device=source.device, dtype=torch.float32)
        stats = torch.empty((rows, 2), device=source.device, dtype=torch.float32)
        vocab_ce_partial_kernel[(rows, parts)](
            source, target, partial_max, partial_sum, vocab, parts, ignore_index, TILE,
            num_warps=4,
        )
        vocab_ce_finish_kernel[(rows,)](
            source, target, partial_max, partial_sum, loss, stats, vocab, parts,
            ignore_index, triton.next_power_of_2(parts), num_warps=4, debug=True,
        )
        ctx.save_for_backward(source, target, stats)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        source, target, stats = ctx.saved_tensors
        rows, vocab = source.shape
        grad_source = torch.empty_like(source)
        vocab_ce_backward_kernel[(rows, triton.cdiv(vocab, TILE))](
            source, target, stats, grad_loss.contiguous(), grad_source,
            vocab, ctx.ignore_index, TILE, num_warps=4,
        )
        return grad_source, None, None


def token_cross_entropy(source, target, ignore_index=-100):
    """Unreduced CE for contiguous fp32 rows and int64 labels on the same device.

    Internal kernel entry point: positive token/vocabulary dimensions, valid
    labels or ignore_index, and first-order gradients. CPU tensors are usable
    through Triton's interpreter for development tests, not normal execution.
    """
    if (source.ndim != 2 or target.ndim != 1 or source.shape[0] != target.numel()
            or min(source.shape) < 1 or source.dtype != torch.float32
            or target.dtype != torch.int64 or source.device != target.device
            or not source.is_contiguous() or not target.is_contiguous()):
        raise ValueError("expected contiguous fp32 [N,V] and int64 [N], N,V > 0")
    return _TokenCrossEntropy.apply(source, target, ignore_index)


def fixed_vocab_cross_entropy(source, target, num_items_in_batch=None, ignore_index=-100):
    """Match Transformers 5.16.1's sum/external-divisor or valid-token mean."""
    loss = token_cross_entropy(source, target, ignore_index).sum()
    if num_items_in_batch is None:
        divisor = (target != ignore_index).sum()
    else:
        divisor = num_items_in_batch
        if torch.is_tensor(divisor):
            divisor = divisor.to(loss.device)
    return loss / divisor


def _eligible(source, target, ignore_index):
    return (isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor)
            and source.is_cuda and source.dtype == torch.float32
            and source.ndim == 2 and source.shape[0] > 0 and source.shape[1] == VOCAB_SIZE
            and target.dtype == torch.int64 and target.ndim == 1
            and target.numel() == source.shape[0] and target.device == source.device
            and source.is_contiguous() and target.is_contiguous() and ignore_index == -100)


def patch_transformers_cross_entropy():
    """Idempotent, version-bound hook; unsupported inputs use the exact original."""
    if version("transformers") != "5.16.1":
        return False
    from transformers.loss import loss_utils

    original = loss_utils.fixed_cross_entropy
    if getattr(original, "_soup_vocab_ce", False):
        return True

    @wraps(original)
    def fixed_cross_entropy(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
        # PEFT forwards these Qwen2 model-output flags to ForCausalLMLoss;
        # Transformers' fixed_cross_entropy ignores them as well.
        if kwargs.keys() <= _IGNORED_MODEL_OUTPUT_KWARGS and _eligible(source, target, ignore_index):
            return fixed_vocab_cross_entropy(source, target, num_items_in_batch, ignore_index)
        return original(source, target, num_items_in_batch, ignore_index, **kwargs)

    fixed_cross_entropy._soup_vocab_ce = True
    loss_utils.fixed_cross_entropy = fixed_cross_entropy
    return True
