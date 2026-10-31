"""Fuse nested-scale reconstruction and NF4 weight dequantization.

GEMM, its autograd, and PEFT's adapter path remain in bitsandbytes/PyTorch.
Only the pinned library's existing dequantization calls are replaced.
"""

from functools import wraps
from importlib.metadata import version
from math import prod

import torch
import triton
import triton.language as tl


QWEN_SHAPES = {(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)}


@triton.jit
def nested_nf4_dequant_kernel(PACKED, SCALES8, SCALE_MAX, SCALE_CODE, OFFSET, OUT,
                              N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    scale_index = i // 64
    scale_code = tl.load(SCALES8 + scale_index, valid, other=0).to(tl.int32)
    scale_value = tl.load(SCALE_CODE + scale_code)
    scale_max = tl.load(SCALE_MAX + scale_index // 256, valid, other=0.0)
    # Preserve the two fp32 rounding steps of dequantize_blockwise + offset.
    scale = scale_value * scale_max
    scale = scale + tl.load(OFFSET)
    packed = tl.load(PACKED + i // 2, valid, other=0).to(tl.int32)
    code = tl.where(i % 2 == 0, packed >> 4, packed & 15)
    # NF4 format constants, not an approximated uniform quantizer.
    levels: tl.constexpr = (-1.0, -0.6961928009986877, -0.5250730514526367,
                           -0.39491748809814453, -0.28444138169288635,
                           -0.18477343022823334, -0.09105003625154495, 0.0,
                           0.07958029955625534, 0.16093020141124725,
                           0.24611230194568634, 0.33791524171829224,
                           0.44070982933044434, 0.5626170039176941,
                           0.7229568362236023, 1.0)
    # Select adjacent pairs, then groups of 4/8/16: four dependent stages.
    # The constants and the subsequent fp32 multiplication are unchanged.
    b0 = (code & 1) != 0
    b1 = (code & 2) != 0
    b2 = (code & 4) != 0
    b3 = (code & 8) != 0
    p0 = tl.where(b0, levels[1], levels[0])
    p1 = tl.where(b0, levels[3], levels[2])
    p2 = tl.where(b0, levels[5], levels[4])
    p3 = tl.where(b0, levels[7], levels[6])
    p4 = tl.where(b0, levels[9], levels[8])
    p5 = tl.where(b0, levels[11], levels[10])
    p6 = tl.where(b0, levels[13], levels[12])
    p7 = tl.where(b0, levels[15], levels[14])
    q0 = tl.where(b1, p1, p0)
    q1 = tl.where(b1, p3, p2)
    q2 = tl.where(b1, p5, p4)
    q3 = tl.where(b1, p7, p6)
    value = tl.where(b3, tl.where(b2, q3, q2), tl.where(b2, q1, q0))
    tl.store(OUT + i, value * scale, valid)


def dequantize_nested(packed, scales8, scale_max, scale_code, offset, shape):
    """Internal primitive: canonical fp16 [out_features, in_features]."""
    result = torch.empty(shape, device=packed.device, dtype=torch.float16)
    n = prod(shape)
    nested_nf4_dequant_kernel[(triton.cdiv(n, 1024),)](
        packed, scales8, scale_max, scale_code, offset, result, n, 1024,
        num_warps=4, enable_fp_fusion=False,
    )
    return result


def _on_cuda(tensor):
    return tensor.is_cuda


def _supported(packed, shape, scales8, scale_max, scale_code, offset):
    if tuple(shape) not in QWEN_SHAPES:
        return False
    n = prod(shape)
    specs = ((packed, torch.uint8, n // 2), (scales8, torch.uint8, n // 64),
             (scale_max, torch.float32, triton.cdiv(n // 64, 256)),
             (scale_code, torch.float32, 256), (offset, torch.float32, 1))
    return all(isinstance(x, torch.Tensor) and _on_cuda(x)
               and x.device == packed.device and x.dtype == dtype
               and x.numel() == count and x.is_contiguous() and not x.requires_grad
               for x, dtype, count in specs)


def _state_supported(packed, state):
    return (state is not None and state.quant_type == "nf4"
            and state.dtype == torch.float16 and state.blocksize == 64
            and state.nested and state.state2.blocksize == 256
            and state.state2.dtype == torch.float32 and not state.state2.nested
            and packed.ndim == 2 and 1 in packed.shape
            and _supported(packed, state.shape, state.absmax, state.state2.absmax,
                           state.state2.code, state.offset))


def _patch_cuda_fallback(ops):
    original = ops._dequant_linear_fallback
    if getattr(original, "_soup_nested_nf4", False):
        return

    @wraps(original)
    def fallback(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                 absmax_8bit=None, absmax_code=None, absmax_offset=None):
        if (A.dtype == torch.float16 and _on_cuda(A) and A.device == B.device
                and blocksize == 64 and quant_type == "nf4"
                and _supported(B, shapeB, absmax_8bit, absmax, absmax_code, absmax_offset)):
            weight = dequantize_nested(B, absmax_8bit, absmax, absmax_code, absmax_offset, shapeB)
            return torch.nn.functional.linear(A, weight, bias)
        return original(A, B, shapeB, absmax, blocksize, quant_type, bias=bias,
                        absmax_8bit=absmax_8bit, absmax_code=absmax_code, absmax_offset=absmax_offset)

    fallback._soup_nested_nf4 = True
    ops._dequant_linear_fallback = fallback


def patch_bitsandbytes_nf4():
    """Keep native fused-GEMM dispatch; fuse only existing fallback/backward work."""
    if version("bitsandbytes") != "0.50.2":
        return False
    import bitsandbytes.functional as F

    original = F.dequantize_4bit
    if not getattr(original, "_soup_nested_nf4", False):
        @wraps(original)
        def dequantize_4bit(A, quant_state=None, absmax=None, out=None,
                            blocksize=None, quant_type="fp4"):
            if out is None and _state_supported(A, quant_state):
                state = quant_state
                result = dequantize_nested(A, state.absmax, state.state2.absmax,
                                          state.state2.code, state.offset, state.shape)
                return result.t() if A.shape[0] == 1 else result
            return original(A, quant_state, absmax, out, blocksize, quant_type)

        dequantize_4bit._soup_nested_nf4 = True
        F.dequantize_4bit = dequantize_4bit

    if torch.cuda.is_available():
        from bitsandbytes.backends.cuda import ops
        _patch_cuda_fallback(ops)
    return True
