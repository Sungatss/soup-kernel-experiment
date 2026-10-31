"""Separate diagnostic only: actual Soup model metadata and attributed GPU trace.

Never imported by apply(). Uses the official profiler's 24-step schedule.
These instrumented timings are not admission measurements.
"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def tensor_info(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "requires_grad": tensor.requires_grad, "stride": list(tensor.stride())}


def instrument(model, records):
    import torch

    for name, module in model.named_modules():
        if name.split(".")[-1] not in {"lm_head", "q_proj", "v_proj"}:
            continue
        base = getattr(module, "base_layer", module)
        weight = getattr(base, "weight", None)
        if weight is None:
            continue
        record = {"class": type(module).__name__, "base_class": type(base).__name__,
                  "weight": tensor_info(weight), "compute_dtype": str(getattr(base, "compute_dtype", None))}
        state = getattr(weight, "quant_state", None)
        if state is not None:
            record["quant_state"] = {"shape": list(state.shape), "dtype": str(state.dtype),
                                     "type": state.quant_type, "blocksize": state.blocksize,
                                     "nested": state.nested,
                                     "nested_blocksize": state.state2.blocksize if state.nested else None}
        if hasattr(module, "lora_A"):
            record["lora"] = {key: {"A": tensor_info(module.lora_A[key].weight),
                                     "B": tensor_info(module.lora_B[key].weight),
                                     "scale": module.scaling[key],
                                     "dropout": str(module.lora_dropout[key])}
                              for key in module.lora_A}
        records[name] = record

        def pre(mod, args, record=record, name=name):
            scope = torch.profiler.record_function("projection::" + name)
            scope.__enter__()
            record["_scope"] = scope
            if "input" not in record:
                x = args[0]
                record["input"] = tensor_info(x)
                record["autocast"] = torch.is_autocast_enabled("cuda")
                record["autocast_dtype"] = str(torch.get_autocast_dtype("cuda"))
                if x.requires_grad:
                    def input_grad(grad):
                        record["input_gradient"] = tensor_info(grad)
                    x.register_hook(input_grad)

        def post(mod, args, output, record=record):
            record.pop("_scope").__exit__(None, None, None)
            if "output" not in record and isinstance(output, torch.Tensor):
                record["output"] = tensor_info(output)
                if output.requires_grad:
                    def output_grad(grad):
                        record["output_gradient"] = tensor_info(grad)
                    output.register_hook(output_grad)

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post, always_call=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "harness"))
    from common import OFFLINE_ENV
    from run_task import make_config

    os.environ.update(OFFLINE_ENV)
    os.environ.update(SOUP_DB_PATH=str(out / "experiments.db"),
                      TRITON_CACHE_DIR=str(out / "triton_cache"),
                      TORCHINDUCTOR_CACHE_DIR=str(out / "inductor_cache"))
    config = make_config(out, repo / "data/train.jsonl", 1)
    import torch
    from soup_cli.trainer.sft import SFTTrainerWrapper

    records = {}
    original_setup = SFTTrainerWrapper._setup_transformers

    def setup(self, *a, **kw):
        result = original_setup(self, *a, **kw)
        instrument(self.model, records)
        return result

    SFTTrainerWrapper._setup_transformers = setup
    original_profile = torch.profiler.profile

    def ready(prof):
        prof.export_chrome_trace(str(out / "trace.json"))
        rows = [{"name": e.key, "input_shapes": e.input_shapes, "count": e.count,
                 "device_time_us": e.device_time_total,
                 "self_device_time_us": e.self_device_time_total}
                for e in prof.key_averages(group_by_input_shape=True)]
        (out / "operators.json").write_text(json.dumps(rows, indent=2))

    def profile(*a, **kw):
        return original_profile(*a, **kw, record_shapes=True, on_trace_ready=ready)

    torch.profiler.profile = profile
    sys.argv = [str(repo / "harness/_train.py"), "--config", str(config),
                "--metrics", str(out / "metrics.json"), "--solution",
                "--profile", str(out / "profile.json")]
    try:
        runpy.run_path(str(repo / "harness/_train.py"), run_name="__main__")
    finally:
        for record in records.values():
            record.pop("_scope", None)
        (out / "projections.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
