"""Run unchanged official validation and export evidence even after a failure.

Run as a script, after the notebook's dependency/clone/data preparation cells.
No package installs, task edits, or benchmark patches are performed here.
"""

import argparse
import csv
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


EXPECTED = {
    "vocab_ce_partial_kernel": "solution/vocab_ce.py",
    "vocab_ce_finish_kernel": "solution/vocab_ce.py",
    "vocab_ce_backward_kernel": "solution/vocab_ce.py",
    "vocab_entropy_partial_kernel": "solution/entropy.py",
    "vocab_entropy_finish_kernel": "solution/entropy.py",
    "nested_nf4_dequant_kernel": "solution/nf4.py",
}


def audit_launches(out):
    rows = []
    for repeat in (1, 2):
        path = out / f"solution_{repeat}/metrics.json"
        metrics = json.loads(path.read_text())
        for name, source in EXPECTED.items():
            assert metrics["triton_launches"].get(name, 0) > 0, (path, name, "no launch")
            assert metrics["triton_files"].get(name) == source, (path, name, "wrong source")
            assert metrics["own_kernel_ms"].get(name, 0) > 0, (path, name, "not timed")
        milliseconds = sum(metrics["own_kernel_ms"][name] for name in EXPECTED)
        rows.append({"run": path.parent.name, "eligible_ms": milliseconds,
                     "timed_loop_s": metrics["timed_loop_s"],
                     "share_pct": milliseconds / (10 * metrics["timed_loop_s"])})
    return rows


def export_evidence(repo, out, archive):
    names = {"result.json", "checks.json", "metrics.json", "profile.json", "task.yaml",
             "trace.json", "operators.json", "projections.json", "validation.json",
             "launch_audit.json", "trainer_state.json", "rewind.jsonl", "adapter_config.json"}
    files = [p for p in out.rglob("*") if p.is_file() and
             (p.name in names or p.suffix == ".log" or p.name.startswith("eval_") and p.suffix == ".json")]
    files += list((repo / "solution").glob("*.py"))
    files += [repo / name for name in ("BENCHMARKS.md", "PROGRESS.md", "PROJECTION_PATH.md")]
    manifest = {}
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            if not path.exists():
                continue
            relative = str(path.relative_to(repo))
            bundle.write(path, relative)
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        for profile in sorted(out.rglob("profile.json")):
            try:
                data = json.loads(profile.read_text())
                if not data.get("total_gpu_us") or not data.get("active_steps"):
                    continue
            except (ValueError, OSError):
                continue  # The raw failed/partial profile is already in the archive.
            table = io.StringIO()
            writer = csv.writer(table)
            writer.writerow(["name", "time_us", "count", "gpu_pct", "ms_per_active_step"])
            for row in data.get("kernels", []):
                writer.writerow([row["name"], row["time_us"], row["count"],
                                 100 * row["time_us"] / data["total_gpu_us"],
                                 row["time_us"] / (1000 * data["active_steps"])])
            bundle.writestr(str(profile.relative_to(repo).with_suffix(".csv")), table.getvalue())
        bundle.writestr("sha256.json", json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs_m43")
    parser.add_argument("--diagnose", action="store_true", help="extra instrumented 24-step run")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = (repo / args.out).resolve()
    out.relative_to(repo)  # evidence paths must stay inside this checkout
    archive = repo.parent / (out.name + "-evidence.zip")
    if archive.exists():
        parser.error(f"Already exists: {archive}; use a fresh --out name")
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
    env.pop("TRITON_INTERPRET", None)
    evidence = {"commands": [], "diagnostic_is_not_benchmark": True}
    status = 0

    def run(label, command):
        print("Running:", " ".join(command), flush=True)
        with (out / (label + ".log")).open("w") as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = child.wait()
        evidence["commands"].append({"command": command, "exit_code": code})
        return code

    try:
        evidence["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        evidence["git_status"] = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
        evidence["versions"] = {p: version(p) for p in
                                ("soup-cli", "torch", "triton", "transformers", "peft", "trl", "bitsandbytes", "accelerate")}
        status = run("cuda", [sys.executable, "-c", "import torch; assert torch.cuda.is_available(); "
                              "assert 'T4' in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0))"])
        if status == 0:
            status = run("solution_tests", [sys.executable, "-m", "pytest", "solution", "-q", "-p", "no:cacheprovider"])
        if status == 0:
            status = run("benchmark", [sys.executable, "harness/run_task.py", "--repeats", "2", "--profile", "--out", str(out)])
        if (out / "result.json").exists():
            code = run("admission", [sys.executable, "harness/check_rules.py", "--result", str(out / "result.json")])
            status = status or code
            code = run("score", [sys.executable, "harness/score.py", str(out / "result.json")])
            status = status or code
            try:
                rows = audit_launches(out)
                (out / "launch_audit.json").write_text(json.dumps(rows, indent=2))
                print(json.dumps(rows, indent=2))
            except (AssertionError, KeyError, OSError, ValueError) as exc:
                evidence["launch_audit_error"] = str(exc)
                status = status or 1
            if args.diagnose:
                code = run("diagnostic", [sys.executable, "-m", "solution.diagnose_projections",
                                          "--out", str(out / "diagnostic")])
                status = status or code
    except Exception as exc:
        evidence["error"] = repr(exc)
        status = 1
    finally:
        evidence["exit_code"] = status
        (out / "validation.json").write_text(json.dumps(evidence, indent=2))
        export_evidence(repo, out, archive)
        print(f"Evidence: {archive}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
