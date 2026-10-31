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
import math
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
KNOWN_GOOD = "ed35c4b947a0076445ad4ed164b1d55caaf3b729"


def speed_decision(out, source_sha):
    """Conservative promotion gate; never changes official results/checks."""
    raw = (out / "result.json").read_bytes()
    result = json.loads(raw)
    checks = json.loads((out / "checks.json").read_text())
    errors = []
    if checks.get("result_sha256") != hashlib.sha256(raw).hexdigest():
        errors.append("checks do not match result.json")
    if checks.get("solution_sha256") != source_sha:
        errors.append("checks do not match measured solution source")
    rows = checks.get("checks", [])
    if len(rows) != 10 or not all(row.get("ok") is True for row in rows):
        errors.append("all 10 official checks must pass")
    if (result.get("task", {}).get("repeats") != 2
            or any(len(result.get("runs", {}).get(side, [])) != 2 for side in ("base", "solution"))):
        errors.append("two paired official repeats are required")
    ratios = result.get("summary", {}).get("ratios", {})
    speed = ratios.get("speedup")
    memory = ratios.get("memory_ratio")
    valid = lambda x: isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)
    if not valid(speed) or speed < 1.05:
        errors.append("speedup below the 1.05x goal or missing")
    if valid(speed) and speed < 1.038:
        errors.append("speedup worse than the user-verified 1.038x reference")
    if not valid(memory) or memory < 1.158:
        errors.append("memory ratio below the user-verified 1.158x reference or missing")
    launches = audit_launches(out)
    minimum_share = min(row["share_pct"] for row in launches)
    if not math.isfinite(minimum_share) or minimum_share < 5:
        errors.append("eligible share below 5% in at least one repeat")
    return {"keep_candidate": not errors, "reasons": errors, "speedup": speed,
            "memory_ratio": memory, "minimum_share_pct": minimum_share,
            "preferred_1_08x_reached": valid(speed) and speed >= 1.08,
            "known_good_commit": KNOWN_GOOD,
            "reference": "user-reported historical T4 ratios; no same-session ed35c4b rerun"}


def rollback_checkout(repo, measured_commit):
    """Restore the clean Kaggle checkout only after candidate evidence is archived."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    if git("rev-parse", "HEAD") != measured_commit:
        raise RuntimeError("HEAD changed during validation; rollback requires manual review")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files changed; refusing to overwrite them during rollback")
    git("cat-file", "-e", KNOWN_GOOD + "^{commit}")
    subprocess.run(["git", "checkout", "--detach", KNOWN_GOOD], cwd=repo, check=True)
    return {"restored_commit": git("rev-parse", "HEAD"), "experiment_commit": measured_commit}


def nf4_cache_audit(out):
    """Read metadata after measured processes exit; export no compiled caches."""
    rows = {}
    for repeat in (1, 2):
        run = out / f"solution_{repeat}"
        entries = []
        for path in sorted((run / "triton_cache").rglob("nested_nf4_dequant_kernel.json")):
            entry = json.loads(path.read_text())
            entries.append({key: entry.get(key) for key in
                            ("hash", "name", "num_warps", "shared", "triton_version", "target")})
        rows[run.name] = {"metadata_entries": len(entries), "kernels": entries}
    return rows


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
             "launch_audit.json", "trainer_state.json", "rewind.jsonl", "adapter_config.json",
             "speed_decision.json", "nf4_cache_audit.json"}
    files = [p for p in out.rglob("*") if p.is_file() and
             (p.name in names or p.suffix == ".log" or p.name.startswith("eval_") and p.suffix == ".json")]
    files += list((repo / "solution").glob("*.py"))
    files += [repo / name for name in ("BENCHMARKS.md", "PROGRESS.md", "PROJECTION_PATH.md", "SPEED_EXPERIMENT.md")]
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
    parser.add_argument("--speed-gate", action="store_true",
                        help="M4.4: require >=1.05x speed, preserve memory/share; rollback on failure")
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
            command = [sys.executable, "harness/run_task.py", "--repeats", "2", "--out", str(out)]
            if not args.speed_gate:
                command.append("--profile")
            status = run("benchmark", command)
        if (out / "result.json").exists():
            code = run("admission", [sys.executable, "harness/check_rules.py", "--result", str(out / "result.json")])
            status = status or code
            code = run("score", [sys.executable, "harness/score.py", str(out / "result.json")])
            status = status or code
            try:
                rows = audit_launches(out)
                (out / "launch_audit.json").write_text(json.dumps(rows, indent=2))
                print(json.dumps(rows, indent=2))
                (out / "nf4_cache_audit.json").write_text(json.dumps(nf4_cache_audit(out), indent=2))
            except (AssertionError, KeyError, OSError, ValueError) as exc:
                evidence["launch_audit_error"] = str(exc)
                status = status or 1
            if args.speed_gate:
                sys.path.insert(0, str(repo / "harness"))
                from check_rules import solution_sha256
                decision = speed_decision(out, solution_sha256(repo / "solution"))
                (out / "speed_decision.json").write_text(json.dumps(decision, indent=2))
                print(json.dumps(decision, indent=2))
                if not decision["keep_candidate"]:
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
        if args.speed_gate and status:
            try:
                rollback = rollback_checkout(repo, evidence.get("commit"))
            except (RuntimeError, subprocess.CalledProcessError) as exc:
                rollback = {"error": str(exc), "manual_restore_commit": KNOWN_GOOD}
            (out / "rollback.json").write_text(json.dumps(rollback, indent=2))
            with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(out / "rollback.json", (out / "rollback.json").relative_to(repo))
            print(json.dumps(rollback, indent=2), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
