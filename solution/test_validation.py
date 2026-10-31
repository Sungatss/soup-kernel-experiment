"""Synthetic CPU fixtures for the post-run promotion/rollback policy, not benchmarks."""

import hashlib
import json
import subprocess

import pytest

from solution import validate_kaggle as validation


@pytest.fixture
def evidence(tmp_path):
    result = {"task": {"repeats": 2}, "runs": {"base": [{}, {}], "solution": [{}, {}]},
              "summary": {"ratios": {"speedup": 1.06, "memory_ratio": 1.16}}}
    checks = {"checks": [{"ok": True} for _ in range(10)], "solution_sha256": "fixture-source"}
    for i in (1, 2):
        path = tmp_path / f"solution_{i}"
        path.mkdir()
        metrics = {"triton_launches": {name: 1 for name in validation.EXPECTED},
                   "triton_files": validation.EXPECTED,
                   "own_kernel_ms": {name: 300 for name in validation.EXPECTED},
                   "timed_loop_s": 30}
        (path / "metrics.json").write_text(json.dumps(metrics))

    def save():
        raw = json.dumps(result).encode()
        (tmp_path / "result.json").write_bytes(raw)
        checks["result_sha256"] = hashlib.sha256(raw).hexdigest()
        (tmp_path / "checks.json").write_text(json.dumps(checks))

    save()
    return tmp_path, result, checks, save


def test_speed_gate_accepts_only_complete_evidence(evidence):
    path, *_ = evidence
    decision = validation.speed_decision(path, "fixture-source")
    assert decision["keep_candidate"]
    assert decision["minimum_share_pct"] == 6
    assert not decision["preferred_1_08x_reached"]


@pytest.mark.parametrize("case", ["check", "count", "repeat", "speed", "regression", "memory",
                                  "source_hash", "result_hash", "share", "nan"])
def test_speed_gate_rejects_failed_or_stale_evidence(evidence, case):
    path, result, checks, save = evidence
    if case == "check":
        checks["checks"][3]["ok"] = False
    elif case == "count":
        checks["checks"].pop()
    elif case == "repeat":
        result["runs"]["solution"].pop()
    elif case in {"speed", "regression", "nan"}:
        result["summary"]["ratios"]["speedup"] = {"speed": 1.04, "regression": 1.02, "nan": float("nan")}[case]
    elif case == "memory":
        result["summary"]["ratios"]["memory_ratio"] = 1.10
    elif case == "source_hash":
        checks["solution_sha256"] = "different-source"
    elif case == "share":
        target = path / "solution_2/metrics.json"
        metrics = json.loads(target.read_text())
        metrics["timed_loop_s"] = 60
        target.write_text(json.dumps(metrics))
    save()
    if case == "result_hash":
        with (path / "result.json").open("a") as output:
            output.write("\n")
    assert not validation.speed_decision(path, "fixture-source")["keep_candidate"]


def test_rollback_restores_commit_and_protects_changes(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    file = tmp_path / "example.txt"
    for content in ("known-good", "candidate"):
        file.write_text(content)
        git("add", "example.txt")
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", content)
        if content == "known-good":
            monkeypatch.setattr(validation, "KNOWN_GOOD", git("rev-parse", "HEAD"))
    candidate = git("rev-parse", "HEAD")
    file.write_text("uncommitted user change")
    with pytest.raises(RuntimeError, match="tracked files changed"):
        validation.rollback_checkout(tmp_path, candidate)
    assert file.read_text() == "uncommitted user change"
    git("restore", "example.txt")
    report = validation.rollback_checkout(tmp_path, candidate)
    assert report["restored_commit"] == validation.KNOWN_GOOD
    assert file.read_text() == "known-good"
    with pytest.raises(RuntimeError, match="HEAD changed"):
        validation.rollback_checkout(tmp_path, candidate)
