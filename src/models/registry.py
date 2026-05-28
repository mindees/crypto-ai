"""Model registry — the source of truth for model lifecycle.

Backed by ``metadata/model_registry.json``. Each run trained under
``artifacts/runs/<run_id>/`` can be registered as a candidate, promoted to
production, archived, rejected, or rolled back. Artifacts are never
overwritten — the registry only moves pointers.

CLI::

    python -m src.models.registry --list
    python -m src.models.registry --register <run_id>
    python -m src.models.registry --sync        # register any unregistered runs as candidates
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils.io import read_json, repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("models.registry")

VALID_STATUSES = ("candidate", "production", "archived", "rejected", "rolled_back")


def _registry_path(root: Path) -> Path:
    return root / "metadata" / "model_registry.json"


def _current_model_path(root: Path) -> Path:
    return root / "artifacts" / "production" / "current_model.json"


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git_sha(root: Path) -> str:
    # The repo may or may not have git; degrade gracefully.
    head = root / ".git" / "HEAD"
    try:
        if head.exists():
            ref = head.read_text(encoding="utf-8").strip()
            if ref.startswith("ref:"):
                ref_path = root / ".git" / ref.split(" ", 1)[1].strip()
                if ref_path.exists():
                    return ref_path.read_text(encoding="utf-8").strip()[:12]
                return "uncommitted"  # branch ref exists but no commits yet
            return ref[:12]  # detached HEAD points directly at a SHA
    except OSError:
        pass
    return "unknown"


def load_registry(root: Path | None = None) -> dict:
    root = root or repo_root()
    path = _registry_path(root)
    if path.exists():
        return read_json(path)
    return {"schema_version": 1, "models": []}


def save_registry(payload: dict, root: Path | None = None) -> Path:
    root = root or repo_root()
    path = _registry_path(root)
    write_json(path, payload)
    return path


def _build_record(run_dir: Path, root: Path, *, status: str = "candidate") -> dict:
    metrics = {}
    # Pull metrics from a run's training history / eval if present
    th = run_dir / "training_history.json"
    if th.exists():
        hist = read_json(th)
        for key in ("val_direction_direction_macro_f1", "val_trade_quality_auc"):
            if key in hist and hist[key]:
                metrics[key] = float(hist[key][-1])
    return {
        "model_id": run_dir.name,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": status,
        "artifact_path": str(run_dir.relative_to(root)).replace("\\", "/"),
        "dataset_manifest_hash": _hash_file(run_dir / "dataset_spec.json"),
        "feature_schema_hash": _hash_file(run_dir / "dataset_spec.json"),
        "label_config_hash": _hash_file(root / "configs" / "config.yaml"),
        "git_sha": _git_sha(root),
        "metrics": {
            "direction_macro_f1": metrics.get("val_direction_direction_macro_f1", 0.0),
            "trade_quality_auc": metrics.get("val_trade_quality_auc", 0.0),
            "backtest_profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "expectancy_r": 0.0,
        },
        "promotion_decision": "not_evaluated",
        "notes": "",
    }


def register_run(run_id: str, *, status: str = "candidate", root: Path | None = None) -> dict:
    root = root or repo_root()
    run_dir = root / "artifacts" / "runs" / run_id
    if not (run_dir / "model.keras").exists():
        raise FileNotFoundError(f"no model.keras under {run_dir}")
    reg = load_registry(root)
    existing = {m["model_id"]: i for i, m in enumerate(reg["models"])}
    record = _build_record(run_dir, root, status=status)
    if run_id in existing:
        # Preserve status if already promoted/archived; only refresh metrics
        prev = reg["models"][existing[run_id]]
        record["status"] = prev.get("status", status)
        record["promotion_decision"] = prev.get("promotion_decision", "not_evaluated")
        reg["models"][existing[run_id]] = record
    else:
        reg["models"].append(record)
    save_registry(reg, root)
    return record


def sync_runs(root: Path | None = None) -> list[str]:
    """Register any run dirs with a model.keras that aren't in the registry yet."""
    root = root or repo_root()
    reg = load_registry(root)
    known = {m["model_id"] for m in reg["models"]}
    base = root / "artifacts" / "runs"
    newly: list[str] = []
    if base.exists():
        for run_dir in sorted(base.iterdir()):
            if run_dir.is_dir() and (run_dir / "model.keras").exists() and run_dir.name not in known:
                register_run(run_dir.name, root=root)
                newly.append(run_dir.name)
    return newly


def set_status(model_id: str, status: str, *, root: Path | None = None,
               promotion_decision: str | None = None, note: str | None = None) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}; must be one of {VALID_STATUSES}")
    root = root or repo_root()
    reg = load_registry(root)
    for m in reg["models"]:
        if m["model_id"] == model_id:
            m["status"] = status
            if promotion_decision is not None:
                m["promotion_decision"] = promotion_decision
            if note is not None:
                m["notes"] = note
            save_registry(reg, root)
            return
    raise KeyError(f"model_id {model_id!r} not in registry")


def get_model(model_id: str, *, root: Path | None = None) -> dict | None:
    reg = load_registry(root)
    for m in reg["models"]:
        if m["model_id"] == model_id:
            return m
    return None


def get_production(root: Path | None = None) -> dict | None:
    reg = load_registry(root)
    prod = [m for m in reg["models"] if m["status"] == "production"]
    return prod[-1] if prod else None


def latest_candidate(root: Path | None = None) -> dict | None:
    reg = load_registry(root)
    cands = [m for m in reg["models"] if m["status"] == "candidate"]
    return cands[-1] if cands else None


def set_current_model_pointer(model_id: str, *, root: Path | None = None) -> None:
    root = root or repo_root()
    rec = get_model(model_id, root=root)
    write_json(_current_model_path(root), {
        "schema_version": 1,
        "current_model_id": model_id,
        "promoted_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "artifact_path": rec["artifact_path"] if rec else None,
        "notes": "",
    })


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.registry")
    p.add_argument("--list", action="store_true")
    p.add_argument("--register", default=None, help="run_id to register as candidate")
    p.add_argument("--sync", action="store_true", help="register all unregistered runs")
    args = p.parse_args(argv)

    root = repo_root()

    if args.register:
        rec = register_run(args.register, root=root)
        print(f"registered {rec['model_id']} as {rec['status']}")
        return 0

    # --list always syncs first so it reflects every trained run.
    newly = sync_runs(root)
    if newly:
        print(f"(synced {len(newly)} new run(s) as candidates)")

    reg = load_registry(root)
    models = reg.get("models", [])
    if not models:
        print("model registry is empty. Train a model then run with --sync or --register <run_id>.")
        return 0

    print(f"{'model_id':<40} {'status':<12} {'dir_macro_f1':>12} {'tq_auc':>8} {'promotion'}")
    for m in models:
        met = m.get("metrics", {})
        print(
            f"{m['model_id']:<40} {m['status']:<12} "
            f"{met.get('direction_macro_f1', 0.0):>12.4f} "
            f"{met.get('trade_quality_auc', 0.0):>8.4f} {m.get('promotion_decision', '')}"
        )
    prod = get_production(root)
    print(f"\nproduction: {prod['model_id'] if prod else '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
