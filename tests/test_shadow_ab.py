"""Shadow A/B tests.

We avoid loading a real TF model by writing a synthetic shadow-predictions
JSONL and exercising the A/B comparison logic on it. The key invariants:

* Both production and candidate predictions are logged per bar.
* The candidate signal never replaces the production signal (separate fields).
* The comparison report is generated with the expected schema.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models import ab_compare


def _write_shadow_log(root: Path, candidate_id: str, rows: list[dict]) -> Path:
    shadow_dir = root / "reports" / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    path = shadow_dir / f"shadow_predictions_{candidate_id}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def _cfg(root: Path) -> Path:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "config.yaml").write_text(
        "retraining:\n"
        "  shadow_ab_testing:\n"
        "    min_shadow_signals: 3\n",
        encoding="utf-8",
    )
    return root / "configs" / "config.yaml"


def _row(prod_sig, cand_sig, *, up_p=0.5, down_p=0.3, tq=0.6):
    return {
        "timestamp_idx": 0,
        "asset": "BTCUSDT", "timeframe": "1h",
        "production_model_id": "prod", "candidate_model_id": "cand",
        "production_signal": prod_sig, "candidate_signal": cand_sig,
        "production_probs": {"down": down_p, "sideways": 0.2, "up": up_p},
        "candidate_probs": {"down": down_p, "sideways": 0.2, "up": up_p},
        "production_trade_quality": tq, "candidate_trade_quality": tq,
    }


def test_both_models_logged_separately(tmp_path: Path, monkeypatch):
    import yaml
    _cfg(tmp_path)
    rows = [_row("no_trade", "long_bias"), _row("long_bias", "long_bias")]
    _write_shadow_log(tmp_path, "cand", rows)
    monkeypatch.setattr(ab_compare, "repo_root", lambda: tmp_path)
    cfg = yaml.safe_load((tmp_path / "configs" / "config.yaml").read_text())
    summary = ab_compare.compare("cand", root=tmp_path, cfg=cfg)
    # Production and candidate signals are tracked independently
    assert summary["production_trade_signals"] == 1   # only the 2nd row
    assert summary["candidate_trade_signals"] == 2     # both rows
    assert summary["candidate_only_signals"] == 1      # 1st row


def test_agreement_rate(tmp_path: Path, monkeypatch):
    import yaml
    _cfg(tmp_path)
    rows = [
        _row("long_bias", "long_bias"),
        _row("no_trade", "no_trade"),
        _row("no_trade", "short_bias"),
    ]
    _write_shadow_log(tmp_path, "cand", rows)
    monkeypatch.setattr(ab_compare, "repo_root", lambda: tmp_path)
    cfg = yaml.safe_load((tmp_path / "configs" / "config.yaml").read_text())
    summary = ab_compare.compare("cand", root=tmp_path, cfg=cfg)
    assert abs(summary["signal_agreement_rate"] - (2 / 3)) < 1e-9


def test_compare_writes_reports_and_does_not_promote(tmp_path: Path, monkeypatch):
    import yaml
    _cfg(tmp_path)
    rows = [_row("no_trade", "long_bias") for _ in range(5)]
    _write_shadow_log(tmp_path, "cand", rows)
    monkeypatch.setattr(ab_compare, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(ab_compare.registry, "sync_runs", lambda root=None: [])
    rc = ab_compare.main(["--candidate", "cand", "--sample", "true"])
    assert rc == 0
    md = tmp_path / "reports" / "shadow" / "shadow_compare_cand.md"
    js = tmp_path / "reports" / "shadow" / "shadow_compare_cand.json"
    assert md.exists() and js.exists()
    data = json.loads(js.read_text(encoding="utf-8"))
    # 5 candidate signals ≥ min_shadow_signals(3) AND no production trades → still informational
    assert "promotion_recommended_after_shadow" in data
    # Comparison must never itself change the registry — there's no production set here.
    reg_path = tmp_path / "metadata" / "model_registry.json"
    assert not reg_path.exists() or json.loads(reg_path.read_text()).get("models", []) == []


def test_missing_shadow_log_returns_none(tmp_path: Path, monkeypatch):
    import yaml
    _cfg(tmp_path)
    monkeypatch.setattr(ab_compare, "repo_root", lambda: tmp_path)
    cfg = yaml.safe_load((tmp_path / "configs" / "config.yaml").read_text())
    assert ab_compare.compare("does_not_exist", root=tmp_path, cfg=cfg) is None
