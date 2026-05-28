"""Drift detection + visualization tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import drift, drift_viz


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    assert drift.compute_psi(x, x) < 1e-6


def test_psi_increases_with_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    small = rng.normal(0.3, 1, 2000)
    big = rng.normal(2.0, 1, 2000)
    psi_small = drift.compute_psi(ref, small)
    psi_big = drift.compute_psi(ref, big)
    assert psi_big > psi_small > 0


def test_severity_thresholds():
    assert drift.severity(0.05) == drift.STABLE
    assert drift.severity(0.15) == drift.MODERATE
    assert drift.severity(0.40) == drift.SIGNIFICANT


def test_feature_drift_table_sorted_desc():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({
        "stable": rng.normal(0, 1, 1000),
        "drifted": rng.normal(0, 1, 1000),
    })
    cur = pd.DataFrame({
        "stable": rng.normal(0, 1, 500),
        "drifted": rng.normal(3, 1, 500),
    })
    table = drift.feature_drift_table(ref, cur)
    assert table[0].feature == "drifted"
    assert table[0].psi >= table[1].psi


def test_low_variance_feature_does_not_crash():
    ref = pd.DataFrame({"const": np.zeros(500)})
    cur = pd.DataFrame({"const": np.zeros(200)})
    table = drift.feature_drift_table(ref, cur)
    assert table[0].psi == 0.0


def test_drift_viz_sample_generates_charts_and_dashboard(tmp_path: Path, monkeypatch):
    # Point repo_root at a temp dir with a minimal config so outputs are isolated.
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text("project:\n  seed: 42\n", encoding="utf-8")
    monkeypatch.setattr(drift_viz, "repo_root", lambda: tmp_path)

    rc = drift_viz.main(["--sample", "true"])
    assert rc == 0

    drift_dir = tmp_path / "reports" / "drift"
    pngs = list(drift_dir.glob("*.png"))
    assert len(pngs) >= 5, f"expected ≥5 charts, found {len(pngs)}"
    # Required chart families exist
    names = " ".join(p.name for p in pngs)
    for token in ("psi_top_features", "prediction_distribution_drift",
                  "regime_distribution_drift", "calibration_drift", "live_expectancy_curve"):
        assert token in names, f"missing chart {token}"
    # JSON beside charts
    assert list(drift_dir.glob("drift_summary_*.json"))
    # HTML dashboard
    assert list((tmp_path / "reports").glob("drift_dashboard_*.html"))


def test_drift_summary_json_schema(tmp_path: Path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text("project:\n  seed: 42\n", encoding="utf-8")
    monkeypatch.setattr(drift_viz, "repo_root", lambda: tmp_path)
    drift_viz.main(["--sample", "true"])
    import json
    summary_files = list((tmp_path / "reports" / "drift").glob("drift_summary_*.json"))
    data = json.loads(summary_files[0].read_text(encoding="utf-8"))
    for key in ("generated_at_utc", "source", "top_features", "prediction_head_psi", "charts"):
        assert key in data
    assert isinstance(data["top_features"], list)
