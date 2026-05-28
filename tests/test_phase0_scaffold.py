"""Phase 0 sanity tests — only checks the scaffold is wired correctly.

Heavier tests (no-lookahead, labeling, ingestion idempotency, etc.) land in
Phases 1+ and live in their own files per the spec.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import src
from src.utils import io as io_utils
from src.utils import time as time_utils
from src.utils.hardware import detect_hardware
from src.utils.seeds import set_global_seed
from src.utils.validation import REQUIRED_OHLCV_COLUMNS, assert_columns


def test_package_imports_and_version():
    assert src.__version__ == "0.1.0"


def test_repo_root_contains_config():
    root = io_utils.repo_root()
    assert (root / "configs" / "config.yaml").exists()
    assert (root / "requirements.txt").exists()


def test_config_yaml_loads_with_expected_keys():
    root = io_utils.repo_root()
    cfg = io_utils.read_yaml(root / "configs" / "config.yaml")
    for key in (
        "project",
        "assets",
        "market_types",
        "timeframes",
        "features",
        "labels",
        "validation",
        "model",
        "backtest",
        "retraining",
        "serving",
    ):
        assert key in cfg, f"missing top-level config key: {key}"
    assert cfg["assets"] == ["BTCUSDT", "ETHUSDT"]
    assert cfg["project"]["timezone"] == "UTC"


def test_halvings_reference_file_present():
    root = io_utils.repo_root()
    halvings = root / "reference" / "halvings.csv"
    assert halvings.exists()
    lines = halvings.read_text(encoding="utf-8").strip().splitlines()
    # header + 4 confirmed halvings + 1 projected
    assert len(lines) >= 5


def test_timeframe_ms_covers_all_supported_intervals():
    for tf in [
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1mo",
    ]:
        assert time_utils.timeframe_ms(tf) > 0

    with pytest.raises(ValueError):
        time_utils.timeframe_ms("7m")


def test_required_ohlcv_columns_validator():
    import pandas as pd

    good = pd.DataFrame(
        {c: [1] for c in REQUIRED_OHLCV_COLUMNS},
    )
    assert_columns(good, REQUIRED_OHLCV_COLUMNS)

    bad = good.drop(columns=["close"])
    with pytest.raises(ValueError):
        assert_columns(bad, REQUIRED_OHLCV_COLUMNS)


def test_seed_setter_is_idempotent_and_safe():
    set_global_seed(42)
    set_global_seed(42)  # repeated call must not raise


def test_hardware_detection_returns_required_keys():
    report = detect_hardware()
    for key in (
        "environment",
        "platform",
        "python_version",
        "tensorflow_installed",
        "gpu_count",
        "would_use_mirrored_strategy",
        "is_kaggle_dual_t4",
    ):
        assert key in report


def test_directory_scaffold_exists():
    root = io_utils.repo_root()
    required_dirs = [
        "src/ingest", "src/features", "src/labels", "src/datasets",
        "src/models", "src/backtest", "src/strategies", "src/serve", "src/utils",
        "configs", "metadata", "reference",
        "data/raw", "data/interim", "data/processed", "data/features", "data/labels", "data/samples",
        "artifacts/runs", "artifacts/production",
        "notebooks", "tests",
        "reports", "reports/drift", "reports/shadow",
        "reports/training_plots", "reports/training_logs",
        ".github/workflows",
    ]
    for d in required_dirs:
        assert (root / d).is_dir(), f"missing scaffold dir: {d}"
