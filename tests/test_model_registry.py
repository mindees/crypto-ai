"""Registry / promotion / rollback tests (no TF model needed — we fake runs)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models import promote, registry, rollback


def _fake_run(root: Path, run_id: str, *, macro_f1: float) -> None:
    run_dir = root / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.keras").write_bytes(b"not-a-real-model")
    (run_dir / "dataset_spec.json").write_text(json.dumps({"seq_len": 32}), encoding="utf-8")
    (run_dir / "training_history.json").write_text(
        json.dumps({"val_direction_direction_macro_f1": [macro_f1],
                    "val_trade_quality_auc": [0.6]}),
        encoding="utf-8",
    )


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text(
        "retraining:\n"
        "  promotion:\n"
        "    require_beats_current_production: true\n"
        "    min_direction_macro_f1_improvement: 0.01\n"
        "    min_backtest_profit_factor: 1.10\n"
        "    max_drawdown_not_worse_by_pct: 10\n"
        "    require_positive_expectancy: true\n",
        encoding="utf-8",
    )
    (tmp_path / "metadata").mkdir()
    (tmp_path / "artifacts" / "production").mkdir(parents=True)
    return tmp_path


def test_register_and_list(tmp_repo: Path):
    _fake_run(tmp_repo, "run_a", macro_f1=0.30)
    rec = registry.register_run("run_a", root=tmp_repo)
    assert rec["status"] == "candidate"
    assert rec["model_id"] == "run_a"
    reg = registry.load_registry(tmp_repo)
    assert len(reg["models"]) == 1


def test_sync_registers_unregistered_runs(tmp_repo: Path):
    _fake_run(tmp_repo, "run_a", macro_f1=0.30)
    _fake_run(tmp_repo, "run_b", macro_f1=0.35)
    newly = registry.sync_runs(tmp_repo)
    assert set(newly) == {"run_a", "run_b"}
    # idempotent
    assert registry.sync_runs(tmp_repo) == []


def test_promote_cold_start(tmp_repo: Path):
    _fake_run(tmp_repo, "run_a", macro_f1=0.30)
    registry.register_run("run_a", root=tmp_repo)
    cand = registry.get_model("run_a", root=tmp_repo)
    cfg = registry.read_json  # noqa — not used; load config via promote
    import yaml
    cfg = yaml.safe_load((tmp_repo / "configs" / "config.yaml").read_text())
    decision = promote.evaluate_promotion(cand, None, cfg=cfg)
    assert decision.approved is True
    promote.apply_promotion(decision, root=tmp_repo)
    prod = registry.get_production(tmp_repo)
    assert prod["model_id"] == "run_a"


def test_promote_requires_improvement_over_production(tmp_repo: Path):
    import yaml
    cfg = yaml.safe_load((tmp_repo / "configs" / "config.yaml").read_text())
    # Production at 0.40, candidate at 0.40 (no improvement) → reject
    _fake_run(tmp_repo, "prod", macro_f1=0.40)
    registry.register_run("prod", root=tmp_repo)
    registry.set_status("prod", "production", root=tmp_repo)
    _fake_run(tmp_repo, "cand", macro_f1=0.40)
    registry.register_run("cand", root=tmp_repo)

    cand = registry.get_model("cand", root=tmp_repo)
    prod = registry.get_production(tmp_repo)
    decision = promote.evaluate_promotion(cand, prod, cfg=cfg)
    assert decision.checks["beats_production_macro_f1"] is False
    assert decision.approved is False


def test_promote_then_rollback(tmp_repo: Path):
    import yaml
    cfg = yaml.safe_load((tmp_repo / "configs" / "config.yaml").read_text())
    _fake_run(tmp_repo, "v1", macro_f1=0.30)
    registry.register_run("v1", root=tmp_repo)
    d1 = promote.evaluate_promotion(registry.get_model("v1", root=tmp_repo), None, cfg=cfg)
    promote.apply_promotion(d1, root=tmp_repo)
    assert registry.get_production(tmp_repo)["model_id"] == "v1"

    _fake_run(tmp_repo, "v2", macro_f1=0.50)
    registry.register_run("v2", root=tmp_repo)
    d2 = promote.evaluate_promotion(
        registry.get_model("v2", root=tmp_repo), registry.get_production(tmp_repo), cfg=cfg)
    promote.apply_promotion(d2, root=tmp_repo)
    assert registry.get_production(tmp_repo)["model_id"] == "v2"
    # v1 archived
    assert registry.get_model("v1", root=tmp_repo)["status"] == "archived"

    # Rollback to v1
    assert rollback.rollback_to("v1", root=tmp_repo) is True
    assert registry.get_production(tmp_repo)["model_id"] == "v1"
    assert registry.get_model("v2", root=tmp_repo)["status"] == "rolled_back"


def test_artifacts_never_overwritten(tmp_repo: Path):
    """Promotion/rollback must not touch the model.keras bytes."""
    _fake_run(tmp_repo, "v1", macro_f1=0.30)
    original = (tmp_repo / "artifacts" / "runs" / "v1" / "model.keras").read_bytes()
    registry.register_run("v1", root=tmp_repo)
    import yaml
    cfg = yaml.safe_load((tmp_repo / "configs" / "config.yaml").read_text())
    d = promote.evaluate_promotion(registry.get_model("v1", root=tmp_repo), None, cfg=cfg)
    promote.apply_promotion(d, root=tmp_repo)
    rollback.rollback_to("v1", root=tmp_repo)
    after = (tmp_repo / "artifacts" / "runs" / "v1" / "model.keras").read_bytes()
    assert original == after


def test_current_model_pointer_updated(tmp_repo: Path):
    import yaml
    cfg = yaml.safe_load((tmp_repo / "configs" / "config.yaml").read_text())
    _fake_run(tmp_repo, "v1", macro_f1=0.30)
    registry.register_run("v1", root=tmp_repo)
    d = promote.evaluate_promotion(registry.get_model("v1", root=tmp_repo), None, cfg=cfg)
    promote.apply_promotion(d, root=tmp_repo)
    ptr = json.loads((tmp_repo / "artifacts" / "production" / "current_model.json").read_text())
    assert ptr["current_model_id"] == "v1"
