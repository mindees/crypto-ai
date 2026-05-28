"""FastAPI serving-contract tests via TestClient (no TF, no live server)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.serve import api


@pytest.fixture
def client_with_predictions(tmp_path: Path, monkeypatch):
    # Build an isolated repo root with a latest_predictions.json
    (tmp_path / "reports").mkdir()
    (tmp_path / "artifacts" / "production").mkdir(parents=True)
    (tmp_path / "metadata").mkdir()

    predictions = {
        "generated_at_utc": "2026-05-26T12:00:00Z",
        "model_id": "test_model",
        "predictions": [
            {
                "timestamp_utc": "2026-05-26T12:00:00Z",
                "model_id": "test_model", "asset": "BTCUSDT", "timeframe": "1h",
                "model_outputs": {
                    "direction": {"down": 0.2, "sideways": 0.2, "up": 0.6},
                    "regime": {"predicted": "trending_up", "confidence": 0.6},
                    "cycle": {"predicted": "bull", "confidence": 0.5},
                    "trade_quality": {"probability": 0.62},
                },
                "signal": {"action": "no_trade", "reason": "below threshold"},
                "scorecard": {"trend_direction": "up", "rsi_14": 60.0},
                "risk_warning": "Decision-support only. Not financial advice.",
            },
        ],
    }
    (tmp_path / "reports" / "latest_predictions.json").write_text(
        json.dumps(predictions), encoding="utf-8")
    (tmp_path / "artifacts" / "production" / "current_model.json").write_text(
        json.dumps({"current_model_id": "test_model", "artifact_path": "artifacts/runs/test_model"}),
        encoding="utf-8")
    (tmp_path / "metadata" / "model_registry.json").write_text(
        json.dumps({"schema_version": 1, "models": [{"model_id": "test_model", "status": "production"}]}),
        encoding="utf-8")

    monkeypatch.setattr(api, "repo_root", lambda: tmp_path)
    return TestClient(api.app)


def test_health_returns_ok(client_with_predictions):
    r = client_with_predictions.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["has_latest_predictions"] is True
    assert "Not financial advice" in body["disclaimer"]


def test_model_current(client_with_predictions):
    r = client_with_predictions.get("/model/current")
    assert r.status_code == 200
    assert r.json()["current_model_id"] == "test_model"


def test_registry(client_with_predictions):
    r = client_with_predictions.get("/registry")
    assert r.status_code == 200
    assert r.json()["models"][0]["model_id"] == "test_model"


def test_predict_latest_schema(client_with_predictions):
    r = client_with_predictions.get("/predict/latest")
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    p = body["predictions"][0]
    for key in ("timestamp_utc", "model_id", "asset", "timeframe",
                "model_outputs", "signal", "scorecard", "risk_warning"):
        assert key in p


def test_predict_combo_found_and_not_found(client_with_predictions):
    r = client_with_predictions.get("/predict/BTCUSDT/1h")
    assert r.status_code == 200
    assert r.json()["asset"] == "BTCUSDT"

    r404 = client_with_predictions.get("/predict/BTCUSDT/4h")
    assert r404.status_code == 404


def test_scorecard_endpoint(client_with_predictions):
    r = client_with_predictions.get("/scorecard/BTCUSDT/1h")
    assert r.status_code == 200
    body = r.json()
    assert body["asset"] == "BTCUSDT"
    assert "scorecard" in body


def test_predict_refresh_without_tf(client_with_predictions):
    # Default run_predict=false → just re-reads file, no TF subprocess
    r = client_with_predictions.post("/predict/refresh")
    assert r.status_code == 200
    assert r.json()["refreshed"] is True
    assert r.json()["count"] == 1


def test_health_when_no_predictions(tmp_path: Path, monkeypatch):
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(api, "repo_root", lambda: tmp_path)
    client = TestClient(api.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["has_latest_predictions"] is False
    # latest endpoint still returns valid JSON (empty list)
    r2 = client.get("/predict/latest")
    assert r2.status_code == 200
    assert r2.json()["predictions"] == []
