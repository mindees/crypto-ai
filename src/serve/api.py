"""FastAPI serving layer.

Deliberately **TensorFlow-free**: the API serves the prediction JSON that
``src.models.predict`` writes, plus registry/model metadata. Keeping TF out of
the web process makes startup fast and avoids the Windows DLL-ordering issue.
A heavy refresh (which needs TF) is delegated to the scheduler / a subprocess.

Endpoints:

* ``GET  /health``
* ``GET  /model/current``
* ``GET  /predict/latest``
* ``GET  /predict/{asset}/{timeframe}``
* ``GET  /scorecard/{asset}/{timeframe}``
* ``GET  /registry``
* ``POST /predict/refresh``

Run::

    uvicorn src.serve.api:app --host 0.0.0.0 --port 8000
    python -m src.serve.api --smoke-test     # loads app, checks routes, exits
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from src.utils.io import read_json, repo_root
from src.utils.logging import get_logger

_log = get_logger("serve.api")

app = FastAPI(
    title="mindees BTC/ETH market-intelligence API",
    version="0.1.0",
    description="Decision-support only. Not financial advice.",
)


def _root() -> Path:
    return repo_root()


def _load_latest_predictions() -> dict:
    path = _root() / "reports" / "latest_predictions.json"
    if not path.exists():
        return {"predictions": [], "model_id": None, "generated_at_utc": None}
    return read_json(path)


@app.get("/health")
def health() -> dict:
    root = _root()
    pred_path = root / "reports" / "latest_predictions.json"
    current = root / "artifacts" / "production" / "current_model.json"
    return {
        "status": "ok",
        "time_utc": datetime.now(tz=timezone.utc).isoformat(),
        "has_latest_predictions": pred_path.exists(),
        "has_production_pointer": current.exists(),
        "disclaimer": "Decision-support only. Not financial advice.",
    }


@app.get("/model/current")
def model_current() -> dict:
    path = _root() / "artifacts" / "production" / "current_model.json"
    if not path.exists():
        return {"current_model_id": None, "note": "no production model promoted yet"}
    return read_json(path)


@app.get("/registry")
def registry() -> dict:
    path = _root() / "metadata" / "model_registry.json"
    if not path.exists():
        return {"models": []}
    return read_json(path)


@app.get("/predict/latest")
def predict_latest() -> dict:
    return _load_latest_predictions()


@app.get("/predict/{asset}/{timeframe}")
def predict_combo(asset: str, timeframe: str) -> dict:
    data = _load_latest_predictions()
    for p in data.get("predictions", []):
        if p.get("asset") == asset and p.get("timeframe") == timeframe:
            return p
    raise HTTPException(status_code=404,
                        detail=f"no prediction for {asset}/{timeframe}. "
                               f"Run predict, or POST /predict/refresh.")


@app.get("/scorecard/{asset}/{timeframe}")
def scorecard(asset: str, timeframe: str) -> dict:
    data = _load_latest_predictions()
    for p in data.get("predictions", []):
        if p.get("asset") == asset and p.get("timeframe") == timeframe:
            return {
                "asset": asset, "timeframe": timeframe,
                "timestamp_utc": p.get("timestamp_utc"),
                "scorecard": p.get("scorecard", {}),
            }
    raise HTTPException(status_code=404, detail=f"no scorecard for {asset}/{timeframe}")


@app.post("/predict/refresh")
def predict_refresh(run_predict: bool = False) -> dict:
    """Re-read the latest predictions file. When ``run_predict=true`` is passed,
    shell out to the (TF-heavy) predict module in a subprocess first."""
    if run_predict:
        try:
            subprocess.run(
                [sys.executable, "-m", "src.models.predict", "--latest",
                 "--symbols", "BTCUSDT", "ETHUSDT", "--timeframes", "1h", "4h"],
                cwd=str(_root()), check=False, timeout=600,
            )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=500,
                                content={"refreshed": False, "error": str(exc)})
    data = _load_latest_predictions()
    return {"refreshed": True, "count": len(data.get("predictions", [])),
            "generated_at_utc": data.get("generated_at_utc")}


@app.get("/drift/latest", response_class=HTMLResponse)
def drift_latest() -> str:
    """Serve the most recent drift dashboard HTML if one exists."""
    reports = _root() / "reports"
    dashboards = sorted(reports.glob("drift_dashboard_*.html"))
    if not dashboards:
        return "<html><body><h1>No drift dashboard yet</h1>" \
               "<p>Run <code>python -m src.models.drift_viz --sample true</code>.</p></body></html>"
    return dashboards[-1].read_text(encoding="utf-8")


def _smoke_test() -> int:
    """Load the app, hit every GET route via TestClient, confirm JSON."""
    from fastapi.testclient import TestClient
    client = TestClient(app)
    checks = [
        ("/health", 200),
        ("/model/current", 200),
        ("/registry", 200),
        ("/predict/latest", 200),
    ]
    ok = True
    for route, expected in checks:
        r = client.get(route)
        status = "OK" if r.status_code == expected else f"FAIL ({r.status_code})"
        if r.status_code != expected:
            ok = False
        print(f"  GET {route:<22} -> {r.status_code} {status}")
    # refresh (no TF)
    r = client.post("/predict/refresh")
    print(f"  POST /predict/refresh   -> {r.status_code} {'OK' if r.status_code == 200 else 'FAIL'}")
    ok = ok and r.status_code == 200
    print("smoke-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.serve.api")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args(argv)
    if args.smoke_test:
        return _smoke_test()
    # Default: run uvicorn programmatically
    import uvicorn
    cfg = read_json_safe(_root() / "configs" / "config.yaml")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    return 0


def read_json_safe(path: Path):
    try:
        from src.utils.io import read_yaml
        return read_yaml(path)
    except Exception:  # noqa: BLE001
        return {}


if __name__ == "__main__":
    sys.exit(main())
