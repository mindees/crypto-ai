"""Canonical alert payload + human-readable renderers.

All alert channels (Telegram/Discord/email) share ONE payload schema so the
content is consistent and auditable. ``should_alert`` enforces the spec's
gating rules: only ``long_bias``/``short_bias`` with sufficient confidence +
trade-quality, respecting cooldown, model staleness, and data coverage.

CLI::

    python -m src.serve.alert_templates --sample true
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

ACTIONABLE_SIGNALS = {"long_bias", "short_bias"}
NON_ALERTING_SIGNALS = {"no_trade", "range_wait", "high_risk", "model_stale", "data_coverage_low"}

DEFAULT_WARNINGS = [
    "Decision-support only. Not financial advice.",
    "Validate manually before trading.",
]


@dataclass
class AlertGateConfig:
    min_signal_confidence: float = 0.65
    min_trade_quality_probability: float = 0.60
    cooldown_minutes: int = 60


def build_alert_payload(
    *,
    prediction: dict,
    entry_reference: float = 0.0,
    stop_reference: float = 0.0,
    tp1_reference: float = 0.0,
    tp2_reference: float = 0.0,
    tp3_reference: float = 0.0,
    estimated_rr: float = 2.0,
    risk_per_trade_pct: float = 1.0,
    leverage: int = 1,
    liquidation_buffer_pct: float | None = None,
    cooldown_minutes: int = 60,
) -> dict[str, Any]:
    """Translate a prediction JSON (from predict.py) into the canonical alert payload."""
    mo = prediction.get("model_outputs", {})
    direction = mo.get("direction", {})
    signal = prediction.get("signal", {}).get("action", "no_trade")
    # Direction confidence = max class probability
    direction_confidence = max(direction.values()) if direction else 0.0
    scorecard = prediction.get("scorecard", {})

    return {
        "alert_type": "model_signal",
        "timestamp_utc": prediction.get("timestamp_utc", datetime.now(tz=timezone.utc).isoformat()),
        "model_id": prediction.get("model_id", "unknown"),
        "asset": prediction.get("asset", "unknown"),
        "timeframe": prediction.get("timeframe", "unknown"),
        "signal": signal,
        "direction_confidence": round(float(direction_confidence), 4),
        "trade_quality_probability": round(float(mo.get("trade_quality", {}).get("probability", 0.0)), 4),
        "regime": mo.get("regime", {}).get("predicted", "unknown"),
        "cycle_phase": mo.get("cycle", {}).get("predicted", "unknown"),
        "entry_reference": float(entry_reference),
        "stop_reference": float(stop_reference),
        "tp1_reference": float(tp1_reference),
        "tp2_reference": float(tp2_reference),
        "tp3_reference": float(tp3_reference),
        "estimated_rr": float(estimated_rr),
        "risk_per_trade_pct": float(risk_per_trade_pct),
        "leverage": int(leverage),
        "liquidation_buffer_pct": liquidation_buffer_pct,
        "scorecard": {
            "trend": scorecard.get("trend_direction", "unavailable"),
            "ema_stack": scorecard.get("structure_state", "unavailable"),
            "rsi": scorecard.get("rsi_14", "unavailable"),
            "macd": scorecard.get("macd_state", "unavailable"),
            "funding": scorecard.get("funding_state", "unavailable"),
            "open_interest": scorecard.get("open_interest_change_pct", "unavailable"),
            "fear_greed": scorecard.get("fear_greed_state", "unavailable"),
            "data_coverage_score": scorecard.get("onchain_coverage_score", "unavailable"),
        },
        "warnings": list(DEFAULT_WARNINGS),
        "cooldown_minutes": int(cooldown_minutes),
    }


def should_alert(
    payload: dict, gate: AlertGateConfig, *, model_stale: bool = False,
    data_coverage_ok: bool = True, last_alert_utc: datetime | None = None,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Return (send?, reason). Enforces the spec's alert gates."""
    now = now or datetime.now(tz=timezone.utc)
    signal = payload.get("signal")
    if signal in NON_ALERTING_SIGNALS or signal not in ACTIONABLE_SIGNALS:
        return False, f"signal '{signal}' is non-alerting"
    if model_stale:
        return False, "model is stale"
    if not data_coverage_ok:
        return False, "data coverage below acceptable threshold"
    if payload.get("direction_confidence", 0.0) < gate.min_signal_confidence:
        return False, (
            f"direction confidence {payload.get('direction_confidence')} "
            f"< {gate.min_signal_confidence}"
        )
    if payload.get("trade_quality_probability", 0.0) < gate.min_trade_quality_probability:
        return False, (
            f"trade quality {payload.get('trade_quality_probability')} "
            f"< {gate.min_trade_quality_probability}"
        )
    if last_alert_utc is not None:
        elapsed_min = (now - last_alert_utc).total_seconds() / 60.0
        if elapsed_min < gate.cooldown_minutes:
            return False, f"cooldown active ({elapsed_min:.0f} < {gate.cooldown_minutes} min)"
    return True, "all gates passed"


def render_telegram(payload: dict) -> str:
    sig = payload["signal"].upper()
    emoji = "🟢" if payload["signal"] == "long_bias" else "🔴" if payload["signal"] == "short_bias" else "⚪"
    sc = payload["scorecard"]
    return (
        f"{emoji} {payload['asset']} {payload['timeframe']} — {sig}\n\n"
        f"Model: {payload['model_id']}\n"
        f"Confidence: {payload['direction_confidence'] * 100:.0f}%\n"
        f"Trade Quality: {payload['trade_quality_probability'] * 100:.0f}%\n"
        f"Regime: {payload['regime']}\n"
        f"Cycle: {payload['cycle_phase']}\n\n"
        f"Entry Ref: {payload['entry_reference']}\n"
        f"SL Ref: {payload['stop_reference']}\n"
        f"TP1 / TP2 / TP3: {payload['tp1_reference']} / {payload['tp2_reference']} / {payload['tp3_reference']}\n"
        f"Estimated R:R: 1:{payload['estimated_rr']:.0f}+\n"
        f"Risk: {payload['risk_per_trade_pct']}%\n"
        f"Leverage: {payload['leverage']}x\n\n"
        f"Scorecard:\n"
        f"- Trend: {sc['trend']}\n"
        f"- EMA Stack: {sc['ema_stack']}\n"
        f"- RSI: {sc['rsi']}\n"
        f"- MACD: {sc['macd']}\n"
        f"- Funding/OI: {sc['funding']}\n"
        f"- Fear & Greed: {sc['fear_greed']}\n"
        f"- Data Coverage: {sc['data_coverage_score']}\n\n"
        f"Warnings:\n" + "\n".join(payload["warnings"])
    )


def render_discord(payload: dict) -> str:
    # Discord shares the Telegram layout; webhooks accept the same text content.
    return render_telegram(payload)


def render_email(payload: dict) -> dict[str, str]:
    subject = (
        f"[mindees] {payload['asset']} {payload['timeframe']} "
        f"{payload['signal']} ({payload['direction_confidence'] * 100:.0f}%)"
    )
    body = render_telegram(payload) + "\n\n--- Full JSON ---\n" + json.dumps(payload, indent=2)
    return {"subject": subject, "body": body}


def _sample_prediction() -> dict:
    return {
        "timestamp_utc": "2026-05-26T12:00:00Z",
        "model_id": "sample_model_id",
        "asset": "BTCUSDT",
        "timeframe": "1h",
        "model_outputs": {
            "direction": {"down": 0.18, "sideways": 0.15, "up": 0.67},
            "regime": {"predicted": "trending_up", "confidence": 0.63},
            "cycle": {"predicted": "bull", "confidence": 0.58},
            "trade_quality": {"probability": 0.64},
        },
        "signal": {"action": "long_bias", "reason": "up + quality above thresholds"},
        "scorecard": {
            "trend_direction": "up", "structure_state": "higher_highs_higher_lows",
            "rsi_14": 61.2, "macd_state": "bullish_histogram_rising",
            "funding_state": "slightly_positive", "open_interest_change_pct": 0.03,
            "fear_greed_state": "greed", "onchain_coverage_score": 0.82,
        },
    }


def main(argv=None) -> int:
    # Alert text contains emoji (valid for Telegram/Discord). Ensure the
    # console can print UTF-8 on Windows (default cp1252 would crash).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(prog="python -m src.serve.alert_templates")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    args = p.parse_args(argv)

    pred = _sample_prediction()
    payload = build_alert_payload(
        prediction=pred, entry_reference=68000.0, stop_reference=66500.0,
        tp1_reference=69500.0, tp2_reference=71000.0, tp3_reference=72500.0,
        estimated_rr=2.0, risk_per_trade_pct=1.0, leverage=1,
    )
    gate = AlertGateConfig()
    send, reason = should_alert(payload, gate)

    print("=== Canonical alert payload ===")
    print(json.dumps(payload, indent=2))
    print("\n=== should_alert ===")
    print(f"send: {send}  reason: {reason}")
    print("\n=== Telegram/Discord render ===")
    print(render_telegram(payload))
    print("\n=== Email render ===")
    email = render_email(payload)
    print(f"Subject: {email['subject']}")

    # Verify required fields are present (spec checklist)
    required = ["asset", "timeframe", "signal", "direction_confidence",
                "trade_quality_probability", "model_id", "scorecard", "warnings"]
    missing = [k for k in required if k not in payload]
    if missing:
        print(f"\nMISSING required fields: {missing}")
        return 1
    print("\nall required alert fields present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
