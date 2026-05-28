"""Alert template + gating tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.serve.alert_templates import (
    AlertGateConfig,
    build_alert_payload,
    render_email,
    render_telegram,
    should_alert,
)


def _pred(action: str, *, up=0.67, tq=0.64) -> dict:
    return {
        "timestamp_utc": "2026-05-26T12:00:00Z",
        "model_id": "m1", "asset": "BTCUSDT", "timeframe": "1h",
        "model_outputs": {
            "direction": {"down": 0.18, "sideways": 1 - up - 0.18, "up": up},
            "regime": {"predicted": "trending_up", "confidence": 0.6},
            "cycle": {"predicted": "bull", "confidence": 0.5},
            "trade_quality": {"probability": tq},
        },
        "signal": {"action": action, "reason": "test"},
        "scorecard": {"trend_direction": "up", "rsi_14": 61.2,
                      "macd_state": "bullish_histogram_rising",
                      "funding_state": "neutral", "fear_greed_state": "greed",
                      "onchain_coverage_score": 0.82},
    }


def test_payload_has_all_required_fields():
    payload = build_alert_payload(prediction=_pred("long_bias"))
    for key in ("alert_type", "timestamp_utc", "model_id", "asset", "timeframe",
                "signal", "direction_confidence", "trade_quality_probability",
                "regime", "cycle_phase", "entry_reference", "stop_reference",
                "tp1_reference", "tp2_reference", "tp3_reference",
                "estimated_rr", "risk_per_trade_pct", "leverage",
                "liquidation_buffer_pct", "scorecard", "warnings", "cooldown_minutes"):
        assert key in payload, f"missing {key}"


def test_telegram_contains_required_summary_fields():
    payload = build_alert_payload(prediction=_pred("long_bias"))
    text = render_telegram(payload)
    for token in ("BTCUSDT", "1h", "LONG_BIAS", "Confidence", "Trade Quality",
                  "Regime", "Cycle", "Scorecard", "Not financial advice"):
        assert token in text


def test_email_includes_subject_body_and_full_json():
    payload = build_alert_payload(prediction=_pred("short_bias", up=0.2))
    payload["signal"] = "short_bias"
    email = render_email(payload)
    assert "subject" in email and "body" in email
    assert "BTCUSDT" in email["subject"]
    assert "Full JSON" in email["body"]


def test_no_alert_for_no_trade():
    payload = build_alert_payload(prediction=_pred("no_trade"))
    send, reason = should_alert(payload, AlertGateConfig())
    assert send is False
    assert "non-alerting" in reason


@pytest.mark.parametrize("bad_signal", ["range_wait", "high_risk", "model_stale", "data_coverage_low"])
def test_no_alert_for_non_actionable_signals(bad_signal):
    payload = build_alert_payload(prediction=_pred(bad_signal))
    send, _ = should_alert(payload, AlertGateConfig())
    assert send is False


def test_alert_sent_when_gates_pass():
    payload = build_alert_payload(prediction=_pred("long_bias", up=0.7, tq=0.7))
    send, reason = should_alert(payload, AlertGateConfig(min_signal_confidence=0.65,
                                                          min_trade_quality_probability=0.60))
    assert send is True
    assert "passed" in reason


def test_alert_blocked_by_low_confidence():
    payload = build_alert_payload(prediction=_pred("long_bias", up=0.55, tq=0.7))
    send, reason = should_alert(payload, AlertGateConfig(min_signal_confidence=0.65))
    assert send is False
    assert "confidence" in reason


def test_alert_blocked_by_low_trade_quality():
    payload = build_alert_payload(prediction=_pred("long_bias", up=0.7, tq=0.4))
    send, reason = should_alert(payload, AlertGateConfig(min_trade_quality_probability=0.60))
    assert send is False
    assert "trade quality" in reason


def test_alert_blocked_by_cooldown():
    payload = build_alert_payload(prediction=_pred("long_bias", up=0.7, tq=0.7))
    now = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)
    recent = now - timedelta(minutes=10)
    send, reason = should_alert(payload, AlertGateConfig(cooldown_minutes=60),
                                last_alert_utc=recent, now=now)
    assert send is False
    assert "cooldown" in reason


def test_alert_blocked_by_stale_model_and_coverage():
    payload = build_alert_payload(prediction=_pred("long_bias", up=0.7, tq=0.7))
    s1, r1 = should_alert(payload, AlertGateConfig(), model_stale=True)
    s2, r2 = should_alert(payload, AlertGateConfig(), data_coverage_ok=False)
    assert s1 is False and "stale" in r1
    assert s2 is False and "coverage" in r2


def test_alerts_disabled_by_default_in_config():
    from src.serve import alerts
    # With an empty cfg, every channel is disabled.
    results = alerts.dispatch(build_alert_payload(prediction=_pred("long_bias")), cfg={})
    assert all(not r.attempted for r in results)
    assert {r.channel for r in results} == {"telegram", "discord", "email"}
