"""Disabled-by-default alert adapters: Telegram, Discord webhook, email SMTP.

Per the spec, NOTHING sends unless explicitly enabled in
``configs/config.yaml`` (``serving.alerting.enable_*``) AND the relevant
credentials are present in the environment. Each adapter degrades gracefully
and reports why it didn't send.

This module never raises on a missing channel — it returns a per-channel
result so the scheduler can log outcomes.
"""
from __future__ import annotations

import os
import smtplib
import urllib.request
from dataclasses import dataclass
from email.mime.text import MIMEText

from src.serve.alert_templates import render_discord, render_email, render_telegram
from src.utils.io import read_yaml, repo_root
from src.utils.logging import get_logger

_log = get_logger("serve.alerts")


@dataclass
class ChannelResult:
    channel: str
    attempted: bool
    sent: bool
    reason: str


def _alerting_cfg(root=None) -> dict:
    root = root or repo_root()
    try:
        cfg = read_yaml(root / "configs" / "config.yaml")
    except Exception:  # noqa: BLE001
        return {}
    return ((cfg.get("serving") or {}).get("alerting")) or {}


def send_telegram(payload: dict, *, cfg: dict | None = None) -> ChannelResult:
    cfg = cfg if cfg is not None else _alerting_cfg()
    if not cfg.get("enable_telegram"):
        return ChannelResult("telegram", False, False, "disabled in config")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return ChannelResult("telegram", False, False, "TELEGRAM_BOT_TOKEN/CHAT_ID not set")
    text = render_telegram(payload)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        urllib.request.urlopen(req, timeout=15)
        return ChannelResult("telegram", True, True, "sent")
    except Exception as exc:  # noqa: BLE001
        return ChannelResult("telegram", True, False, f"send failed: {exc}")


def send_discord(payload: dict, *, cfg: dict | None = None) -> ChannelResult:
    cfg = cfg if cfg is not None else _alerting_cfg()
    if not cfg.get("enable_discord"):
        return ChannelResult("discord", False, False, "disabled in config")
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return ChannelResult("discord", False, False, "DISCORD_WEBHOOK_URL not set")
    import json as _json
    body = _json.dumps({"content": render_discord(payload)}).encode()
    try:
        req = urllib.request.Request(
            webhook, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=15)
        return ChannelResult("discord", True, True, "sent")
    except Exception as exc:  # noqa: BLE001
        return ChannelResult("discord", True, False, f"send failed: {exc}")


def send_email(payload: dict, *, cfg: dict | None = None) -> ChannelResult:
    cfg = cfg if cfg is not None else _alerting_cfg()
    if not cfg.get("enable_email"):
        return ChannelResult("email", False, False, "disabled in config")
    host = os.environ.get("SMTP_HOST")
    port = os.environ.get("SMTP_PORT", "587")
    user = os.environ.get("SMTP_USER")
    pwd = os.environ.get("SMTP_PASS")
    sender = os.environ.get("SMTP_FROM", user)
    recipient = os.environ.get("SMTP_TO")
    if not all([host, user, pwd, recipient]):
        return ChannelResult("email", False, False, "SMTP_* env vars incomplete")
    email = render_email(payload)
    msg = MIMEText(email["body"])
    msg["Subject"] = email["subject"]
    msg["From"] = sender
    msg["To"] = recipient
    try:
        with smtplib.SMTP(host, int(port), timeout=20) as server:
            server.starttls()
            server.login(user, pwd)
            server.send_message(msg)
        return ChannelResult("email", True, True, "sent")
    except Exception as exc:  # noqa: BLE001
        return ChannelResult("email", True, False, f"send failed: {exc}")


def dispatch(payload: dict, *, cfg: dict | None = None) -> list[ChannelResult]:
    """Attempt all channels; each self-gates on its enable flag + creds."""
    cfg = cfg if cfg is not None else _alerting_cfg()
    return [
        send_telegram(payload, cfg=cfg),
        send_discord(payload, cfg=cfg),
        send_email(payload, cfg=cfg),
    ]


# urllib.parse is needed by send_telegram; import lazily-safe at module level
import urllib.parse  # noqa: E402
