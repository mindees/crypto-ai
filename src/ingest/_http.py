"""Tiny HTTP helper for Binance public-data downloads.

Why a hand-rolled helper: this module is the only HTTP surface for ingestion
and we want controlled retry / backoff / rate-limit handling without dragging
async machinery into a sync ingestion pipeline.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.utils.logging import get_logger

_log = get_logger("ingest.http")

DEFAULT_USER_AGENT = "mindees-crypto-ai/0.1 (+github.com/mindees)"
DEFAULT_TIMEOUT_SEC = 30


@dataclass(frozen=True)
class HttpStatus:
    code: int
    content_length: int | None
    url: str


def _request(url: str, method: str, timeout: int) -> urllib.request.addinfourl:
    req = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Encoding": "identity"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def head(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    max_retries: int = 5,
    backoff_base: float = 1.5,
) -> HttpStatus:
    """HEAD request. Retries transient network errors. 404 is returned as a
    status (not raised) so the caller can treat it as "absent"."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = _request(url, "HEAD", timeout)
            length_hdr = resp.headers.get("Content-Length")
            return HttpStatus(
                code=resp.status,
                content_length=int(length_hdr) if length_hdr else None,
                url=url,
            )
        except urllib.error.HTTPError as exc:
            return HttpStatus(code=exc.code, content_length=None, url=url)
        except OSError as exc:
            if attempt >= max_retries:
                raise
            sleep_for = backoff_base ** attempt
            _log.warning(
                "HEAD error on %s (%s: %s) — retry %d/%d in %.1fs",
                url, type(exc).__name__, exc, attempt, max_retries, sleep_for,
            )
            time.sleep(sleep_for)


def get_bytes(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    max_retries: int = 5,
    backoff_base: float = 1.5,
) -> bytes:
    """GET request returning raw bytes. Retries on 429/5xx and transient errors.

    Raises urllib.error.HTTPError on definitive non-success (e.g. 404 — caller
    can probe with head() first if missing-is-expected).
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = _request(url, "GET", timeout)
            return resp.read()
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 425, 429, 500, 502, 503, 504}
            if not retryable or attempt >= max_retries:
                raise
            sleep_for = _retry_after(exc) or (backoff_base ** attempt)
            _log.warning(
                "HTTP %s on %s — retry %d/%d in %.1fs",
                exc.code, url, attempt, max_retries, sleep_for,
            )
            time.sleep(sleep_for)
        except OSError as exc:
            # Covers URLError, TimeoutError, ConnectionResetError (WinError
            # 10054 — Binance occasionally closes mid-stream), socket errors.
            # HTTPError is handled above; it's a URLError subclass so order
            # matters.
            if attempt >= max_retries:
                raise
            sleep_for = backoff_base ** attempt
            _log.warning(
                "Network error on %s (%s: %s) — retry %d/%d in %.1fs",
                url, type(exc).__name__, exc, attempt, max_retries, sleep_for,
            )
            time.sleep(sleep_for)


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    raw = exc.headers.get("Retry-After") if exc.headers else None
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
