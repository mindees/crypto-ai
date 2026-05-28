"""Binance public-data bulk OHLCV ingestion.

Source: ``data.binance.vision`` (verified URL schema in Phase 1 work). The
project relies on these public archives as the deepest verified free history
per (market, symbol, timeframe). No paid endpoints are used.

URL schemas (verified via HEAD requests at build time):

* Spot monthly:   ``/data/spot/monthly/klines/{SYMBOL}/{TF}/{SYMBOL}-{TF}-{YYYY-MM}.zip``
* Spot daily:     ``/data/spot/daily/klines/{SYMBOL}/{TF}/{SYMBOL}-{TF}-{YYYY-MM-DD}.zip``
* USDT-M futures monthly: ``/data/futures/um/monthly/klines/...``
* USDT-M futures daily:   ``/data/futures/um/daily/klines/...``
* Checksum: ``{url}.CHECKSUM`` with body ``"<sha256>  <filename>"``

CSV schema (no header, 12 columns)::

    open_time, open, high, low, close, volume,
    close_time, quote_asset_volume, number_of_trades,
    taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore

Some newer (post-2025) files ship with a header row. Detected at parse time.

Idempotency:

* Raw zips are cached at ``data/raw/binance/<market>/<symbol>/<tf>/<file>.zip``.
* On re-run, a file is **not** redownloaded when local SHA256 already matches
  the server CHECKSUM (or when the server lacks a CHECKSUM and the local file
  parses cleanly).
* Processed Parquet is regenerated each run from the cache; row content is
  identical across re-runs (verified by ``test_ingest_idempotent.py``).
* Watermarks recorded in ``metadata/watermarks.json``.

CLI::

    python -m src.ingest.binance_bulk \\
        --symbols BTCUSDT ETHUSDT \\
        --market-types spot futures_um \\
        --timeframes 1h 4h 1d \\
        [--start-year 2024] \\
        [--end-date 2026-05-26] \\
        [--latest-only] \\
        [--dry-run false]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import sys
import time
import urllib.error
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingest import _http
from src.utils.io import read_json, repo_root, write_json, write_yaml
from src.utils.logging import get_logger
from src.utils.time import TIMEFRAME_TO_MS

_log = get_logger("ingest.binance_bulk")

BINANCE_BASE = "https://data.binance.vision"

MARKET_PATH: dict[str, str] = {
    "spot": "spot",
    "futures_um": "futures/um",
}

# Lower bound for discovery scans. Binance spot launched mid-2017; USDT-M
# futures launched Sep 2019. We start a little before each to be safe.
DEFAULT_DISCOVERY_FLOOR: dict[str, str] = {
    "spot": "2017-07",
    "futures_um": "2019-09",
}

KLINES_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)

# Binance spot archives use the legacy 12-column names. USDT-M futures archives
# (and some newer spot archives) ship with a header row that uses shorter
# aliases. Map them to the canonical KLINES_COLUMNS names at parse time.
_HEADER_ALIASES: dict[str, str] = {
    "quote_volume": "quote_asset_volume",
    "count": "number_of_trades",
    "taker_buy_volume": "taker_buy_base_asset_volume",
    "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
}

# Numeric columns we coerce to float / int after CSV parse.
_FLOAT_COLS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
)
_INT_COLS = ("open_time", "close_time", "number_of_trades")

# Binance switched klines timestamps from milliseconds to microseconds starting
# with the 2025-01 monthly archive. We normalize everything to milliseconds in
# the processed Parquet so downstream code can assume one unit.
# Any open_time value above this threshold is microseconds; otherwise ms.
# 1e14 ms = year 5138, far beyond any real bar — safe sentinel.
_TS_MS_TO_US_THRESHOLD = 10**14


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

def _market_segment(market_type: str) -> str:
    try:
        return MARKET_PATH[market_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported market_type {market_type!r}; expected one of {list(MARKET_PATH)}"
        ) from exc


def monthly_zip_url(market_type: str, symbol: str, timeframe: str, year: int, month: int) -> str:
    seg = _market_segment(market_type)
    return (
        f"{BINANCE_BASE}/data/{seg}/monthly/klines/{symbol}/{timeframe}/"
        f"{symbol}-{timeframe}-{year:04d}-{month:02d}.zip"
    )


def daily_zip_url(
    market_type: str, symbol: str, timeframe: str, year: int, month: int, day: int
) -> str:
    seg = _market_segment(market_type)
    return (
        f"{BINANCE_BASE}/data/{seg}/daily/klines/{symbol}/{timeframe}/"
        f"{symbol}-{timeframe}-{year:04d}-{month:02d}-{day:02d}.zip"
    )


def checksum_url(zip_url: str) -> str:
    return zip_url + ".CHECKSUM"


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

def _parse_year_month(s: str) -> tuple[int, int]:
    """'2024-05' -> (2024, 5). Strict."""
    y_str, m_str = s.split("-")
    return int(y_str), int(m_str)


def _month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _today_utc() -> date:
    return datetime.now(tz=timezone.utc).date()


def iter_months(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> Iterator[tuple[int, int]]:
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        y, m = _next_month(y, m)


def iter_days_in_month(year: int, month: int, *, until: date | None = None) -> Iterator[date]:
    d = date(year, month, 1)
    while d.month == month:
        if until is not None and d > until:
            return
        yield d
        d = date.fromordinal(d.toordinal() + 1)


# ---------------------------------------------------------------------------
# Cache + checksum
# ---------------------------------------------------------------------------

def cache_zip_path(
    market_type: str, symbol: str, timeframe: str, filename: str, *, root: Path | None = None
) -> Path:
    root = root or repo_root()
    return root / "data" / "raw" / "binance" / market_type / symbol / timeframe / filename


def parse_checksum_body(body: str) -> str:
    """Body is '<sha256>  <filename>'. Return the sha256 hex digest."""
    stripped = body.strip()
    if not stripped:
        raise ValueError("Empty checksum body")
    return stripped.split()[0].lower()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        fh.write(data)
    tmp.replace(path)


@dataclass
class FileFetchResult:
    url: str
    cache_path: Path
    bytes_count: int
    sha256: str
    checksum_status: str  # "verified" | "absent_on_server" | "skipped_cached"
    from_cache: bool


def fetch_zip_with_checksum(
    zip_url: str,
    cache_path: Path,
    *,
    head_fn: Callable[[str], _http.HttpStatus] = _http.head,
    get_fn: Callable[[str], bytes] = _http.get_bytes,
) -> FileFetchResult | None:
    """Fetch a zip, verifying against its .CHECKSUM if present.

    Returns ``None`` when the zip is absent on the server (404) — the caller
    treats that as "no data for this period" (not an error).

    Idempotent: a cached file with matching SHA256 short-circuits the network.
    """
    expected_sha: str | None = None
    cksum_status_for_cached = "skipped_cached"

    cksum_url = checksum_url(zip_url)
    try:
        cksum_status = head_fn(cksum_url)
    except OSError as exc:
        _log.warning("HEAD checksum %s failed (%s) — proceeding without verification", cksum_url, exc)
        cksum_status = None
    if cksum_status is not None and cksum_status.code == 200:
        try:
            cksum_body = get_fn(cksum_url).decode("utf-8")
            expected_sha = parse_checksum_body(cksum_body)
            cksum_status_for_cached = "verified"
        except (urllib.error.HTTPError, ValueError, OSError) as exc:
            _log.warning(
                "Could not read checksum for %s (%s) — proceeding without verification",
                zip_url, exc,
            )

    if cache_path.exists():
        cached_bytes = cache_path.read_bytes()
        cached_sha = hashlib.sha256(cached_bytes).hexdigest()
        if expected_sha is None or cached_sha == expected_sha:
            return FileFetchResult(
                url=zip_url,
                cache_path=cache_path,
                bytes_count=len(cached_bytes),
                sha256=cached_sha,
                checksum_status=cksum_status_for_cached if expected_sha else "absent_on_server",
                from_cache=True,
            )
        _log.warning("Cached %s sha256 mismatch — re-downloading", cache_path.name)

    zip_status = head_fn(zip_url)
    if zip_status.code == 404:
        return None
    if zip_status.code >= 400 and zip_status.code != 200:
        _log.warning("HEAD %s returned %s — skipping", zip_url, zip_status.code)
        return None

    try:
        body = get_fn(zip_url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise

    sha = hashlib.sha256(body).hexdigest()
    if expected_sha is not None and sha != expected_sha:
        raise ValueError(
            f"Downloaded {zip_url} sha256 {sha} does not match checksum {expected_sha}"
        )

    _atomic_write_bytes(cache_path, body)
    return FileFetchResult(
        url=zip_url,
        cache_path=cache_path,
        bytes_count=len(body),
        sha256=sha,
        checksum_status=("verified" if expected_sha is not None else "absent_on_server"),
        from_cache=False,
    )


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _starts_with_digit(b: bytes) -> bool:
    return bool(b) and b[:1].isdigit()


def parse_zip_to_dataframe(zip_bytes: bytes, *, source_filename: str) -> pd.DataFrame:
    """Unzip and parse a single Binance klines monthly/daily archive.

    Handles both the no-header legacy format and the header-row format some
    post-2025 files use. Coerces ``open_time``/``close_time`` to integer
    milliseconds; numeric columns to float; ``number_of_trades`` to int.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(
                f"Expected exactly one CSV member in {source_filename}, found {members}"
            )
        csv_bytes = zf.read(members[0])

    stripped = csv_bytes.lstrip(b"\xef\xbb\xbf \n\r\t")
    has_header = not _starts_with_digit(stripped)

    df = pd.read_csv(
        io.BytesIO(csv_bytes),
        header=0 if has_header else None,
        names=None if has_header else list(KLINES_COLUMNS),
    )
    if has_header:
        normalized = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
        df = df.rename(columns=normalized)
        df = df.rename(columns=_HEADER_ALIASES)
        missing = [c for c in KLINES_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{source_filename}: header missing columns {missing}")
        df = df[list(KLINES_COLUMNS)]

    for col in _FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
    for col in _INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")

    # Normalize timestamps to milliseconds (handles Binance's 2025-01 ms->us switch).
    if len(df) and df["open_time"].iloc[0] > _TS_MS_TO_US_THRESHOLD:
        df["open_time"] = df["open_time"] // 1000
        df["close_time"] = df["close_time"] // 1000
    return df


# ---------------------------------------------------------------------------
# Per-combo ingestion
# ---------------------------------------------------------------------------

@dataclass
class CombinationResult:
    market_type: str
    symbol: str
    timeframe: str
    files_processed: int = 0
    files_from_cache: int = 0
    files_downloaded: int = 0
    bytes_downloaded: int = 0
    rows_total: int = 0
    duplicates_dropped: int = 0
    first_open_time_utc: str | None = None
    last_open_time_utc: str | None = None
    missing_candle_estimate: int = 0
    checksum_verified: int = 0
    checksum_absent: int = 0
    parquet_path: str | None = None
    skipped_periods_404: int = 0
    errors: list[str] = field(default_factory=list)


def _processed_parquet_path(
    market_type: str, symbol: str, timeframe: str, *, root: Path | None = None
) -> Path:
    root = root or repo_root()
    return (
        root
        / "data"
        / "processed"
        / "ohlcv"
        / f"source=binance"
        / f"market_type={market_type}"
        / f"symbol={symbol}"
        / f"timeframe={timeframe}"
        / "data.parquet"
    )


def _missing_candle_estimate(open_times_ms: pd.Series, timeframe: str) -> int:
    if len(open_times_ms) < 2:
        return 0
    step = TIMEFRAME_TO_MS[timeframe]
    expected = (open_times_ms.iloc[-1] - open_times_ms.iloc[0]) // step + 1
    return int(expected - len(open_times_ms))


def ingest_combo(
    market_type: str,
    symbol: str,
    timeframe: str,
    *,
    start_year_month: tuple[int, int] | None = None,
    end_date: date | None = None,
    dry_run: bool = False,
    head_fn: Callable[[str], _http.HttpStatus] = _http.head,
    get_fn: Callable[[str], bytes] = _http.get_bytes,
    root: Path | None = None,
) -> CombinationResult:
    """Ingest all monthly archives for one (market, symbol, timeframe).

    For the current month we also probe daily archives that postdate the most
    recent monthly file.
    """
    result = CombinationResult(market_type=market_type, symbol=symbol, timeframe=timeframe)
    today = end_date or _today_utc()
    start_y, start_m = start_year_month or _parse_year_month(
        DEFAULT_DISCOVERY_FLOOR[market_type]
    )

    if (start_y, start_m) > (today.year, today.month):
        return result

    frames: list[pd.DataFrame] = []
    last_monthly_year_month: tuple[int, int] | None = None

    # Monthly files: from start through previous month (current month often has
    # no monthly yet — it gets published shortly after month end).
    monthly_end_year, monthly_end_month = (
        (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
    )

    for y, m in iter_months(start_y, start_m, monthly_end_year, monthly_end_month):
        url = monthly_zip_url(market_type, symbol, timeframe, y, m)
        cache_path = cache_zip_path(
            market_type, symbol, timeframe,
            f"{symbol}-{timeframe}-{y:04d}-{m:02d}.zip",
            root=root,
        )
        if dry_run:
            status = head_fn(url)
            if status.code == 200:
                result.files_processed += 1
                last_monthly_year_month = (y, m)
            elif status.code == 404:
                result.skipped_periods_404 += 1
            else:
                result.errors.append(f"HEAD {url} -> {status.code}")
            continue

        try:
            fetched = fetch_zip_with_checksum(url, cache_path, head_fn=head_fn, get_fn=get_fn)
        except Exception as exc:  # noqa: BLE001 — report & continue, don't abort whole run
            result.errors.append(f"fetch {url}: {exc}")
            continue

        if fetched is None:
            result.skipped_periods_404 += 1
            continue

        result.files_processed += 1
        if fetched.from_cache:
            result.files_from_cache += 1
        else:
            result.files_downloaded += 1
            result.bytes_downloaded += fetched.bytes_count
        if fetched.checksum_status == "verified":
            result.checksum_verified += 1
        elif fetched.checksum_status == "absent_on_server":
            result.checksum_absent += 1

        try:
            df = parse_zip_to_dataframe(fetched.cache_path.read_bytes(), source_filename=cache_path.name)
            frames.append(df)
            last_monthly_year_month = (y, m)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"parse {cache_path.name}: {exc}")

    # Daily fill: cover the current month (and any days of the previous month
    # that didn't make it into a monthly yet).
    daily_start = (
        date(*_next_month(*last_monthly_year_month), day=1)
        if last_monthly_year_month
        else date(start_y, start_m, 1)
    )
    if daily_start <= today:
        d = daily_start
        # Bound the daily probe at "today minus 1" — today's bar isn't usually
        # published as a daily archive yet.
        last_daily = date.fromordinal(today.toordinal() - 1)
        while d <= last_daily:
            url = daily_zip_url(market_type, symbol, timeframe, d.year, d.month, d.day)
            cache_path = cache_zip_path(
                market_type, symbol, timeframe,
                f"{symbol}-{timeframe}-{d.year:04d}-{d.month:02d}-{d.day:02d}.zip",
                root=root,
            )
            if dry_run:
                status = head_fn(url)
                if status.code == 200:
                    result.files_processed += 1
                elif status.code == 404:
                    result.skipped_periods_404 += 1
                d = date.fromordinal(d.toordinal() + 1)
                continue

            try:
                fetched = fetch_zip_with_checksum(url, cache_path, head_fn=head_fn, get_fn=get_fn)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"fetch {url}: {exc}")
                d = date.fromordinal(d.toordinal() + 1)
                continue

            if fetched is None:
                result.skipped_periods_404 += 1
            else:
                result.files_processed += 1
                if fetched.from_cache:
                    result.files_from_cache += 1
                else:
                    result.files_downloaded += 1
                    result.bytes_downloaded += fetched.bytes_count
                if fetched.checksum_status == "verified":
                    result.checksum_verified += 1
                elif fetched.checksum_status == "absent_on_server":
                    result.checksum_absent += 1

                try:
                    df = parse_zip_to_dataframe(
                        fetched.cache_path.read_bytes(), source_filename=cache_path.name
                    )
                    frames.append(df)
                except Exception as exc:  # noqa: BLE001
                    result.errors.append(f"parse {cache_path.name}: {exc}")
            d = date.fromordinal(d.toordinal() + 1)

    if dry_run or not frames:
        return result

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["open_time"], keep="last")
    after = len(merged)
    merged = merged.sort_values("open_time").reset_index(drop=True)

    result.rows_total = after
    result.duplicates_dropped = before - after
    result.first_open_time_utc = pd.Timestamp(merged["open_time"].iloc[0], unit="ms", tz="UTC").isoformat()
    result.last_open_time_utc = pd.Timestamp(merged["open_time"].iloc[-1], unit="ms", tz="UTC").isoformat()
    result.missing_candle_estimate = _missing_candle_estimate(merged["open_time"], timeframe)

    out_path = _processed_parquet_path(market_type, symbol, timeframe, root=root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(merged, preserve_index=False)
    tmp_out = out_path.with_suffix(out_path.suffix + ".part")
    pq.write_table(table, tmp_out, compression="snappy")
    tmp_out.replace(out_path)
    result.parquet_path = str(out_path.relative_to((root or repo_root())))

    return result


# ---------------------------------------------------------------------------
# Top-level run + metadata writers
# ---------------------------------------------------------------------------

def _write_watermarks(results: Sequence[CombinationResult], *, root: Path | None = None) -> Path:
    root = root or repo_root()
    path = root / "metadata" / "watermarks.json"
    payload = read_json(path) if path.exists() else {"schema_version": 1, "watermarks": {}}
    payload.setdefault("watermarks", {})
    payload["last_updated_utc"] = datetime.now(tz=timezone.utc).isoformat()
    for r in results:
        if r.rows_total == 0:
            continue
        key = f"binance/{r.market_type}/{r.symbol}/{r.timeframe}"
        payload["watermarks"][key] = {
            "first_open_time_utc": r.first_open_time_utc,
            "last_open_time_utc": r.last_open_time_utc,
            "rows": r.rows_total,
        }
    write_json(path, payload)
    return path


def _write_source_registry(results: Sequence[CombinationResult], *, root: Path | None = None) -> Path:
    root = root or repo_root()
    path = root / "metadata" / "source_registry.yaml"
    entries: list[dict] = []
    for r in results:
        if r.rows_total == 0 and r.files_processed == 0:
            continue
        entries.append({
            "source": "binance_public_data",
            "market_type": r.market_type,
            "symbol": r.symbol,
            "timeframe": r.timeframe,
            "first_open_time_utc": r.first_open_time_utc,
            "last_open_time_utc": r.last_open_time_utc,
            "rows": r.rows_total,
            "files_processed": r.files_processed,
            "checksum_verified": r.checksum_verified,
            "checksum_absent": r.checksum_absent,
            "missing_candle_estimate": r.missing_candle_estimate,
        })
    write_yaml(path, {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "sources": entries,
    })
    return path


def _write_coverage_report(results: Sequence[CombinationResult], *, root: Path | None = None) -> Path:
    root = root or repo_root()
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / "reports" / f"binance_ingest_{stamp}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Binance bulk ingestion report — {stamp}",
        "",
        "| Market | Symbol | TF | Rows | First | Last | Missing est. | Files | Cached | DL'd | DL MB | CS verified | CS absent | 404s | Errors |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.market_type} | {r.symbol} | {r.timeframe} | {r.rows_total} | "
            f"{r.first_open_time_utc or '—'} | {r.last_open_time_utc or '—'} | "
            f"{r.missing_candle_estimate} | {r.files_processed} | {r.files_from_cache} | "
            f"{r.files_downloaded} | {r.bytes_downloaded / (1024*1024):.2f} | "
            f"{r.checksum_verified} | {r.checksum_absent} | {r.skipped_periods_404} | "
            f"{len(r.errors)} |"
        )
    if any(r.errors for r in results):
        lines += ["", "## Errors", ""]
        for r in results:
            for e in r.errors:
                lines.append(f"- `{r.market_type}/{r.symbol}/{r.timeframe}`: {e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(
    symbols: Iterable[str],
    market_types: Iterable[str],
    timeframes: Iterable[str],
    *,
    start_year: int | None = None,
    end_date: date | None = None,
    dry_run: bool = False,
    head_fn: Callable[[str], _http.HttpStatus] = _http.head,
    get_fn: Callable[[str], bytes] = _http.get_bytes,
    root: Path | None = None,
) -> list[CombinationResult]:
    results: list[CombinationResult] = []
    t0 = time.perf_counter()

    for market_type in market_types:
        if market_type not in MARKET_PATH:
            raise ValueError(f"Unsupported market_type {market_type!r}")
        for symbol in symbols:
            for tf in timeframes:
                if tf not in TIMEFRAME_TO_MS:
                    raise ValueError(f"Unsupported timeframe {tf!r}")
                floor_y, floor_m = _parse_year_month(DEFAULT_DISCOVERY_FLOOR[market_type])
                if start_year is not None:
                    start_ym = (max(start_year, floor_y), 1 if start_year > floor_y else floor_m)
                else:
                    start_ym = (floor_y, floor_m)

                _log.info("ingest %s/%s/%s starting at %s", market_type, symbol, tf, start_ym)
                r = ingest_combo(
                    market_type, symbol, tf,
                    start_year_month=start_ym,
                    end_date=end_date,
                    dry_run=dry_run,
                    head_fn=head_fn, get_fn=get_fn,
                    root=root,
                )
                results.append(r)
                _log.info(
                    "done %s/%s/%s: rows=%d files=%d downloaded=%d cached=%d 404s=%d errors=%d",
                    market_type, symbol, tf,
                    r.rows_total, r.files_processed, r.files_downloaded,
                    r.files_from_cache, r.skipped_periods_404, len(r.errors),
                )

    if not dry_run:
        _write_watermarks(results, root=root)
        _write_source_registry(results, root=root)
        report = _write_coverage_report(results, root=root)
        _log.info("coverage report: %s", report)

    elapsed = time.perf_counter() - t0
    _log.info("ingest run finished in %.1fs (%d combos)", elapsed, len(results))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y", "t"}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingest.binance_bulk",
        description="Bulk OHLCV ingestion from data.binance.vision.",
    )
    p.add_argument("--symbols", nargs="+", required=True, help="e.g. BTCUSDT ETHUSDT")
    p.add_argument(
        "--market-types",
        nargs="+",
        required=True,
        choices=sorted(MARKET_PATH.keys()),
        help="spot and/or futures_um",
    )
    p.add_argument(
        "--timeframes",
        nargs="+",
        required=True,
        help="any of " + ", ".join(sorted(TIMEFRAME_TO_MS.keys())),
    )
    p.add_argument("--start-year", type=int, default=None,
                   help="Limit to this year and after (default: source's earliest)")
    p.add_argument("--end-date", type=str, default=None,
                   help="YYYY-MM-DD upper bound (default: today UTC)")
    p.add_argument("--dry-run", type=_parse_bool, default=False,
                   help="HEAD-probe only; do not download or write Parquet")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None
    )
    results = run(
        symbols=args.symbols,
        market_types=args.market_types,
        timeframes=args.timeframes,
        start_year=args.start_year,
        end_date=end_date,
        dry_run=args.dry_run,
    )

    print("\n=== Binance bulk ingestion summary ===")
    print(
        f"{'market':<12} {'symbol':<9} {'tf':<4} "
        f"{'rows':>10} {'first':<26} {'last':<26} "
        f"{'miss':>6} {'files':>5} {'cache':>5} {'dl':>4} {'404':>4} {'err':>3}"
    )
    for r in results:
        print(
            f"{r.market_type:<12} {r.symbol:<9} {r.timeframe:<4} "
            f"{r.rows_total:>10} {(r.first_open_time_utc or '-'):<26} "
            f"{(r.last_open_time_utc or '-'):<26} "
            f"{r.missing_candle_estimate:>6} {r.files_processed:>5} "
            f"{r.files_from_cache:>5} {r.files_downloaded:>4} "
            f"{r.skipped_periods_404:>4} {len(r.errors):>3}"
        )
    return 0 if not any(r.errors for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
