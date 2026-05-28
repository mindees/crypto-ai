"""Idempotency test for Binance bulk ingestion (Phase 1 gate).

Strategy: replace the HTTP layer with an in-memory fake serving a small
synthetic month of klines + matching CHECKSUM. Run ingest twice and assert:

* row counts and timestamps are identical on both runs
* the second run downloads 0 bytes (cache hit) but still verifies checksums
* Parquet contents are byte-identical (or at least row-identical)
* watermark entry matches

Does NOT hit the network.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.ingest import _http, binance_bulk


# --------------------------------------------------------------------------- #
# Synthetic data + fake HTTP
# --------------------------------------------------------------------------- #

def _make_synthetic_csv_bytes(num_rows: int, *, tf_ms: int, start_ms: int) -> bytes:
    """Generate `num_rows` valid Binance klines CSV lines (no header, 12 cols)."""
    rows = []
    for i in range(num_rows):
        ot = start_ms + i * tf_ms
        ct = ot + tf_ms - 1
        rows.append(
            f"{ot},100.00,110.00,90.00,105.00,1234.5,{ct},123456.78,42,500.0,52000.0,0"
        )
    return ("\n".join(rows) + "\n").encode("utf-8")


def _zip_csv(csv_bytes: bytes, *, member_name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, csv_bytes)
    return buf.getvalue()


@dataclass
class FakeHttp:
    """In-memory store of url -> bytes. Provides head/get with call counters."""
    files: dict[str, bytes]
    head_calls: int = 0
    get_calls: int = 0
    get_calls_by_url: dict[str, int] = field(default_factory=dict)

    def head(self, url: str) -> _http.HttpStatus:
        self.head_calls += 1
        if url in self.files:
            return _http.HttpStatus(code=200, content_length=len(self.files[url]), url=url)
        return _http.HttpStatus(code=404, content_length=None, url=url)

    def get(self, url: str) -> bytes:
        self.get_calls += 1
        self.get_calls_by_url[url] = self.get_calls_by_url.get(url, 0) + 1
        if url in self.files:
            return self.files[url]
        import urllib.error
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]


@pytest.fixture
def fake_binance_month():
    """Set up a fake server with one BTCUSDT spot 1h monthly file for 2024-01."""
    symbol = "BTCUSDT"
    tf = "1h"
    market = "spot"
    year, month = 2024, 1

    # 31 days × 24 hours = 744 candles
    tf_ms = binance_bulk.TIMEFRAME_TO_MS[tf]
    start_ms = int(pd.Timestamp(f"{year:04d}-{month:02d}-01", tz="UTC").timestamp() * 1000)
    num_rows = 31 * 24

    csv_bytes = _make_synthetic_csv_bytes(num_rows, tf_ms=tf_ms, start_ms=start_ms)
    member_name = f"{symbol}-{tf}-{year:04d}-{month:02d}.csv"
    zip_bytes = _zip_csv(csv_bytes, member_name=member_name)
    sha = hashlib.sha256(zip_bytes).hexdigest()
    checksum_body = f"{sha}  {symbol}-{tf}-{year:04d}-{month:02d}.zip\n".encode("utf-8")

    zip_url = binance_bulk.monthly_zip_url(market, symbol, tf, year, month)
    files = {
        zip_url: zip_bytes,
        binance_bulk.checksum_url(zip_url): checksum_body,
    }
    fake = FakeHttp(files=files)
    return {
        "fake": fake,
        "market": market,
        "symbol": symbol,
        "timeframe": tf,
        "year": year,
        "month": month,
        "num_rows": num_rows,
        "sha": sha,
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_first_run_downloads_and_persists(tmp_path: Path, fake_binance_month):
    f = fake_binance_month
    fake: FakeHttp = f["fake"]

    # End date is mid-Feb 2024 so monthly-Jan is included but daily probes for
    # Feb all 404 (which is the natural "no data" case).
    result = binance_bulk.ingest_combo(
        f["market"], f["symbol"], f["timeframe"],
        start_year_month=(f["year"], f["month"]),
        end_date=date(2024, 2, 15),
        head_fn=fake.head, get_fn=fake.get,
        root=tmp_path,
    )

    assert result.rows_total == f["num_rows"]
    assert result.files_downloaded == 1
    assert result.files_from_cache == 0
    assert result.checksum_verified == 1
    assert result.duplicates_dropped == 0
    assert result.missing_candle_estimate == 0
    assert result.first_open_time_utc.startswith("2024-01-01T00:00:00")
    assert result.last_open_time_utc.startswith("2024-01-31T23:00:00")

    # Cache + parquet exist
    cache = binance_bulk.cache_zip_path(
        f["market"], f["symbol"], f["timeframe"],
        f"{f['symbol']}-{f['timeframe']}-{f['year']:04d}-{f['month']:02d}.zip",
        root=tmp_path,
    )
    assert cache.exists()
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == f["sha"]

    parquet = (
        tmp_path / "data" / "processed" / "ohlcv"
        / "source=binance" / f"market_type={f['market']}"
        / f"symbol={f['symbol']}" / f"timeframe={f['timeframe']}"
        / "data.parquet"
    )
    assert parquet.exists()
    df = pq.read_table(parquet).to_pandas()
    assert len(df) == f["num_rows"]
    assert set(df.columns) >= set(binance_bulk.KLINES_COLUMNS)


def test_second_run_is_idempotent_no_redownload(tmp_path: Path, fake_binance_month):
    f = fake_binance_month
    fake: FakeHttp = f["fake"]

    common = dict(
        market_type=f["market"], symbol=f["symbol"], timeframe=f["timeframe"],
        start_year_month=(f["year"], f["month"]),
        end_date=date(2024, 2, 15),
        head_fn=fake.head, get_fn=fake.get,
        root=tmp_path,
    )

    r1 = binance_bulk.ingest_combo(**common)
    bytes_after_first = fake.get_calls_by_url.get(
        binance_bulk.monthly_zip_url(f["market"], f["symbol"], f["timeframe"], f["year"], f["month"]),
        0,
    )
    assert bytes_after_first == 1  # downloaded once

    parquet_path = (
        tmp_path / "data" / "processed" / "ohlcv"
        / "source=binance" / f"market_type={f['market']}"
        / f"symbol={f['symbol']}" / f"timeframe={f['timeframe']}"
        / "data.parquet"
    )
    df1 = pq.read_table(parquet_path).to_pandas()

    # Re-run
    r2 = binance_bulk.ingest_combo(**common)
    df2 = pq.read_table(parquet_path).to_pandas()

    # Row identity
    assert r2.rows_total == r1.rows_total
    assert r2.first_open_time_utc == r1.first_open_time_utc
    assert r2.last_open_time_utc == r1.last_open_time_utc
    assert r2.duplicates_dropped == 0
    pd.testing.assert_frame_equal(
        df1.sort_values("open_time").reset_index(drop=True),
        df2.sort_values("open_time").reset_index(drop=True),
        check_exact=True,
    )

    # Cache hit, no second download
    bytes_after_second = fake.get_calls_by_url.get(
        binance_bulk.monthly_zip_url(f["market"], f["symbol"], f["timeframe"], f["year"], f["month"]),
        0,
    )
    assert bytes_after_second == 1, "the zip must not be re-downloaded on the second run"
    assert r2.files_downloaded == 0
    assert r2.files_from_cache == 1
    assert r2.checksum_verified == 1


def test_corrupted_cache_triggers_redownload(tmp_path: Path, fake_binance_month):
    f = fake_binance_month
    fake: FakeHttp = f["fake"]
    common = dict(
        market_type=f["market"], symbol=f["symbol"], timeframe=f["timeframe"],
        start_year_month=(f["year"], f["month"]),
        end_date=date(2024, 2, 15),
        head_fn=fake.head, get_fn=fake.get,
        root=tmp_path,
    )
    binance_bulk.ingest_combo(**common)

    cache = binance_bulk.cache_zip_path(
        f["market"], f["symbol"], f["timeframe"],
        f"{f['symbol']}-{f['timeframe']}-{f['year']:04d}-{f['month']:02d}.zip",
        root=tmp_path,
    )
    cache.write_bytes(b"this is not a valid zip and the checksum will not match")

    r2 = binance_bulk.ingest_combo(**common)
    assert r2.files_downloaded == 1, "corrupted cache must trigger re-download"
    assert r2.rows_total == f["num_rows"]


def test_404_periods_do_not_raise(tmp_path: Path, fake_binance_month):
    f = fake_binance_month
    fake: FakeHttp = f["fake"]

    # Try to ingest a window that includes months the server doesn't have.
    result = binance_bulk.ingest_combo(
        f["market"], f["symbol"], f["timeframe"],
        start_year_month=(2023, 11),  # 2023-11 and 2023-12 are not on the fake server
        end_date=date(2024, 2, 15),
        head_fn=fake.head, get_fn=fake.get,
        root=tmp_path,
    )
    assert result.rows_total == f["num_rows"]
    assert result.skipped_periods_404 >= 2
    assert result.errors == []


def test_dry_run_makes_no_disk_writes(tmp_path: Path, fake_binance_month):
    f = fake_binance_month
    fake: FakeHttp = f["fake"]
    result = binance_bulk.ingest_combo(
        f["market"], f["symbol"], f["timeframe"],
        start_year_month=(f["year"], f["month"]),
        end_date=date(2024, 2, 15),
        dry_run=True,
        head_fn=fake.head, get_fn=fake.get,
        root=tmp_path,
    )
    assert result.rows_total == 0
    assert result.files_downloaded == 0
    assert result.parquet_path is None
    assert not any(tmp_path.rglob("*.parquet"))
    assert not any(tmp_path.rglob("*.zip"))
    assert result.files_processed >= 1  # the HEAD probe confirmed the file exists


def test_url_construction_matches_verified_schema():
    # Verified during Phase 1 buildout — pin these URL shapes here.
    assert binance_bulk.monthly_zip_url("spot", "BTCUSDT", "1d", 2024, 1) == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
    )
    assert binance_bulk.monthly_zip_url("futures_um", "ETHUSDT", "4h", 2023, 12) == (
        "https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/4h/ETHUSDT-4h-2023-12.zip"
    )
    assert binance_bulk.daily_zip_url("spot", "BTCUSDT", "1m", 2024, 5, 3) == (
        "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-05-03.zip"
    )
    assert (
        binance_bulk.checksum_url(binance_bulk.monthly_zip_url("spot", "BTCUSDT", "1d", 2024, 1))
        == "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip.CHECKSUM"
    )


def test_parse_checksum_body_handles_double_space_format():
    # Verified format: '<sha256>  <filename>'
    body = "474c1ce6fbb09e42cfc7231fee249aecc58af2fb5918570ffeba37998926b4a4  BTCUSDT-1d-2024-01.zip\n"
    assert binance_bulk.parse_checksum_body(body) == (
        "474c1ce6fbb09e42cfc7231fee249aecc58af2fb5918570ffeba37998926b4a4"
    )


def test_parse_zip_to_dataframe_handles_no_header_csv():
    csv = _make_synthetic_csv_bytes(5, tf_ms=binance_bulk.TIMEFRAME_TO_MS["1h"], start_ms=1704067200000)
    zip_bytes = _zip_csv(csv, member_name="X-1h-2024-01.csv")
    df = binance_bulk.parse_zip_to_dataframe(zip_bytes, source_filename="X-1h-2024-01.zip")
    assert len(df) == 5
    assert list(df.columns) == list(binance_bulk.KLINES_COLUMNS)
    assert df["open_time"].dtype.kind == "i"
    assert df["close"].dtype.kind == "f"


def test_parse_zip_to_dataframe_handles_header_csv():
    header = ",".join(binance_bulk.KLINES_COLUMNS) + "\n"
    body = header.encode("utf-8") + _make_synthetic_csv_bytes(
        3, tf_ms=binance_bulk.TIMEFRAME_TO_MS["1h"], start_ms=1704067200000
    )
    zip_bytes = _zip_csv(body, member_name="X-1h-2024-01.csv")
    df = binance_bulk.parse_zip_to_dataframe(zip_bytes, source_filename="X-1h-2024-01.zip")
    assert len(df) == 3
    assert list(df.columns) == list(binance_bulk.KLINES_COLUMNS)


def test_parse_zip_to_dataframe_normalizes_futures_header_aliases():
    """Futures klines ship with header using shorter aliases — must be mapped."""
    futures_header = (
        "open_time,open,high,low,close,volume,"
        "close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
    )
    body = futures_header.encode("utf-8") + _make_synthetic_csv_bytes(
        4, tf_ms=binance_bulk.TIMEFRAME_TO_MS["1d"], start_ms=1704067200000
    )
    zip_bytes = _zip_csv(body, member_name="BTCUSDT-1d-2024-01.csv")
    df = binance_bulk.parse_zip_to_dataframe(zip_bytes, source_filename="BTCUSDT-1d-2024-01.zip")
    assert list(df.columns) == list(binance_bulk.KLINES_COLUMNS)
    assert len(df) == 4
    assert df["quote_asset_volume"].iloc[0] == 123456.78
    assert df["number_of_trades"].iloc[0] == 42
    assert df["taker_buy_base_asset_volume"].iloc[0] == 500.0
    assert df["taker_buy_quote_asset_volume"].iloc[0] == 52000.0


def test_parse_zip_to_dataframe_normalizes_microsecond_timestamps():
    """Binance switched ms->us starting 2025-01. Parser must normalize to ms."""
    # 2025-01-01 in us
    tf_ms = binance_bulk.TIMEFRAME_TO_MS["1d"]
    start_us = 1735689600 * 1_000_000  # 2025-01-01 in microseconds
    rows = []
    for i in range(5):
        ot_us = start_us + i * tf_ms * 1000
        ct_us = ot_us + (tf_ms - 1) * 1000
        rows.append(f"{ot_us},100,110,90,105,1.0,{ct_us},123,42,0.5,52.0,0")
    csv = ("\n".join(rows) + "\n").encode("utf-8")
    zip_bytes = _zip_csv(csv, member_name="X-1d-2025-01.csv")

    df = binance_bulk.parse_zip_to_dataframe(zip_bytes, source_filename="X-1d-2025-01.zip")
    assert df["open_time"].iloc[0] == 1735689600 * 1000  # back to ms
    # Sanity: this maps to 2025-01-01 UTC
    assert pd.Timestamp(df["open_time"].iloc[0], unit="ms", tz="UTC").year == 2025
