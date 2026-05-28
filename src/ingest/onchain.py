"""On-chain metrics — BTC via blockchain.com (free, no key) + ETH via Etherscan
(only when ``ETHERSCAN_API_KEY`` is provided).

Per the spec, this adapter must:

* Not break the pipeline when a metric is unavailable — just log and skip.
* Emit an ``onchain_coverage_score`` per asset (0.0–1.0) reflecting how many
  of the configured metrics actually came back, so the modeling layer can
  weight features accordingly.

BTC metrics (all from ``api.blockchain.info/charts/<series>``, daily resolution):

* hash-rate                  -> btc_hash_rate
* difficulty                 -> btc_difficulty
* miners-revenue             -> btc_miner_revenue_usd
* n-transactions             -> btc_n_transactions
* n-unique-addresses         -> btc_active_addresses
* total-bitcoins             -> btc_supply
* transaction-fees-usd       -> btc_tx_fees_usd

ETH metrics (Etherscan free tier — only snapshots available without key
upgrades; we ignore unless ``ETHERSCAN_API_KEY`` is set):

* /api?module=stats&action=ethsupply2          -> eth_supply (snapshot)
* /api?module=gastracker&action=gasoracle      -> eth_gas_safe/proposed/fast (snapshot)

CLI::

    python -m src.ingest.onchain
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ingest._adapter_base import (
    AdapterResult,
    finalize_dataframe,
    populate_timestamps,
    print_result,
    update_source_registry,
    write_parquet_idempotent,
)
from src.ingest import _http
from src.utils.io import repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("ingest.onchain")

BLOCKCHAIN_INFO_BASE = "https://api.blockchain.info/charts"

BTC_SERIES: dict[str, str] = {
    "btc_hash_rate":          "hash-rate",
    "btc_difficulty":         "difficulty",
    "btc_miner_revenue_usd":  "miners-revenue",
    "btc_n_transactions":     "n-transactions",
    "btc_active_addresses":   "n-unique-addresses",
    "btc_supply":             "total-bitcoins",
    "btc_tx_fees_usd":        "transaction-fees-usd",
}

ETHERSCAN_BASE = "https://api.etherscan.io/api"

KNOWN_LIMITATIONS_BTC = [
    "Daily resolution; blockchain.info publishes around midday UTC.",
    "blockchain.info occasionally rate-limits or returns 5xx — retried with backoff.",
]
KNOWN_LIMITATIONS_ETH = [
    "Etherscan free tier exposes snapshots, not deep time series — we append one row per run.",
    "Adapter is unavailable unless ETHERSCAN_API_KEY env var is set.",
]


# ---------------------------------------------------------------------------
# BTC via blockchain.info
# ---------------------------------------------------------------------------

def _fetch_blockchain_series(series: str) -> pd.DataFrame | None:
    """Fetch one blockchain.info chart series at maximum (``timespan=all``) resolution."""
    url = f"{BLOCKCHAIN_INFO_BASE}/{series}?timespan=all&format=json"
    try:
        raw = _http.get_bytes(url)
    except (urllib.error.HTTPError, OSError) as exc:
        _log.warning("blockchain.info %s failed: %s", series, exc)
        return None

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        _log.warning("blockchain.info %s returned non-JSON: %s", series, exc)
        return None

    values = payload.get("values")
    if not isinstance(values, list) or not values:
        _log.warning("blockchain.info %s: no values in response", series)
        return None

    df = pd.DataFrame(values)
    if "x" not in df.columns or "y" not in df.columns:
        _log.warning("blockchain.info %s: unexpected schema %s", series, list(df.columns))
        return None
    df["timestamp_utc"] = pd.to_datetime(df["x"].astype("int64"), unit="s", utc=True)
    df = df[["timestamp_utc", "y"]].rename(columns={"y": "value"})
    return df


def fetch_btc() -> list[AdapterResult]:
    results: list[AdapterResult] = []
    for metric_name, series in BTC_SERIES.items():
        res = AdapterResult(
            name=f"onchain_{metric_name}",
            source=f"blockchain.info:{series}",
            available=False,
            known_limitations=list(KNOWN_LIMITATIONS_BTC),
        )
        df = _fetch_blockchain_series(series)
        # Small inter-request delay to be nice to a free endpoint.
        time.sleep(0.25)
        if df is None or df.empty:
            res.reason = f"no data returned for series {series!r}"
            results.append(res)
            continue
        df = finalize_dataframe(df, index_name="timestamp_utc")
        res.df = df
        res.metric_columns = ["value"]
        res.available = True
        populate_timestamps(res)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# ETH via Etherscan (snapshot only)
# ---------------------------------------------------------------------------

def _etherscan_request(params: dict[str, str], api_key: str) -> dict | None:
    qs = urllib.parse.urlencode({**params, "apikey": api_key})
    url = f"{ETHERSCAN_BASE}?{qs}"
    try:
        raw = _http.get_bytes(url)
    except (urllib.error.HTTPError, OSError) as exc:
        _log.warning("etherscan request failed: %s", exc)
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        _log.warning("etherscan returned non-JSON for %s", params)
        return None


def fetch_eth() -> list[AdapterResult]:
    api_key = os.environ.get("ETHERSCAN_API_KEY")
    results: list[AdapterResult] = []
    if not api_key:
        results.append(AdapterResult(
            name="onchain_eth_supply", source="etherscan",
            available=False,
            reason="ETHERSCAN_API_KEY env var not set; ETH on-chain disabled (spec default).",
            known_limitations=list(KNOWN_LIMITATIONS_ETH),
        ))
        results.append(AdapterResult(
            name="onchain_eth_gas", source="etherscan",
            available=False,
            reason="ETHERSCAN_API_KEY env var not set; ETH on-chain disabled (spec default).",
            known_limitations=list(KNOWN_LIMITATIONS_ETH),
        ))
        return results

    now_utc = datetime.now(tz=timezone.utc).replace(microsecond=0)

    # Supply snapshot
    supply_res = AdapterResult(
        name="onchain_eth_supply", source="etherscan:ethsupply2",
        available=False, known_limitations=list(KNOWN_LIMITATIONS_ETH),
    )
    payload = _etherscan_request({"module": "stats", "action": "ethsupply2"}, api_key)
    if payload and payload.get("status") == "1" and isinstance(payload.get("result"), dict):
        r = payload["result"]
        wei = float(r.get("EthSupply", 0))
        df = pd.DataFrame([{
            "timestamp_utc": now_utc,
            "eth_supply": wei / 1e18 if wei else float("nan"),
            "eth_burnt_fees": float(r.get("BurntFees", 0)) / 1e18 if r.get("BurntFees") else float("nan"),
        }])
        df = finalize_dataframe(df, index_name="timestamp_utc")
        supply_res.df = df
        supply_res.metric_columns = ["eth_supply", "eth_burnt_fees"]
        supply_res.available = True
        populate_timestamps(supply_res)
    else:
        supply_res.reason = f"etherscan ethsupply2 unavailable; response={payload}"
    results.append(supply_res)

    # Gas oracle snapshot
    gas_res = AdapterResult(
        name="onchain_eth_gas", source="etherscan:gasoracle",
        available=False, known_limitations=list(KNOWN_LIMITATIONS_ETH),
    )
    payload = _etherscan_request({"module": "gastracker", "action": "gasoracle"}, api_key)
    if payload and payload.get("status") == "1" and isinstance(payload.get("result"), dict):
        r = payload["result"]
        df = pd.DataFrame([{
            "timestamp_utc": now_utc,
            "gas_safe_gwei": float(r.get("SafeGasPrice", float("nan"))),
            "gas_propose_gwei": float(r.get("ProposeGasPrice", float("nan"))),
            "gas_fast_gwei": float(r.get("FastGasPrice", float("nan"))),
        }])
        df = finalize_dataframe(df, index_name="timestamp_utc")
        gas_res.df = df
        gas_res.metric_columns = ["gas_safe_gwei", "gas_propose_gwei", "gas_fast_gwei"]
        gas_res.available = True
        populate_timestamps(gas_res)
    else:
        gas_res.reason = f"etherscan gasoracle unavailable; response={payload}"
    results.append(gas_res)

    return results


# ---------------------------------------------------------------------------
# Coverage score
# ---------------------------------------------------------------------------

def coverage_scores(results: list[AdapterResult]) -> dict[str, float]:
    btc = [r for r in results if r.name.startswith("onchain_btc_")]
    eth = [r for r in results if r.name.startswith("onchain_eth_")]
    return {
        "BTCUSDT": (sum(1 for r in btc if r.available) / len(btc)) if btc else 0.0,
        "ETHUSDT": (sum(1 for r in eth if r.available) / len(eth)) if eth else 0.0,
    }


def _write_coverage_json(scores: dict[str, float], root: Path) -> Path:
    path = root / "metadata" / "onchain_coverage.json"
    write_json(path, {
        "schema_version": 1,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "scores": scores,
    })
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ingest.onchain")
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--btc-only", action="store_true")
    parser.add_argument("--eth-only", action="store_true")
    args = parser.parse_args(argv)

    results: list[AdapterResult] = []
    if not args.eth_only:
        results.extend(fetch_btc())
    if not args.btc_only:
        results.extend(fetch_eth())

    root = repo_root()
    if not args.no_persist:
        for res in results:
            if not res.available:
                continue
            short = res.name.removeprefix("onchain_")
            out = root / "data" / "processed" / "onchain" / f"{short}.parquet"
            write_parquet_idempotent(res.df, out)
            res.parquet_path = str(out.relative_to(root))
        update_source_registry(results)

    scores = coverage_scores(results)
    if not args.no_persist:
        _write_coverage_json(scores, root)
    for res in results:
        print_result(res)
    print(f"\nonchain coverage: BTC={scores['BTCUSDT']:.2f}  ETH={scores['ETHUSDT']:.2f}")

    any_available = any(r.available for r in results)
    return 0 if any_available else 2


if __name__ == "__main__":
    sys.exit(main())
