"""On-chain feature transformations (BTC primary; ETH if Etherscan data exists).

Each blockchain.info series is causally joined onto the target index using
backward merge_asof so a bar at ``t`` never sees a value published after
``t``. blockchain.info publishes once per day around midday UTC.

Outputs (when sources exist):

* btc_active_addresses, btc_hash_rate, btc_difficulty, btc_miner_revenue_usd,
  btc_n_transactions, btc_supply, btc_tx_fees_usd
* For each: a rolling-30 z-score and a daily delta where it makes sense
* eth_supply / eth_burnt_fees / eth_gas_* when Etherscan data is present
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.io import repo_root
from src.utils.logging import get_logger

_log = get_logger("features.onchain")


BTC_METRICS = (
    "btc_active_addresses",
    "btc_hash_rate",
    "btc_difficulty",
    "btc_miner_revenue_usd",
    "btc_n_transactions",
    "btc_supply",
    "btc_tx_fees_usd",
)

ETH_METRICS = (
    "eth_supply",
    "eth_gas",
)


def _load_onchain_metric(metric: str, root: Path) -> pd.DataFrame | None:
    path = root / "data" / "processed" / "onchain" / f"{metric}.parquet"
    if not path.exists():
        return None
    df = pq.read_table(path).to_pandas()
    if "timestamp_utc" not in df.columns:
        return None
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    df = df.set_index("timestamp_utc")
    return df[~df.index.duplicated(keep="last")]


def _causal_join_series(
    target_index: pd.DatetimeIndex, src: pd.DataFrame, value_col: str
) -> pd.Series:
    target_df = pd.DataFrame(index=target_index).sort_index()
    target_df["__key"] = target_df.index
    src_reset = src.reset_index().rename(columns={"timestamp_utc": "__key"})
    merged = pd.merge_asof(
        target_df.reset_index(drop=True).sort_values("__key"),
        src_reset[["__key", value_col]].sort_values("__key"),
        on="__key", direction="backward",
    )
    out = pd.Series(merged[value_col].to_numpy(), index=target_df.index)
    return out


def build_onchain_features(
    target_index: pd.DatetimeIndex,
    symbol: str,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    root = root or repo_root()
    out = pd.DataFrame(index=target_index)

    if symbol == "BTCUSDT":
        for metric in BTC_METRICS:
            df = _load_onchain_metric(metric, root)
            if df is None or "value" not in df.columns or df.empty:
                continue
            s = _causal_join_series(target_index, df, "value")
            out[metric] = s
            mu = s.rolling(30, min_periods=30).mean()
            sd = s.rolling(30, min_periods=30).std(ddof=0)
            out[f"{metric}_zscore_30"] = (s - mu) / sd.replace(0, np.nan)
            out[f"{metric}_delta"] = s.diff()

    elif symbol == "ETHUSDT":
        for metric in ETH_METRICS:
            df = _load_onchain_metric(metric, root)
            if df is None or df.empty:
                continue
            for col in df.columns:
                s = _causal_join_series(target_index, df, col)
                out[col] = s

    return out
