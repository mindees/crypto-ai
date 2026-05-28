"""Derivatives / order-flow features built from the Phase-2 derivatives data.

Causal: every joined value at bar t uses the most recent derivative value
known **at or before** the bar's open_time. Values published after open_time
are excluded via ``merge_asof(direction="backward")``.

Outputs (when source data is available):

* funding_rate, funding_rate_zscore, funding_extreme flag
* open_interest, open_interest_change_pct, oi_x_price_action features
* taker buy/sell volume (where present), taker imbalance, CVD proxy
* global_long_short_ratio, top_trader_position_ratio, long_short_zscore
* funding_oi_governor_risk_score (composite)
* basis = (futures_close - spot_close) / spot_close, when both feeds available

If a derivatives feed is missing entirely for a (symbol, timeframe), the
matching columns are written as NaN — feature selection will drop them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.io import repo_root
from src.utils.logging import get_logger

_log = get_logger("features.flow")


def _read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pq.read_table(path).to_pandas()
    except Exception as exc:  # noqa: BLE001
        _log.warning("could not read %s: %s", path, exc)
        return None
    if "timestamp_utc" not in df.columns:
        return None
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    df = df.set_index("timestamp_utc")
    df = df[~df.index.duplicated(keep="last")]
    return df


def _derivatives_path(symbol: str, metric: str, root: Path) -> Path:
    return root / "data" / "processed" / "derivatives" / symbol / f"{metric}.parquet"


def _causal_join(
    target_index: pd.DatetimeIndex,
    src: pd.DataFrame | None,
    *,
    column_prefix: str,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Backward-merge ``src`` onto ``target_index``. No forward leakage."""
    if src is None or src.empty:
        return pd.DataFrame(index=target_index)
    cols = keep_cols or list(src.columns)
    s = src[cols].copy()
    s = s.sort_index()
    target_df = pd.DataFrame(index=target_index).sort_index()
    target_df["__key"] = target_df.index
    s = s.reset_index()
    s.rename(columns={"timestamp_utc": "__key"}, inplace=True)
    merged = pd.merge_asof(
        target_df.reset_index(drop=True).sort_values("__key"),
        s.sort_values("__key"),
        on="__key",
        direction="backward",
    )
    merged.index = target_df.index
    merged = merged.drop(columns="__key")
    merged.columns = [f"{column_prefix}_{c}" for c in merged.columns]
    return merged


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def build_flow_features(
    ohlcv: pd.DataFrame,
    symbol: str,
    *,
    root: Path | None = None,
    spot_ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return derivatives-derived features aligned to ``ohlcv``'s index.

    ``spot_ohlcv`` is the spot price series for the same symbol — needed for
    basis calculation. ``ohlcv`` here is typically futures.
    """
    root = root or repo_root()
    idx = ohlcv.index

    funding = _read_parquet_if_exists(_derivatives_path(symbol, "funding_rate", root))
    oi = _read_parquet_if_exists(_derivatives_path(symbol, "open_interest_hist", root))
    ls_global = _read_parquet_if_exists(
        _derivatives_path(symbol, "global_long_short_ratio", root)
    )
    ls_top = _read_parquet_if_exists(
        _derivatives_path(symbol, "top_trader_position_ratio", root)
    )
    taker = _read_parquet_if_exists(
        _derivatives_path(symbol, "taker_long_short_volume", root)
    )

    parts: list[pd.DataFrame] = []

    if funding is not None:
        f_joined = _causal_join(idx, funding, column_prefix="deriv", keep_cols=["funding_rate"])
        parts.append(f_joined)

    if oi is not None:
        oi_joined = _causal_join(
            idx, oi, column_prefix="deriv",
            keep_cols=[c for c in ("open_interest", "open_interest_usd") if c in oi.columns],
        )
        parts.append(oi_joined)

    if ls_global is not None:
        lsg = _causal_join(
            idx, ls_global, column_prefix="deriv",
            keep_cols=[c for c in (
                "global_long_account_pct",
                "global_short_account_pct",
                "global_long_short_ratio",
            ) if c in ls_global.columns],
        )
        parts.append(lsg)

    if ls_top is not None:
        lst = _causal_join(
            idx, ls_top, column_prefix="deriv",
            keep_cols=[c for c in (
                "top_long_position_pct",
                "top_short_position_pct",
                "top_long_short_ratio",
            ) if c in ls_top.columns],
        )
        parts.append(lst)

    if taker is not None:
        tk = _causal_join(
            idx, taker, column_prefix="deriv",
            keep_cols=[c for c in (
                "taker_buy_volume",
                "taker_sell_volume",
                "taker_buy_sell_ratio",
            ) if c in taker.columns],
        )
        parts.append(tk)

    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=idx)

    # Derived features (created only when their source columns exist)
    if "deriv_funding_rate" in out.columns:
        fr = out["deriv_funding_rate"]
        out["funding_rate"] = fr
        out["funding_rate_zscore_50"] = _rolling_zscore(fr, 50)
        out["funding_extreme_high"] = (fr > 0.0003).astype("float64")
        out["funding_extreme_low"] = (fr < -0.0003).astype("float64")
        out["funding_rate_8h_delta"] = fr.diff()

    if "deriv_open_interest" in out.columns:
        oi_series = out["deriv_open_interest"]
        out["open_interest"] = oi_series
        out["open_interest_change_pct"] = oi_series.pct_change()
        out["open_interest_zscore_50"] = _rolling_zscore(oi_series, 50)

        # OI vs price action quadrants — price/oi divergence proxies.
        ret = ohlcv["close"].pct_change()
        oi_delta = oi_series.pct_change()
        out["price_up_oi_up"] = ((ret > 0) & (oi_delta > 0)).astype("float64")
        out["price_up_oi_down"] = ((ret > 0) & (oi_delta < 0)).astype("float64")
        out["price_down_oi_up"] = ((ret < 0) & (oi_delta > 0)).astype("float64")
        out["price_down_oi_down"] = ((ret < 0) & (oi_delta < 0)).astype("float64")

    if "deriv_taker_buy_volume" in out.columns and "deriv_taker_sell_volume" in out.columns:
        delta = out["deriv_taker_buy_volume"] - out["deriv_taker_sell_volume"]
        out["taker_delta_proxy"] = delta
        out["cvd_proxy"] = delta.cumsum()
        out["taker_imbalance"] = delta / (
            out["deriv_taker_buy_volume"] + out["deriv_taker_sell_volume"]
        ).replace(0, np.nan)

    if "deriv_global_long_short_ratio" in out.columns:
        out["global_long_short_zscore_50"] = _rolling_zscore(
            out["deriv_global_long_short_ratio"], 50
        )

    # Composite funding/OI governor risk score (0=normal, 1=squeeze risk).
    extreme_funding = out.get("funding_extreme_high", pd.Series(0.0, index=idx)).fillna(0.0) + \
                       out.get("funding_extreme_low", pd.Series(0.0, index=idx)).fillna(0.0)
    high_oi_z = (out.get("open_interest_zscore_50", pd.Series(0.0, index=idx)).fillna(0.0).abs() > 2.0).astype("float64")
    out["funding_oi_governor_risk"] = ((extreme_funding > 0) | (high_oi_z > 0)).astype("float64")

    # Basis (futures - spot) / spot, only when both series are available
    if spot_ohlcv is not None and "close" in spot_ohlcv.columns and not spot_ohlcv.empty:
        spot_close = spot_ohlcv["close"].reindex(idx, method="ffill")
        with np.errstate(divide="ignore", invalid="ignore"):
            basis = (ohlcv["close"] - spot_close) / spot_close.replace(0, np.nan)
        out["basis_futures_vs_spot"] = basis

    return out
