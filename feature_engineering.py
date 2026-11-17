"""
feature_engineering.py
======================

Task 3: feature engineering for the ML scalping engine.

Consumes the output of data_pipeline.py:

    symbols: List[str]
    data:    {(symbol, timeframe): UTC-indexed OHLCV DataFrame}

and produces, for every symbol, a tidy feature DataFrame on the 5m index:

    features: {symbol: DataFrame}   # one row per 5m bar, one column per feature

Feature groups
--------------
5m (primary timeframe):
    * multi-horizon log returns
    * trend: EMA distances/slope (ATR-normalised), MACD, ADX/DMI
    * volatility: ATR, ATR%, Bollinger bandwidth + %B position
    * oscillators: RSI, Stochastic RSI
    * volume: z-score, OBV slope
    * candle anatomy: body ratio, close position, wicks, distance to
      rolling highs/lows (breakout context)

15m (higher timeframe, lookahead-safe):
    * return, RSI, ATR%, trend vs EMA, volume z-score
    * aligned onto the 5m index with a one-bar shift so that only fully
      closed 15m bars are ever used (no lookahead).

All features are computed from past data only and are scale-free
(ratios, ATR- or range-normalised) so they pool cleanly across symbols
in the ML stage (Task 4).

Usage (standalone smoke test):
    python feature_engineering.py

Usage (from other modules):
    from data_pipeline import load_market_data
    from feature_engineering import build_features
    symbols, data = await load_market_data()
    features = build_features(symbols, data)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRIMARY_TF = "5m"           # timeframe the feature rows live on
HTF_TF = "15m"              # higher timeframe used for context features

# --- indicator parameters (5m) ---
RSI_LEN = 14
ATR_LEN = 14
ADX_LEN = 14
EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_LEN, BB_STD = 20, 2.0
STOCHRSI_LEN = 14
VOL_Z_WIN = 50              # rolling window for volume z-score
RETURN_HORIZONS = (1, 3, 5, 15)   # log-return horizons, in 5m bars
ROLL_HIGHLOW_WIN = 24       # rolling high/low window (2h of 5m bars)

# Raw OHLCV columns are kept in the frames for convenience (labeling needs
# high/low/close) but are NOT features - see feature_columns().
RAW_COLUMNS = ("open", "high", "low", "close", "volume")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("feature_engineering")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rename_by_prefix(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename pandas-ta output columns (e.g. 'MACDh_12_26_9') to stable names.

    Longer prefixes must come first in `mapping` because iteration stops at
    the first matching prefix ('MACDh' before 'MACD', etc.).
    """
    renamed = {}
    for col in df.columns:
        for prefix, name in mapping.items():
            if col.startswith(prefix):
                renamed[col] = name
                break
    return df.rename(columns=renamed)


def _col(
    df: Optional[pd.DataFrame], name: str, index: pd.Index
) -> pd.Series:
    """Fetch a column by name, or an all-NaN series if absent (robustness)."""
    if df is None or name not in df.columns:
        return pd.Series(np.nan, index=index)
    return df[name]


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Columns of `df` that are usable as ML features (raw OHLCV excluded)."""
    return [c for c in df.columns if c not in RAW_COLUMNS]


# ---------------------------------------------------------------------------
# 5m indicator block
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 5m features on a copy of the OHLCV frame."""
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]
    open_, volume = out["open"], out["volume"]

    # --- returns / momentum -------------------------------------------------
    for h in RETURN_HORIZONS:
        out[f"ret_{h}"] = np.log(close / close.shift(h))
    roc = ta.roc(close, length=10)
    out["roc_10"] = np.nan if roc is None else roc / 100.0  # ta.roc is in %

    # --- trend ---------------------------------------------------------------
    atr = ta.atr(high, low, close, length=ATR_LEN)
    out["atr"] = atr
    out["atr_pct"] = atr / close

    out["ema_fast"] = ta.ema(close, length=EMA_FAST)
    out["ema_slow"] = ta.ema(close, length=EMA_SLOW)
    out["ema_trend"] = ta.ema(close, length=EMA_TREND)
    # ATR-normalised distances keep features scale-free across symbols.
    out["close_to_ema_fast"] = (close - out["ema_fast"]) / atr
    out["close_to_ema_slow"] = (close - out["ema_slow"]) / atr
    out["ema_fast_slope"] = out["ema_fast"].diff(3) / atr

    macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    macd = _rename_by_prefix(
        macd, {"MACDh": "macd_hist", "MACDs": "macd_signal", "MACD": "macd"}
    )
    out["macd"] = _col(macd, "macd", out.index) / close
    out["macd_signal"] = _col(macd, "macd_signal", out.index) / close
    out["macd_hist"] = _col(macd, "macd_hist", out.index) / close

    adx = ta.adx(high, low, close, length=ADX_LEN)
    adx = _rename_by_prefix(adx, {"ADX": "adx", "DMP": "di_plus", "DMN": "di_minus"})
    out["adx"] = _col(adx, "adx", out.index)
    out["di_plus"] = _col(adx, "di_plus", out.index)
    out["di_minus"] = _col(adx, "di_minus", out.index)

    # --- volatility bands ------------------------------------------------------
    bb = ta.bbands(close, length=BB_LEN, std=BB_STD)
    bb = _rename_by_prefix(
        bb,
        {
            "BBL": "bb_low", "BBM": "bb_mid", "BBU": "bb_high",
            "BBB": "bb_width", "BBP": "bb_pctl",
        },
    )
    out["bb_pctl"] = _col(bb, "bb_pctl", out.index)          # 0..1 in-band position
    out["bb_width"] = _col(bb, "bb_width", out.index) / 100.0  # % -> fraction

    # --- oscillators ------------------------------------------------------------
    out["rsi"] = ta.rsi(close, length=RSI_LEN)
    stoch = ta.stochrsi(close, length=STOCHRSI_LEN)
    stoch = _rename_by_prefix(
        stoch, {"STOCHRSIk": "stochrsi_k", "STOCHRSId": "stochrsi_d"}
    )
    out["stochrsi_k"] = _col(stoch, "stochrsi_k", out.index)
    out["stochrsi_d"] = _col(stoch, "stochrsi_d", out.index)

    # --- volume -------------------------------------------------------------------
    vol_mean = volume.rolling(VOL_Z_WIN).mean()
    vol_std = volume.rolling(VOL_Z_WIN).std()
    out["vol_z"] = (volume - vol_mean) / (vol_std + 1e-12)
    obv = ta.obv(close, volume)
    out["obv_slope"] = (
        np.nan if obv is None
        else obv.diff(5) / (volume.rolling(5).sum() + 1e-12)
    )

    # --- candle anatomy (price action) ---------------------------------------------
    rng = (high - low).replace(0, np.nan)
    out["body_ratio"] = (close - open_) / rng                # -1..1
    out["close_pos_in_bar"] = (close - low) / rng            # 0..1
    out["upper_wick"] = (high - np.maximum(close, open_)) / rng
    out["lower_wick"] = (np.minimum(close, open_) - low) / rng
    # Distance to recent extremes in ATR units (breakout context).
    out["dist_to_high_24"] = (close - high.rolling(ROLL_HIGHLOW_WIN).max()) / atr
    out["dist_to_low_24"] = (close - low.rolling(ROLL_HIGHLOW_WIN).min()) / atr

    return out


# ---------------------------------------------------------------------------
# 15m context block (lookahead-safe)
# ---------------------------------------------------------------------------

def add_htf_context(df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Merge higher-timeframe context onto the 5m index without lookahead.

    A 15m bar stamped T only *closes* at T+15m, so the whole context frame is
    shifted one 15m bar forward before alignment. After the shift, the value
    aligned to a 5m timestamp T is the context of the last 15m bar that fully
    closed strictly before T.
    """
    out = df_5m.copy()
    h_close, h_high, h_low, h_vol = (
        df_15m["close"], df_15m["high"], df_15m["low"], df_15m["volume"],
    )
    h_atr = ta.atr(h_high, h_low, h_close, length=ATR_LEN)

    ctx = pd.DataFrame(index=df_15m.index)
    ctx["htf_ret_1"] = np.log(h_close / h_close.shift(1))
    ctx["htf_rsi"] = ta.rsi(h_close, length=RSI_LEN)
    ctx["htf_atr_pct"] = h_atr / h_close
    h_ema = ta.ema(h_close, length=EMA_SLOW)
    ctx["htf_trend"] = (h_close - h_ema) / h_atr
    h_mean = h_vol.rolling(VOL_Z_WIN).mean()
    h_std = h_vol.rolling(VOL_Z_WIN).std()
    ctx["htf_vol_z"] = (h_vol - h_mean) / (h_std + 1e-12)

    ctx = ctx.shift(1).reindex(out.index, method="ffill")
    for col in ctx.columns:
        out[col] = ctx[col]
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_features(
    symbols: List[str],
    data: Dict[Tuple[str, str], pd.DataFrame],
    dropna: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Build the feature matrix for every symbol.

    Returns {symbol: DataFrame} indexed by the 5m UTC timestamp. With
    dropna=True the indicator warm-up rows (leading NaNs) are removed.
    """
    features: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        df_5m = data.get((symbol, PRIMARY_TF))
        df_15m = data.get((symbol, HTF_TF))

        if df_5m is None or df_5m.empty:
            logger.warning("%s: no %s data, skipping", symbol, PRIMARY_TF)
            continue

        feat = add_indicators(df_5m)

        if df_15m is not None and not df_15m.empty:
            feat = add_htf_context(feat, df_15m)
        else:
            logger.warning("%s: no %s data, HTF context skipped", symbol, HTF_TF)

        if dropna:
            before = len(feat)
            feat = feat.dropna()
            logger.debug(
                "%s: dropped %d warm-up rows", symbol, before - len(feat)
            )

        if feat.empty:
            logger.warning("%s: no complete feature rows, skipping", symbol)
            continue

        features[symbol] = feat
        logger.info(
            "%s: %d rows x %d features",
            symbol, len(feat), len(feature_columns(feat)),
        )

    if not features:
        raise RuntimeError("No features could be built - check pipeline output.")
    return features


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

async def _main() -> None:
    from data_pipeline import load_market_data

    symbols, data = await load_market_data()
    features = build_features(symbols, data)

    print("\n" + "=" * 78)
    print("FEATURE ENGINEERING SUMMARY (5m primary + 15m context)")
    print("=" * 78)
    for i, symbol in enumerate(symbols, start=1):
        feat = features.get(symbol)
        if feat is None:
            print(f"{i:>2}. {symbol:<14} MISSING")
            continue
        print(
            f"{i:>2}. {symbol:<14} {len(feat):>4} rows x "
            f"{len(feature_columns(feat)):>3} features "
            f"[{feat.index[0]:%Y-%m-%d %H:%M} -> {feat.index[-1]:%Y-%m-%d %H:%M} UTC]"
        )
    print("=" * 78)

    first_symbol = next(iter(features))
    cols = feature_columns(features[first_symbol])
    print(f"\nFeature columns ({len(cols)}):")
    for c in cols:
        print(f"  - {c}")


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
