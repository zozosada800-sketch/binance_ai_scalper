"""
live_bot.py
===========

Task 6: the main execution engine of the ML scalping system.

Every SCAN_INTERVAL_SECONDS (default 60 s) the bot runs one full cycle in a
continuous asynchronous loop:

    1. Fetch LIVE OHLCV for the top USDT pairs     (data_pipeline.py)
       - ccxt fetch_ohlcv(limit=...) without 'since' returns the most
         recent candles, ending with the current still-forming live candle.
    2. Fetch news sentiment for every coin         (sentiment_analyzer.py)
    3. Build features and merge sentiment          (feature_engineering.py)
    4. Score every coin with the trained model     (ml_model.py joblib bundle)
    5. Pick the ONE coin with the highest P(up) and render a terminal
       dashboard with entry / take-profit / stop-loss / confidence.

Freshness guarantee
-------------------
A stale dataset (e.g. candles ending 2025-01-01) can never silently produce
a signal: after every fetch the bot checks the age of each symbol's newest
candle against MAX_CANDLE_AGE_BARS. Stale symbols are excluded from scoring
and flagged in the dashboard. No dates are hardcoded - freshness is always
measured against the current UTC wall clock, so the bot always operates on
"today" (whatever the machine's current date is).

Timezone
--------
All timestamps displayed in the dashboard and logs are converted from UTC
to Africa/Cairo local time (LOCAL_TZ, DST-aware).

Take-profit / stop-loss use the exact ATR barriers the model was trained on
(persisted in bundle["label_config"]):

    TP = entry + pt_atr_mult * ATR
    SL = entry - sl_atr_mult * ATR

Run:
    python live_bot.py                 # continuous loop, scan every 60 s
    python live_bot.py --train         # train a model first if none exists
    python live_bot.py --interval 30   # scan every 30 s
    python live_bot.py --once          # single scan, then exit
    python live_bot.py --no-color      # plain output (auto-disabled when
                                       # stdout is not a terminal)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from data_pipeline import load_market_data
from feature_engineering import PRIMARY_TF, build_features
from ml_model import (
    DEFAULT_MODEL_PATH,
    PT_ATR_MULT,
    SL_ATR_MULT,
    load_model,
    predict_latest,
    train_models,
)
from sentiment_analyzer import describe_sentiment, get_sentiment_scores

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCAN_INTERVAL_SECONDS = 60    # continuous loop: one scan per minute
CONFIDENCE_TARGET = 0.90      # "targeting >90% confidence"
MAX_CANDLE_AGE_BARS = 2       # newest candle may be at most this many bars old
TOP_N_DISPLAY = 5             # runner-up rows shown in the dashboard
BOX_W = 78                    # dashboard inner width (between the ║ bars)

LOCAL_TZ_NAME = "Africa/Cairo"
LOCAL_TZ_LABEL = "Cairo"

ANSI = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "green_bold": "\033[1;92m",
    "red_bold": "\033[1;91m",
    "yellow_bold": "\033[1;93m",
    "cyan_bold": "\033[1;96m",
}
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("live_bot")


# ---------------------------------------------------------------------------
# Timezone helpers (all display times are Africa/Cairo local)
# ---------------------------------------------------------------------------

def _resolve_local_tz():
    """
    Resolve the local display timezone.

    Prefers the IANA zone 'Africa/Cairo' (DST-aware); falls back to a fixed
    UTC+02:00 offset if the zone database is unavailable (e.g. Windows
    without the 'tzdata' package).
    """
    try:
        pd.Timestamp("2026-01-01", tz="UTC").tz_convert(LOCAL_TZ_NAME)
        return LOCAL_TZ_NAME
    except Exception:
        logger.warning(
            "IANA timezone '%s' unavailable - falling back to fixed UTC+02:00.",
            LOCAL_TZ_NAME,
        )
        return timezone(timedelta(hours=2))


LOCAL_TZ = _resolve_local_tz()


def to_local(ts) -> pd.Timestamp:
    """Convert a UTC (or naive-UTC) timestamp to Africa/Cairo local time."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(LOCAL_TZ)


def fmt_local(ts) -> str:
    """Format a timestamp as 'YYYY-MM-DD HH:MM:SS' in Cairo local time."""
    return f"{to_local(ts):%Y-%m-%d %H:%M:%S}"


def _utc_age(ts) -> pd.Timedelta:
    """Age of a timestamp relative to the current UTC wall clock."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return pd.Timestamp.now(tz="UTC") - ts


def fmt_age(delta: pd.Timedelta) -> str:
    """Human-readable age: '42s', '04m 12s', '03h 12m'."""
    total = max(0, int(delta.total_seconds()))
    if total < 60:
        return f"{total}s"
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes:02d}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}h {minutes:02d}m"


# ---------------------------------------------------------------------------
# Freshness gate - stale data must never produce a "live" signal
# ---------------------------------------------------------------------------

_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
}


def _bar_timedelta(timeframe: str) -> pd.Timedelta:
    """Duration of one bar for a Binance-style timeframe string."""
    minutes = _TF_MINUTES.get(timeframe)
    if minutes is not None:
        return pd.Timedelta(minutes=minutes)
    return pd.Timedelta(timeframe)  # pandas parses e.g. '5min', '1h'


def find_stale_symbols(
    symbols: List[str],
    data: Dict[Tuple[str, str], pd.DataFrame],
    timeframe: str = PRIMARY_TF,
    max_age_bars: int = MAX_CANDLE_AGE_BARS,
) -> Set[str]:
    """
    Return the symbols whose newest candle is too old to trade on.

    The last row of each frame is the live, still-forming candle, so a
    healthy feed is never more than ~1 bar old. Anything older than
    max_age_bars indicates a stale/historical dataset (e.g. data ending
    2025-01-01) and must not be used for live signals.
    """
    now_utc = pd.Timestamp.now(tz="UTC")
    max_age = _bar_timedelta(timeframe) * max_age_bars
    stale: Set[str] = set()

    for symbol in symbols:
        df = data.get((symbol, timeframe))
        if df is None or df.empty:
            logger.warning("%s: no %s data returned - marking stale.", symbol, timeframe)
            stale.add(symbol)
            continue
        last_ts = pd.Timestamp(df.index[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        age = now_utc - last_ts
        if age > max_age:
            logger.warning(
                "%s: newest %s candle is %s old (%s %s local) - STALE, excluded.",
                symbol, timeframe, fmt_age(age), fmt_local(last_ts), LOCAL_TZ_LABEL,
            )
            stale.add(symbol)
    return stale


def log_newest_candle(
    data: Dict[Tuple[str, str], pd.DataFrame], timeframe: str = PRIMARY_TF
) -> None:
    """Log the freshest candle across all symbols (proves real-time data)."""
    newest: Optional[pd.Timestamp] = None
    newest_symbol = ""
    for (symbol, tf), df in data.items():
        if tf != timeframe or df is None or df.empty:
            continue
        ts = pd.Timestamp(df.index[-1])
        if newest is None or ts > newest:
            newest, newest_symbol = ts, symbol
    if newest is None:
        logger.warning("No %s data at all - nothing to trade on.", timeframe)
        return
    if newest.tzinfo is None:
        newest = newest.tz_localize("UTC")
    age = pd.Timestamp.now(tz="UTC") - newest
    logger.info(
        "Newest %s candle: %s %s local (age %s, %s)",
        timeframe, fmt_local(newest), LOCAL_TZ_LABEL, fmt_age(age), newest_symbol,
    )


# ---------------------------------------------------------------------------
# Feature / sentiment merging
# ---------------------------------------------------------------------------

def merge_sentiment(
    features: Dict[str, pd.DataFrame], scores: Dict[str, float]
) -> Dict[str, pd.DataFrame]:
    """
    Add a 'sentiment' column (in [-1, 1]) to every symbol's feature frame.

    The persisted model selects only the columns listed in its bundle, so
    this column is backward compatible: models trained without sentiment
    simply ignore it, while models retrained with it pick it up automatically.
    """
    for symbol, feat in features.items():
        feat["sentiment"] = float(scores.get(symbol, 0.0))
    return features


# ---------------------------------------------------------------------------
# Scanning / trade plans
# ---------------------------------------------------------------------------

def evaluate_symbols(
    symbols: List[str],
    features: Dict[str, pd.DataFrame],
    bundle: Dict[str, Any],
    stale_symbols: Optional[Set[str]] = None,
) -> List[dict]:
    """
    Score the latest (live) bar of every fresh symbol and build its trade plan.

    Returns candidate dicts sorted by descending probability:
    {symbol, proba, entry, atr, tp, sl, pt_mult, sl_mult, sentiment, timestamp}.
    """
    stale_symbols = stale_symbols or set()
    label_cfg = bundle.get("label_config", {})
    pt_mult = float(label_cfg.get("pt_atr_mult", PT_ATR_MULT))
    sl_mult = float(label_cfg.get("sl_atr_mult", SL_ATR_MULT))

    candidates: List[dict] = []
    for symbol in symbols:
        if symbol in stale_symbols:
            logger.warning("%s: skipped - stale market data.", symbol)
            continue
        feat = features.get(symbol)
        if feat is None or feat.empty:
            continue

        try:
            proba = predict_latest(feat, bundle)
        except KeyError as exc:
            logger.warning("%s: feature mismatch with model (%s) - skipped", symbol, exc)
            continue
        if pd.isna(proba):
            logger.warning("%s: incomplete latest feature row - skipped", symbol)
            continue

        last = feat.iloc[-1]
        entry = float(last["close"])
        atr = float(last["atr"])
        if pd.isna(entry) or pd.isna(atr) or atr <= 0:
            logger.warning("%s: invalid entry price / ATR - skipped", symbol)
            continue

        candidates.append(
            {
                "symbol": symbol,
                "proba": float(proba),
                "entry": entry,
                "atr": atr,
                "pt_mult": pt_mult,
                "sl_mult": sl_mult,
                "tp": entry + pt_mult * atr,
                "sl": entry - sl_mult * atr,
                "sentiment": float(last.get("sentiment", 0.0)),
                "timestamp": feat.index[-1],
            }
        )

    candidates.sort(key=lambda c: c["proba"], reverse=True)
    return candidates


async def run_scan(bundle: Dict[str, Any]) -> Tuple[List[dict], int]:
    """
    One full cycle: live data -> freshness gate -> sentiment -> features ->
    model scores. Returns (candidates, stale_count).
    """
    logger.info("[1/4] Fetching LIVE market data (most recent candles) ...")
    symbols, data = await load_market_data()

    stale = find_stale_symbols(symbols, data)
    log_newest_candle(data)

    logger.info("[2/4] Fetching news sentiment ...")
    scores = await get_sentiment_scores(symbols)

    logger.info("[3/4] Building features and merging sentiment ...")
    features = merge_sentiment(build_features(symbols, data), scores)

    logger.info("[4/4] Scoring every coin with the model ...")
    candidates = evaluate_symbols(symbols, features, bundle, stale_symbols=stale)

    if candidates:
        top = candidates[0]
        logger.info(
            "Top signal: %s  P(up)=%.1f%%  entry=%.6f  TP=%.6f  SL=%.6f",
            top["symbol"], top["proba"] * 100.0, top["entry"], top["tp"], top["sl"],
        )
    else:
        logger.warning("No tradable candidates this cycle.")
    return candidates, len(stale)


# ---------------------------------------------------------------------------
# Dashboard rendering (all times in Africa/Cairo local time)
# ---------------------------------------------------------------------------

def _visible_len(text: str) -> int:
    """Length of a string excluding ANSI escape sequences."""
    return len(_ANSI_RE.sub("", text))


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _visible_len(text))


def _center(text: str, width: int) -> str:
    pad = max(0, width - len(text))
    left = pad // 2
    return " " * left + text + " " * (pad - left)


def render_dashboard(
    candidates: List[dict],
    scan_no: int,
    model_path: str,
    stale_count: int = 0,
    use_color: bool = True,
    top_n: int = TOP_N_DISPLAY,
) -> str:
    """Render the terminal dashboard; returns the full string to print."""
    W = BOX_W
    lines: List[str] = []

    def colorize(text: str, color: Optional[str] = None) -> str:
        if color and use_color:
            return f"{ANSI[color]}{text}{ANSI['reset']}"
        return text

    def row(text: str = "", color: Optional[str] = None, center: bool = False) -> None:
        content = _center(text, W) if center else " " + _pad(text, W - 2) + " "
        lines.append("║" + colorize(content, color) + "║")

    def heavy_sep() -> None:
        lines.append("╠" + "═" * W + "╣")

    def light_sep() -> None:
        lines.append("╟" + "─" * W + "╢")

    lines.append("╔" + "═" * W + "╗")
    row(
        f"ML SCALPING BOT  ·  LIVE SIGNAL  ·  {LOCAL_TZ_LABEL} LOCAL TIME",
        color="cyan_bold", center=True,
    )
    heavy_sep()

    now_local = pd.Timestamp.now(tz=LOCAL_TZ)
    header = (
        f"Scan #{scan_no}  ·  {now_local:%Y-%m-%d %H:%M:%S} {LOCAL_TZ_LABEL}"
        f"  ·  {len(candidates)} coins scored"
    )
    if stale_count:
        header += f"  ·  {stale_count} stale skipped"
    row(header)
    row(f"Model: {model_path}")
    light_sep()

    if not candidates:
        if stale_count:
            row(f"{stale_count} symbol(s) excluded: stale market data.", color="yellow")
        row("No tradable candidates this scan.", color="yellow")
        row("Check the log lines above - data/features/model may have failed.")
        lines.append("╚" + "═" * W + "╝")
        return "\n".join(lines)

    best = candidates[0]
    conf_pct = best["proba"] * 100.0
    target_met = best["proba"] > CONFIDENCE_TARGET
    tp_pct = (best["tp"] / best["entry"] - 1.0) * 100.0
    sl_pct = (best["sl"] / best["entry"] - 1.0) * 100.0
    bar_age = fmt_age(_utc_age(best["timestamp"]))

    row(f"★ BEST OPPORTUNITY:  {best['symbol']}", color="green_bold")
    row(
        f"Signal bar: {fmt_local(best['timestamp'])} {LOCAL_TZ_LABEL}  (age {bar_age})",
        color="dim",
    )
    light_sep()
    row(f"Entry Price ............ {best['entry']:.6f} USDT")
    row(
        f"Take Profit (+{best['pt_mult']:.1f} ATR) .. {best['tp']:.6f} USDT  ({tp_pct:+.2f}%)",
        color="green",
    )
    row(
        f"Stop Loss  (-{best['sl_mult']:.1f} ATR) .. {best['sl']:.6f} USDT  ({sl_pct:+.2f}%)",
        color="red",
    )
    if target_met:
        row(
            f"Confidence ............. {conf_pct:.1f}%  [TARGET >{CONFIDENCE_TARGET * 100:.0f}% MET]",
            color="green_bold",
        )
    else:
        row(
            f"Confidence ............. {conf_pct:.1f}%  [below {CONFIDENCE_TARGET * 100:.0f}% target]",
            color="yellow",
        )
    row(
        f"News Sentiment ......... {best['sentiment']:+.2f} "
        f"({describe_sentiment(best['sentiment'])})"
    )

    light_sep()
    row("Ranking by confidence:")
    row(
        f"{'#':<4}{'Symbol':<14}{'Prob':>7}   {'Sentiment':>9}   "
        f"{'Bar (' + LOCAL_TZ_LABEL + ')':>11}",
        color="dim",
    )
    for i, cand in enumerate(candidates[:top_n], start=1):
        row(
            f"{i:<4}{cand['symbol']:<14}{cand['proba'] * 100:>6.1f}%   "
            f"{cand['sentiment']:>+9.2f}   {to_local(cand['timestamp']):%H:%M}"
        )

    lines.append("╚" + "═" * W + "╝")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _model_mtime(path: str) -> Optional[float]:
    """File mtime of the model bundle, or None if unreadable."""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


async def _train_model(model_path: str) -> None:
    """Fetch data, build features (+ sentiment) and train/persist a model."""
    logger.info("No model found - training a new one before starting the bot ...")
    symbols, data = await load_market_data()
    scores = await get_sentiment_scores(symbols)
    features = merge_sentiment(build_features(symbols, data), scores)
    best_name, _model, metrics = train_models(
        symbols, data, features, model_path=model_path
    )
    logger.info(
        "Trained %s (test ROC AUC %.4f) -> %s",
        best_name, metrics[best_name].get("roc_auc", float("nan")), model_path,
    )


# ---------------------------------------------------------------------------
# Main continuous loop
# ---------------------------------------------------------------------------

async def main_loop(args: argparse.Namespace) -> None:
    model_path = args.model

    if not Path(model_path).exists():
        if args.train:
            await _train_model(model_path)
        else:
            raise SystemExit(
                f"Model bundle not found at '{model_path}'.\n"
                "Train one first with:   python ml_model.py\n"
                "or start the bot with:  python live_bot.py --train"
            )

    bundle = load_model(model_path)
    model_mtime = _model_mtime(model_path)
    test_auc = bundle.get("test_metrics", {}).get("roc_auc", float("nan"))
    logger.info("Model loaded from %s (test ROC AUC: %.4f)", model_path, test_auc)
    logger.info(
        "Live bot started - scanning every %d s. Display timezone: %s (%s).",
        args.interval, LOCAL_TZ_NAME, LOCAL_TZ_LABEL,
    )

    use_color = args.color and sys.stdout.isatty()
    scan_no = 0
    while True:                                   # continuous asynchronous loop
        scan_no += 1
        started = pd.Timestamp.now(tz="UTC")
        try:
            candidates, stale_count = await run_scan(bundle)
        except Exception:
            logger.exception("Scan #%d failed - retrying next cycle.", scan_no)
            candidates, stale_count = [], 0

        # Hot-reload the model if it was retrained on disk between scans.
        new_mtime = _model_mtime(model_path)
        if (
            new_mtime is not None and model_mtime is not None
            and new_mtime != model_mtime
        ):
            try:
                bundle = load_model(model_path)
                model_mtime = new_mtime
                logger.info("Model bundle changed on disk - reloaded.")
            except Exception:
                logger.exception("Failed to reload model - keeping previous bundle.")

        print(
            render_dashboard(
                candidates, scan_no, model_path,
                stale_count=stale_count, use_color=use_color, top_n=args.top,
            ),
            flush=True,
        )

        if args.once:
            break

        elapsed = (pd.Timestamp.now(tz="UTC") - started).total_seconds()
        logger.info(
            "Scan #%d took %.1f s - next scan in %d s (Ctrl+C to stop).",
            scan_no, elapsed, args.interval,
        )
        await asyncio.sleep(args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Task 6: live ML scalping bot "
            "(live data -> sentiment -> features -> model -> dashboard)"
        )
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="path to the joblib model bundle",
    )
    parser.add_argument(
        "--interval", type=int, default=SCAN_INTERVAL_SECONDS,
        help="seconds between scans (default 60)",
    )
    parser.add_argument(
        "--top", type=int, default=TOP_N_DISPLAY,
        help="runner-up rows to display (default 5)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="train a model first if the bundle is missing",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="run a single scan and exit",
    )
    parser.add_argument(
        "--no-color", dest="color", action="store_false",
        help="disable ANSI colours",
    )
    bot_args = parser.parse_args()
    bot_args.interval = max(10, bot_args.interval)
    bot_args.top = max(1, bot_args.top)
    try:
        asyncio.run(main_loop(bot_args))
    except KeyboardInterrupt:
        print("\nBot stopped by user. Bye!")
