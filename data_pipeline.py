"""
data_pipeline.py
================

Asynchronous market-data pipeline for the ML scalping engine.

Task 2 scope:
    1. Connect to Binance through CCXT (public endpoints only, no API keys).
    2. Rank all spot USDT pairs by 24h quote volume and keep the top N (20).
    3. Asynchronously download OHLCV candles for every selected pair on the
       5m and 15m timeframes.
    4. Convert everything into tidy pandas DataFrames indexed by UTC timestamp
       and return them in a structure the feature-engineering module (next
       task) can consume directly.

Usage (standalone smoke test):
    python data_pipeline.py

Usage (from other modules):
    from data_pipeline import load_market_data
    symbols, data = await load_market_data()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import ccxt.async_support as ccxt_async
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_N_PAIRS = 20            # how many top-volume USDT pairs to select
TIMEFRAMES: Tuple[str, ...] = ("5m", "15m")
CANDLE_LIMIT = 500          # candles per request (Binance allows up to 1000)
MAX_CONCURRENT_REQUESTS = 5 # parallel OHLCV fetches in flight (rate-limit safe)
MAX_RETRIES = 3             # retries per OHLCV request
RETRY_BACKOFF_SECONDS = 2.0 # linear backoff base: 2s, 4s, 6s ...
MIN_CANDLES_RATIO = 0.8     # drop series with < 80% of requested candles

# Stablecoin / fiat bases quoted against USDT have no volatility and are
# useless as scalping instruments, so they are removed from the universe.
EXCLUDED_BASES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "PAXG", "AEUR",
    "EUR", "GBP", "TRY", "BRL", "JPY", "RUB", "UAH", "ZAR",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("data_pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_leveraged_token(base: str, all_bases: set) -> bool:
    """
    Detect Binance leveraged tokens (BTCUP, ETHDOWN, ADAUP, ...).

    A naive suffix check would also catch legitimate coins such as JUP,
    so we only flag a symbol when stripping the suffix leaves another
    listed base asset:
        BTCUP  -> BTC exists  => leveraged (excluded)
        JUP    -> J does not  => legitimate (kept)
    """
    for suffix in ("UP", "DOWN"):
        if base.endswith(suffix):
            stem = base[: -len(suffix)]
            if stem and stem in all_bases:
                return True
    return False


def ohlcv_to_dataframe(ohlcv: list) -> pd.DataFrame:
    """Convert a raw CCXT OHLCV list into a clean, UTC-indexed DataFrame."""
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna().sort_index()
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BinanceDataPipeline:
    """
    Async pipeline exposing three stages:

        fetch_top_usdt_pairs() -> top-N symbols ranked by 24h quote volume
        fetch_all_ohlcv()      -> {(symbol, timeframe): DataFrame}
        run()                  -> (symbols, data) in a single call
    """

    def __init__(
        self,
        top_n: int = TOP_N_PAIRS,
        timeframes: Tuple[str, ...] = TIMEFRAMES,
        candle_limit: int = CANDLE_LIMIT,
    ) -> None:
        self.top_n = top_n
        self.timeframes = timeframes
        self.candle_limit = candle_limit
        self.exchange = ccxt_async.binance(
            {
                "enableRateLimit": True,            # CCXT throttles for us
                "options": {"defaultType": "spot"}, # spot market only
            }
        )

    # -- lifecycle -----------------------------------------------------------

    async def close(self) -> None:
        """Release the underlying aiohttp session. Call when finished."""
        await self.exchange.close()

    # -- stage 1: universe selection ------------------------------------------

    async def fetch_top_usdt_pairs(self) -> List[str]:
        """Return the top-N USDT spot pairs ranked by 24h quote volume."""
        logger.info("Fetching ticker snapshot to rank USDT pairs by volume ...")
        await self.exchange.load_markets()
        tickers = await self.exchange.fetch_tickers()

        all_bases = {symbol.split("/")[0] for symbol in tickers}
        volumes: Dict[str, float] = {}

        for symbol, ticker in tickers.items():
            # Keep spot USDT-quoted pairs only (':' would indicate a swap).
            if ":" in symbol or not symbol.endswith("/USDT"):
                continue
            base = symbol.split("/")[0]
            if base in EXCLUDED_BASES or _is_leveraged_token(base, all_bases):
                continue
            quote_volume = ticker.get("quoteVolume")
            if quote_volume is None:
                continue
            volumes[symbol] = float(quote_volume)

        ranked = sorted(volumes.items(), key=lambda kv: kv[1], reverse=True)
        top_symbols = [symbol for symbol, _ in ranked[: self.top_n]]

        logger.info(
            "Selected top %d USDT pairs: %s",
            len(top_symbols), ", ".join(top_symbols),
        )
        return top_symbols

    # -- stage 2: OHLCV download ----------------------------------------------

    async def _fetch_ohlcv_with_retry(
        self, symbol: str, timeframe: str
    ) -> Optional[list]:
        """Fetch one OHLCV series with retries; permanent errors fail fast."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=self.candle_limit
                )
            except ccxt_async.NetworkError as exc:
                # Transient (timeout, DDoS guard, rate limit) -> retry.
                wait = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Retry %d/%d for %s %s: %s (sleeping %.1fs)",
                    attempt, MAX_RETRIES, symbol, timeframe, exc, wait,
                )
                await asyncio.sleep(wait)
            except ccxt_async.ExchangeError as exc:
                # Permanent (bad symbol, market closed, ...) -> give up now.
                logger.error("Exchange rejected %s %s: %s", symbol, timeframe, exc)
                return None
        return None

    async def fetch_all_ohlcv(
        self, symbols: List[str]
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Fetch OHLCV for every (symbol, timeframe) pair concurrently."""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        data: Dict[Tuple[str, str], pd.DataFrame] = {}

        async def worker(symbol: str, timeframe: str) -> None:
            async with semaphore:
                raw = await self._fetch_ohlcv_with_retry(symbol, timeframe)
            if not raw:
                logger.error("No OHLCV data for %s %s", symbol, timeframe)
                return
            df = ohlcv_to_dataframe(raw)
            min_candles = int(self.candle_limit * MIN_CANDLES_RATIO)
            if len(df) < min_candles:
                logger.warning(
                    "%s %s: only %d candles (need %d), skipping",
                    symbol, timeframe, len(df), min_candles,
                )
                return
            data[(symbol, timeframe)] = df

        tasks = [
            worker(symbol, timeframe)
            for symbol in symbols
            for timeframe in self.timeframes
        ]
        logger.info(
            "Fetching %d OHLCV series (%d pairs x %d timeframes) ...",
            len(tasks), len(symbols), len(self.timeframes),
        )
        await asyncio.gather(*tasks)
        logger.info("Completed %d/%d series.", len(data), len(tasks))
        return data

    # -- orchestration ---------------------------------------------------------

    async def run(self) -> Tuple[List[str], Dict[Tuple[str, str], pd.DataFrame]]:
        """Execute the full pipeline and return (symbols, data)."""
        symbols = await self.fetch_top_usdt_pairs()
        if not symbols:
            raise RuntimeError("No USDT pairs returned by Binance - aborting.")
        data = await self.fetch_all_ohlcv(symbols)
        return symbols, data


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

async def load_market_data(
    top_n: int = TOP_N_PAIRS,
) -> Tuple[List[str], Dict[Tuple[str, str], pd.DataFrame]]:
    """One-shot helper for downstream modules (feature engineering, ML)."""
    pipeline = BinanceDataPipeline(top_n=top_n)
    try:
        return await pipeline.run()
    finally:
        await pipeline.close()


async def _main() -> None:
    """Standalone smoke test: run the pipeline and print a summary report."""
    pipeline = BinanceDataPipeline()
    try:
        symbols, data = await pipeline.run()
    finally:
        await pipeline.close()

    print("\n" + "=" * 78)
    print(f"TOP {len(symbols)} BINANCE USDT PAIRS BY 24H QUOTE VOLUME")
    print("=" * 78)
    for i, symbol in enumerate(symbols, start=1):
        parts = []
        for timeframe in pipeline.timeframes:
            df = data.get((symbol, timeframe))
            if df is None:
                parts.append(f"{timeframe}: MISSING")
            else:
                last_close = df["close"].iloc[-1]
                parts.append(f"{timeframe}: {len(df)} candles (last {last_close:.6g})")
        print(f"{i:>2}. {symbol:<14} " + " | ".join(parts))
    print("=" * 78)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
