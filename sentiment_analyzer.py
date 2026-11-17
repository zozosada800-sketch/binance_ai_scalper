r"""
sentiment_analyzer.py
=====================

Task 5: news-based sentiment scores for the ML scalping engine.

Data sources (in order of preference)
-------------------------------------
1. CryptoPanic free API - used when the CRYPTOPANIC_API_KEY environment
   variable is set. One batched request covers every base asset, and the
   API itself attributes each headline to its coins.
2. Public RSS feeds (CoinDesk, Cointelegraph) - fallback used when no API
   key is configured or the API call fails. Headlines are attributed to
   coins by symbol / name mention.
3. Neutral default - if every source fails (or a coin simply has no news)
   the coin's score defaults to 0.0. The sentiment stage never raises and
   never crashes the live bot.

Scoring
-------
Each headline title is scored in [-1, 1] with a hand-tuned keyword lexicon
(no external NLP dependencies). Matching uses STRICT word boundaries:

    r"\bban\b"  matches "ban"  but NOT "bank", "banking", "bankrupt"
    r"\bbuy\b"  matches "buy"  but NOT "buying" (listed separately)
    r"\bhack\b" matches "hack" but NOT "hacker" (listed separately)

There is deliberately NO stemming: every inflection that appears in real
headlines is its own lexicon entry, which keeps scoring predictable and
dependency-free. A coin's score is the mean of its attributed title
scores, clipped to [-1, 1].

Usage (standalone smoke test):
    python sentiment_analyzer.py

Usage (from other modules):
    from sentiment_analyzer import get_sentiment_scores
    scores = await get_sentiment_scores(symbols)   # {"BTC/USDT": 0.42, ...}
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Tuple

import aiohttp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CRYPTOPANIC_API_URL = "https://cryptopanic.com/api/free/v1/posts/"
CRYPTOPANIC_KEY_ENV = "CRYPTOPANIC_API_KEY"

RSS_FEEDS = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)
MAX_HEADLINES = 100          # cap per source
SENTIMENT_THRESHOLD = 0.15   # |score| below this is reported as "neutral"

# Base asset -> name tokens used by the RSS fallback for attribution.
COIN_NAMES = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
    "BNB": "binance", "ADA": "cardano", "DOGE": "dogecoin", "AVAX": "avalanche",
    "DOT": "polkadot", "LINK": "chainlink", "MATIC": "polygon", "TRX": "tron",
    "LTC": "litecoin", "BCH": "bitcoin cash", "NEAR": "near protocol",
    "ATOM": "cosmos", "APT": "aptos", "ARB": "arbitrum", "OP": "optimism",
    "FIL": "filecoin", "INJ": "injective", "SUI": "sui", "PEPE": "pepe",
    "SHIB": "shiba", "UNI": "uniswap", "ETC": "ethereum classic",
    "XLM": "stellar", "HBAR": "hedera", "VET": "vechain",
    "ICP": "internet computer", "RNDR": "render", "TIA": "celestia",
    "RUNE": "thorchain", "AAVE": "aave", "ALGO": "algorand", "FTM": "fantom",
}

# ---------------------------------------------------------------------------
# Keyword lexicons
# ---------------------------------------------------------------------------
# Every keyword is compiled into r"\b<keyword>\b" (strict word boundaries),
# so partial-word collisions are impossible:
#     "ban"    will NOT fire on "bank", "banking", "bankrupt"
#     "buy"    will NOT fire on "buying"  (listed separately)
#     "hack"   will NOT fire on "hacker"  (listed separately)
# No stemming is performed - each inflection used in real headlines is its
# own entry. Weights are additive within one headline, then clipped.
# ---------------------------------------------------------------------------

BULLISH_KEYWORDS: Dict[str, float] = {
    # strong positive
    "all-time high": 1.0, "record high": 0.9, "new high": 0.6, "ath": 0.6,
    "skyrocket": 0.9, "surge": 0.9, "surges": 0.9, "surged": 0.8,
    "soar": 0.9, "soars": 0.9, "soared": 0.8, "rally": 0.8, "rallies": 0.8,
    "bullish": 0.9, "bull run": 0.8, "breakout": 0.8, "short squeeze": 0.6,
    # moderate positive
    "moon": 0.7, "mooning": 0.7, "pump": 0.6, "pumps": 0.5, "pumping": 0.5,
    "spike": 0.4, "spikes": 0.5, "jump": 0.5, "jumps": 0.6, "jumped": 0.5,
    "climb": 0.4, "climbs": 0.5, "climbing": 0.4, "gain": 0.4, "gains": 0.5,
    "gained": 0.4, "rise": 0.3, "rises": 0.4, "rising": 0.3, "rose": 0.3,
    "rebound": 0.5, "rebounds": 0.5, "recover": 0.4, "recovers": 0.5,
    "recovery": 0.4, "bounce": 0.4, "uptrend": 0.4, "golden cross": 0.4,
    # news / fundamentals
    "approval": 0.8, "approved": 0.8, "approves": 0.6, "green light": 0.6,
    "partnership": 0.6, "partners": 0.4, "adoption": 0.6, "adopts": 0.4,
    "integration": 0.4, "integrates": 0.4, "upgrade": 0.5, "upgrades": 0.4,
    "upgraded": 0.4, "launch": 0.4, "launches": 0.4, "launched": 0.3,
    "inflow": 0.5, "inflows": 0.5, "accumulation": 0.4, "accumulate": 0.4,
    "buy the dip": 0.5, "buy": 0.3, "buys": 0.3, "buying": 0.3, "bought": 0.3,
    "milestone": 0.4, "surpass": 0.5, "surpasses": 0.5, "bull": 0.5, "bulls": 0.4,
}

BEARISH_KEYWORDS: Dict[str, float] = {
    # strong negative
    "crash": -1.0, "crashes": -1.0, "crashed": -0.9, "plunge": -1.0,
    "plunges": -1.0, "plunged": -0.9, "plummet": -1.0, "plummets": -1.0,
    "bankrupt": -1.0, "bankruptcy": -1.0, "insolvency": -0.9, "insolvent": -0.9,
    "rug pull": -1.0, "rugpull": -1.0, "hack": -1.0, "hacked": -1.0,
    "hacks": -0.8, "hacking": -0.8, "hacker": -0.7, "exploit": -0.9,
    "exploited": -0.9, "fraud": -0.9, "ponzi": -0.9, "scam": -0.9,
    # moderate negative
    "bearish": -0.9, "sec charges": -0.9, "sues": -0.8, "sued": -0.7,
    "lawsuit": -0.7, "lawsuits": -0.7, "charged": -0.6, "indicted": -0.8,
    "dump": -0.8, "dumps": -0.7, "dumped": -0.7, "dumping": -0.7,
    "sell-off": -0.8, "selloff": -0.8, "sell off": -0.8, "selling": -0.4,
    "ban": -0.8, "bans": -0.8, "banned": -0.8, "banning": -0.7,
    "delist": -0.8, "delisted": -0.8, "delisting": -0.8,
    "breach": -0.8, "breached": -0.7, "liquidation": -0.6,
    "liquidations": -0.6, "liquidated": -0.7,
    # mild negative
    "slump": -0.6, "slumps": -0.6, "slumped": -0.5, "tumble": -0.6,
    "tumbles": -0.6, "tumbled": -0.5, "outflow": -0.5, "outflows": -0.5,
    "probe": -0.5, "probes": -0.4, "investigation": -0.5,
    "investigating": -0.4, "fud": -0.5, "downgrade": -0.6, "downgraded": -0.6,
    "reject": -0.6, "rejects": -0.6, "rejected": -0.6, "rejection": -0.5,
    "bear": -0.5, "bears": -0.4, "losses": -0.5, "loss": -0.4,
    "drop": -0.4, "drops": -0.4, "dropped": -0.4, "falls": -0.4,
    "fell": -0.4, "falling": -0.3, "decline": -0.4, "declines": -0.4,
    "warning": -0.4, "warns": -0.4, "warned": -0.3, "risk": -0.3,
    "risks": -0.3, "risky": -0.4, "delay": -0.3, "delayed": -0.3,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("sentiment_analyzer")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _compile_lexicon(lexicon: Dict[str, float]) -> List[Tuple[re.Pattern, float]]:
    r"""
    Pre-compile the keyword lexicon into STRICT word-boundary patterns.

    \b...\b guarantees e.g. 'ban' never matches inside 'bank' or
    'bankruptcy', and 'buy' never matches inside 'buying'.
    """
    return [
        (re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE), weight)
        for keyword, weight in lexicon.items()
    ]


_BULLISH_PATTERNS = _compile_lexicon(BULLISH_KEYWORDS)
_BEARISH_PATTERNS = _compile_lexicon(BEARISH_KEYWORDS)


def _base_asset(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC'; strips exchange lot-size prefixes like '1000PEPE'."""
    base = symbol.split("/")[0].strip().upper()
    for prefix in ("1000000", "100000", "10000", "1000", "1M"):
        if base.startswith(prefix) and len(base) > len(prefix):
            return base[len(prefix):]
    return base


def score_headline(title: str) -> float:
    """Score a single headline title in [-1, 1] using the keyword lexicon."""
    score = 0.0
    for pattern, weight in _BULLISH_PATTERNS:
        if pattern.search(title):
            score += weight
    for pattern, weight in _BEARISH_PATTERNS:
        if pattern.search(title):
            score += weight
    return max(-1.0, min(1.0, score))


def _aggregate(scores: Iterable[float]) -> float:
    """Mean of title scores, clipped to [-1, 1]; 0.0 when empty."""
    values = list(scores)
    if not values:
        return 0.0
    return round(max(-1.0, min(1.0, sum(values) / len(values))), 3)


def describe_sentiment(score: float) -> str:
    """Human-readable label for a sentiment score."""
    if score >= SENTIMENT_THRESHOLD:
        return "bullish"
    if score <= -SENTIMENT_THRESHOLD:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Source 1: CryptoPanic free API (async title fetcher)
# ---------------------------------------------------------------------------

async def fetch_cryptopanic_titles(
    session: aiohttp.ClientSession, bases: List[str], api_key: str
) -> Dict[str, List[str]]:
    """
    Fetch recent post titles from the CryptoPanic free API.

    Returns {base_asset: [title, ...]}. The API attributes each post to its
    currencies, so no keyword-based attribution is needed on this path.
    """
    params = {
        "auth_token": api_key,
        "currencies": ",".join(bases),
        "public": "true",
    }
    async with session.get(CRYPTOPANIC_API_URL, params=params) as resp:
        resp.raise_for_status()
        payload = await resp.json()

    titles: Dict[str, List[str]] = {}
    for post in payload.get("results", [])[:MAX_HEADLINES]:
        title = (post.get("title") or "").strip()
        if not title:
            continue
        for currency in post.get("currencies") or []:
            code = (currency.get("code") or "").upper()
            if code:
                titles.setdefault(code, []).append(title)

    logger.info(
        "CryptoPanic: %d titles fetched for %d coins",
        sum(len(v) for v in titles.values()), len(titles),
    )
    return titles


# ---------------------------------------------------------------------------
# Source 2: RSS fallback (async title fetcher + attribution)
# ---------------------------------------------------------------------------

async def fetch_rss_titles(session: aiohttp.ClientSession) -> List[str]:
    """Fetch recent headlines from the fallback RSS feeds."""
    titles: List[str] = []
    for url in RSS_FEEDS:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                text = await resp.text()
            root = ET.fromstring(text)
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                if title:
                    titles.append(title)
        except Exception as exc:
            logger.warning("RSS feed unavailable (%s): %s", url, exc)
    logger.info("RSS fallback: %d titles fetched", len(titles))
    return titles[: MAX_HEADLINES * len(RSS_FEEDS)]


def _attribute_rss(
    titles: List[str], symbols: List[str]
) -> Dict[str, List[str]]:
    """Attribute RSS titles to base assets by symbol / name mention."""
    per_coin: Dict[str, List[str]] = {}
    for symbol in symbols:
        base = _base_asset(symbol)
        tokens = {base.lower()}
        tokens.update(t for t in COIN_NAMES.get(base, "").lower().split() if t)
        for title in titles:
            if any(re.search(rf"\b{re.escape(t)}\b", title, re.IGNORECASE) for t in tokens):
                per_coin.setdefault(base, []).append(title)
    return per_coin


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_sentiment_scores(symbols: List[str]) -> Dict[str, float]:
    """
    Sentiment score in [-1, 1] for every symbol.

    Tries the CryptoPanic free API (when CRYPTOPANIC_API_KEY is set), then
    the RSS fallback, and finally defaults every coin to 0.0 (neutral).
    This function never raises.
    """
    neutral = {symbol: 0.0 for symbol in symbols}
    if not symbols:
        return neutral

    bases = sorted({_base_asset(s) for s in symbols})
    api_key = os.getenv(CRYPTOPANIC_KEY_ENV, "").strip()
    titles_per_coin: Dict[str, List[str]] = {}
    source = "no data (neutral default)"

    if api_key:
        try:
            async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
                titles_per_coin = await fetch_cryptopanic_titles(session, bases, api_key)
            if titles_per_coin:
                source = "CryptoPanic API"
        except Exception as exc:
            logger.warning("CryptoPanic API failed (%s) - trying RSS fallback", exc)
    else:
        logger.warning(
            "%s not set - using RSS fallback (set the env var for better scores).",
            CRYPTOPANIC_KEY_ENV,
        )

    if not titles_per_coin:
        try:
            async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
                titles = await fetch_rss_titles(session)
            titles_per_coin = _attribute_rss(titles, symbols)
            if titles_per_coin:
                source = "RSS fallback"
        except Exception as exc:
            logger.warning("RSS fallback failed (%s) - defaulting to neutral.", exc)

    scores = {
        symbol: _aggregate(
            score_headline(t) for t in titles_per_coin.get(_base_asset(symbol), [])
        )
        for symbol in symbols
    }
    logger.info("Sentiment source: %s", source)
    logger.info("Sentiment scores: %s", {s: f"{v:+.2f}" for s, v in scores.items()})
    return scores


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

async def _main() -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    scores = await get_sentiment_scores(symbols)

    print("\n" + "=" * 62)
    print("TASK 5 SMOKE TEST - NEWS SENTIMENT")
    print("=" * 62)
    for symbol, score in scores.items():
        bar = "#" * int(round(abs(score) * 10))
        print(
            f"{symbol:<12} {score:+5.2f}  {bar:<10}  {describe_sentiment(score)}"
        )
    print("=" * 62)
    key = os.getenv(CRYPTOPANIC_KEY_ENV, "").strip()
    print(
        f"API key: {'configured (CryptoPanic API)' if key else 'NOT configured (RSS fallback / neutral)'}"
    )


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
