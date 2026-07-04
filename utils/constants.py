"""
Shared constants: index constituent locations, benchmark mapping, and
app-wide limits.

Centralising these here means every module (data fetchers, analysis,
UI) agrees on the same tickers, the same benchmark for a given market,
and the same rate-limit guardrails -- there's a single place to tune
them if yfinance's behaviour changes or a new index is added.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
FTSE100_CSV = DATA_DIR / "ftse100_constituents.csv"
SP500_CSV = DATA_DIR / "sp500_constituents.csv"

INDEX_FILES = {
    "FTSE100": FTSE100_CSV,
    "S&P500": SP500_CSV,
}

# ---------------------------------------------------------------------------
# Benchmark mapping
# ---------------------------------------------------------------------------
# Abnormal returns require a market benchmark to net out systematic
# (market-wide) moves from a stock's raw return -- see analysis/returns.py
# for the full rationale. We pick the benchmark by looking at the ticker's
# exchange suffix: LSE-listed names (".L") are benchmarked against the
# FTSE 100 index, everything else defaults to the S&P 500. This is a
# heuristic, not a perfect classification (e.g. it would mis-benchmark a
# European stock with no ".L" suffix), so the UI allows a manual override.
INDEX_BENCHMARKS = {
    "FTSE100": "^FTSE",
    "S&P500": "^GSPC",
}

DEFAULT_BENCHMARK = "^GSPC"


def infer_benchmark_for_ticker(ticker: str) -> str:
    """Guess the appropriate market benchmark index for a given ticker.

    Uses the Yahoo Finance exchange-suffix convention (e.g. '.L' for
    London Stock Exchange listings) as a cheap proxy for "which market
    is this stock's systematic risk driven by". This is intentionally
    simple: a rigorous approach would look up the primary exchange via
    an API call, but that adds latency and another point of failure for
    a heuristic that only needs to be roughly right.
    """
    ticker = ticker.upper().strip()
    if ticker.endswith(".L"):
        return INDEX_BENCHMARKS["FTSE100"]
    return DEFAULT_BENCHMARK


# ---------------------------------------------------------------------------
# App-wide limits (protect against hammering yfinance / long run times)
# ---------------------------------------------------------------------------
MAX_TICKERS = 25            # hard cap on total tickers processed in one run
DEFAULT_PRESET_SAMPLE = 12  # default number of tickers pre-selected from an index
MAX_ARTICLES_PER_TICKER = 25
DEFAULT_ARTICLES_PER_TICKER = 10
LOOKBACK_OPTIONS = [30, 60, 90]
DEFAULT_LOOKBACK_DAYS = 60

# Trading-day approximations used for annualising backtest statistics.
TRADING_DAYS_PER_YEAR = 252

# A conservative approximate risk-free rate proxy (annualised) used only
# for Sharpe-ratio calculation when no live risk-free series is fetched.
# This is a simplification flagged explicitly in the UI -- see the
# Limitations tab -- rather than pulling a T-bill yield series, which
# would add another fragile external dependency for a small illustrative
# backtest.
APPROX_RISK_FREE_RATE = 0.04
