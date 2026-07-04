"""
Price data retrieval via yfinance, including benchmark index alignment.

We fetch one padded daily OHLC history per ticker/benchmark (rather than
one request per article) because:
  1. It's dramatically fewer API calls -- one per unique symbol instead
     of one per article -- which matters a lot once a user has selected
     15-25 tickers with 10+ articles each.
  2. Return-window calculations (same-day, next-day, 3-day) need to look
     both backwards and forwards from an article's publish date, so we
     need a contiguous price series to slice from, not point lookups.
"""

from datetime import timedelta

import pandas as pd
import yfinance as yf

from utils.caching import cache_prices
from utils.constants import infer_benchmark_for_ticker

# Extra calendar days padded onto both ends of the requested window so
# that return-window slicing (e.g. "3 trading days after" near the start
# or end of the range) always has enough surrounding data to work with,
# even across weekends/holidays.
PAD_DAYS = 10


@cache_prices
def fetch_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLC history for a single symbol (stock or index).

    `start`/`end` are passed as ISO date strings rather than
    datetime/Timestamp objects because Streamlit's cache keys arguments
    by value/hash, and plain strings hash more predictably across
    reruns than datetime objects with potentially differing tz info.

    auto_adjust=True means Close is already split/dividend-adjusted --
    important because we're computing returns, and an unadjusted price
    series would show a fake "return" on every ex-dividend or split
    date that has nothing to do with news sentiment.
    """
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if hist.empty:
        return pd.DataFrame()
    hist = hist.tz_localize(None) if hist.index.tz is not None else hist
    return hist[["Open", "High", "Low", "Close", "Volume"]]


def fetch_prices_for_universe(
    tickers: list[str],
    lookback_days: int,
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    """Fetch padded price history for every ticker plus every benchmark
    that ticker set requires.

    Returns a dict keyed by symbol (both stock tickers and benchmark
    index tickers like '^FTSE' live in the same dict) so returns.py can
    look up either kind of series uniformly.
    """
    end = pd.Timestamp.now().normalize() + timedelta(days=1)
    start = end - timedelta(days=lookback_days + PAD_DAYS + 5)
    # +5 trading-day buffer beyond the pad so a 3-day-forward return on
    # an article published near "today" still has price data to resolve
    # against once the market has actually moved that far forward.
    end_padded = end + timedelta(days=PAD_DAYS)

    start_str, end_str = start.strftime("%Y-%m-%d"), end_padded.strftime("%Y-%m-%d")

    benchmarks_needed = {infer_benchmark_for_ticker(t) for t in tickers}
    symbols = list(dict.fromkeys(list(tickers) + list(benchmarks_needed)))

    prices = {}
    for i, symbol in enumerate(symbols):
        prices[symbol] = fetch_price_history(symbol, start_str, end_str)
        if progress_callback:
            progress_callback(i + 1, len(symbols), symbol)
    return prices
