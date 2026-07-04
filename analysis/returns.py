"""
Return computation: raw returns, benchmark-adjusted ("abnormal") returns,
an optional market-model (beta-adjusted) variant, and winsorization.

This module is the methodological heart of the upgrade over the
original coursework script. The original computed a single "daily
return" as (price ~2 days after publish - price ~2 days before
publish) / price before, with no separation between how much of that
move was market-wide vs. company-specific, and no defined event-day
convention. Both issues are addressed here.

---------------------------------------------------------------------
Why "abnormal" return instead of raw return
---------------------------------------------------------------------
A firm-specific news article should, in an efficient market, move the
stock relative to *how the rest of the market moved that day* -- not in
isolation. If the whole market rallies 2% on macro news the same day a
company publishes a mildly positive article, the raw return will show
a large positive number that has almost nothing to do with the
article's content. Regressing sentiment against raw returns therefore
mixes two very different sources of variation: market-wide beta
exposure (noise, for our purposes) and idiosyncratic, firm-specific
reaction to news (the signal we actually want to test).

The original coursework report's R^2 of ~0.02 is very plausibly
depressed by exactly this contamination -- most of the variance in
daily returns for a FTSE 100 constituent on any given day is explained
by the market factor, not firm-specific news. Stripping out the
market-wide component (via abnormal returns) should, if anything, give
sentiment a fairer chance to show a relationship, precisely because
we've removed a huge source of unrelated variance from the y-variable.

---------------------------------------------------------------------
Event-day alignment
---------------------------------------------------------------------
News can be published outside market hours, on a weekend, or on a
holiday. We do not have precise per-exchange trading-hours data, so we
use a standard, defensible convention from event-study methodology:
the "event day" (day 0) is the first trading day on or after the
article's publish date. This means a Friday-evening article and a
Saturday article both get anchored to the following Monday's session --
the first opportunity the market had to react.

---------------------------------------------------------------------
Window definitions (relative to event day, trading days not calendar days)
---------------------------------------------------------------------
  same_day : return from the close *before* day 0 to the close *on*
             day 0. Captures the reaction during the session in which
             the news was first tradeable.
  next_day : return from day 0's close to day 1's close. Captures
             next-session drift/reaction (news can be digested with a
             lag, especially for less-followed names).
  3day     : cumulative return from day 0's close to day 3's close.
             Captures a short post-event drift window without reaching
             so far out that unrelated news dominates the window.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils.constants import infer_benchmark_for_ticker

WINDOWS = ["same_day", "next_day", "3day"]
WINDOW_LABELS = {
    "same_day": "Same-day (close-to-close)",
    "next_day": "Next-day",
    "3day": "3-day cumulative",
}

# Trailing window (in trading days) used to estimate beta for the
# optional market-model adjustment. 60 trading days (~3 months) balances
# having enough observations for a stable OLS beta estimate against
# staying reasonably responsive to a stock's *current* risk profile
# rather than averaging over a year of possibly-stale beta.
DEFAULT_BETA_WINDOW = 60


def _find_event_position(index: pd.DatetimeIndex, pub_date: pd.Timestamp) -> int | None:
    """Return the integer position of the first trading day >= pub_date.

    Returns None if the publish date is after the last available price
    (i.e. we don't have enough forward price data yet, e.g. very recent
    news) so callers can skip that article rather than fabricate a
    return.
    """
    pub_date_naive = pd.Timestamp(pub_date).tz_localize(None) if pd.Timestamp(pub_date).tzinfo else pd.Timestamp(pub_date)
    pub_day = pub_date_naive.normalize()
    positions = index.searchsorted(pub_day, side="left")
    if positions >= len(index):
        return None
    return int(positions)


def _window_return(closes: np.ndarray, event_pos: int, window: str) -> float:
    """Compute the close-to-close return for one window, given the
    integer position of the event day within a sorted close-price array.

    Returns NaN if the required indices fall outside the available
    price history (e.g. event day is the very first day we have data
    for, so there's no "day before" to anchor the same-day return to).
    """
    n = len(closes)
    try:
        if window == "same_day":
            if event_pos - 1 < 0:
                return np.nan
            return closes[event_pos] / closes[event_pos - 1] - 1.0
        elif window == "next_day":
            if event_pos + 1 >= n:
                return np.nan
            return closes[event_pos + 1] / closes[event_pos] - 1.0
        elif window == "3day":
            if event_pos + 3 >= n:
                return np.nan
            return closes[event_pos + 3] / closes[event_pos] - 1.0
        else:
            raise ValueError(f"Unknown window '{window}'")
    except IndexError:
        return np.nan


def _estimate_trailing_beta(stock_closes: pd.Series, bench_closes: pd.Series,
                             event_pos: int, bench_pos: int, window: int = DEFAULT_BETA_WINDOW) -> float:
    """Estimate beta via OLS of the stock's daily returns on the
    benchmark's daily returns, using only data *before* the event day.

    This is the classic "market model" from event-study methodology
    (MacKinlay, 1997): instead of assuming every stock moves 1-for-1
    with the index (which flat benchmark-subtraction implicitly does),
    we estimate how sensitive *this specific stock* actually is to
    market moves, then remove beta * market_return rather than a flat
    1.0 * market_return. A high-beta stock (e.g. a cyclical miner) gets
    more of its move attributed to "the market", leaving a smaller,
    more accurate idiosyncratic residual; a low-beta stock (e.g. a
    utility) gets less.

    Using only pre-event data avoids lookahead bias in the beta
    estimate itself -- we should not use information from after the
    event to decide how much of the event's own return was "market".

    Trade-off vs. simple subtraction: this is more statistically
    faithful but adds estimation noise (beta is a sample estimate, not
    a known constant) and requires enough trailing history to fit --
    for recently-listed stocks or very short price histories, we fall
    back to beta=1 (equivalent to flat subtraction).

    Note: the stock and benchmark are sliced using their *own*
    positions (`event_pos` and `bench_pos` respectively), not a shared
    position index -- the two price series can have different lengths
    even over the "same" calendar window (a holiday specific to one
    exchange, a stock with a shorter listing history, etc.), so reusing
    one series' integer position to slice the other would silently pull
    the wrong dates. The final `pd.concat(...).dropna()` below then
    aligns the two return series by actual date, not by position.
    """
    stock_start = max(0, event_pos - window)
    bench_start = max(0, bench_pos - window)
    if event_pos - stock_start < 20 or bench_pos - bench_start < 20:  # too few observations for a stable estimate
        return 1.0
    stock_ret = stock_closes.iloc[stock_start:event_pos].pct_change().dropna()
    bench_ret = bench_closes.iloc[bench_start:bench_pos].pct_change().dropna()
    aligned = pd.concat([stock_ret, bench_ret], axis=1, keys=["stock", "bench"]).dropna()
    if len(aligned) < 20:
        return 1.0
    X = sm.add_constant(aligned["bench"])
    try:
        model = sm.OLS(aligned["stock"], X).fit()
        beta = model.params.get("bench", 1.0)
        # Guard against a degenerate/extreme beta estimate (e.g. from a
        # near-constant benchmark window) distorting the abnormal return
        # more than a flat subtraction would.
        return float(np.clip(beta, -3.0, 5.0))
    except Exception:
        return 1.0


@dataclass
class ReturnComputationStats:
    """Bookkeeping for how many articles got usable returns vs. were
    skipped, so the UI can be transparent about data loss rather than
    silently returning a smaller-than-expected dataset.
    """
    total: int = 0
    skipped_no_price_data: int = 0
    skipped_insufficient_window: int = 0


def compute_returns(
    articles_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    use_market_model: bool = False,
    beta_window: int = DEFAULT_BETA_WINDOW,
) -> tuple[pd.DataFrame, ReturnComputationStats]:
    """Attach raw, benchmark, and abnormal returns (for all three
    windows) to every article.

    Adds columns:
      raw_return_{window}        : simple stock return over the window
      benchmark_return_{window}  : benchmark index return over the same
                                    window (same tickers' inferred index)
      abnormal_return_{window}   : raw - benchmark (or, if
                                    use_market_model, raw - beta*benchmark)
      beta_used                  : the beta applied (1.0 under flat
                                    subtraction, estimated otherwise) --
                                    surfaced for transparency
      event_date                 : the trading day the article's return
                                    windows are anchored to

    Articles whose ticker has no usable price series, or whose windows
    fall outside available price history, are dropped -- with counts
    tracked in the returned stats object rather than silently vanishing.
    """
    stats = ReturnComputationStats(total=len(articles_df))
    rows = []

    for _, article in articles_df.iterrows():
        ticker = article["ticker"]
        stock_df = price_data.get(ticker)
        bench_symbol = infer_benchmark_for_ticker(ticker)
        bench_df = price_data.get(bench_symbol)

        if stock_df is None or stock_df.empty or bench_df is None or bench_df.empty:
            stats.skipped_no_price_data += 1
            continue

        stock_closes = stock_df["Close"]
        event_pos = _find_event_position(stock_closes.index, article["publish_date"])
        if event_pos is None:
            stats.skipped_insufficient_window += 1
            continue

        # Align the benchmark series to the same event position by
        # looking up the nearest trading day on/after the same calendar
        # date -- the stock and benchmark don't necessarily share an
        # index object even though both trade on (near-)identical
        # calendars.
        event_date = stock_closes.index[event_pos]
        bench_pos = _find_event_position(bench_df.index, event_date)
        if bench_pos is None:
            stats.skipped_insufficient_window += 1
            continue

        row = {**article.to_dict(), "event_date": event_date}
        any_window_ok = False

        beta = 1.0
        if use_market_model:
            beta = _estimate_trailing_beta(stock_closes, bench_df["Close"], event_pos, bench_pos, beta_window)
        row["beta_used"] = beta

        for window in WINDOWS:
            raw_r = _window_return(stock_closes.values, event_pos, window)
            bench_r = _window_return(bench_df["Close"].values, bench_pos, window)
            row[f"raw_return_{window}"] = raw_r
            row[f"benchmark_return_{window}"] = bench_r
            if np.isnan(raw_r) or np.isnan(bench_r):
                row[f"abnormal_return_{window}"] = np.nan
            else:
                row[f"abnormal_return_{window}"] = raw_r - beta * bench_r
                any_window_ok = True

        if not any_window_ok:
            stats.skipped_insufficient_window += 1
            continue

        rows.append(row)

    if not rows:
        return pd.DataFrame(), stats
    return pd.DataFrame(rows), stats


def winsorize_series(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """Cap extreme values at the given percentiles rather than deleting them.

    ---------------------------------------------------------------
    Why winsorize instead of deleting outlier rows (as the original
    `df.drop(80)` did)
    ---------------------------------------------------------------
    Deleting a specific row by hardcoded index is not reproducible (it
    depends on the exact data pulled that run), not documented (a
    reader has no way to know *why* row 80 was removed), and throws
    away real information -- an extreme return might be a genuine,
    interesting data point (e.g. an earnings surprise) rather than a
    data error.

    Winsorizing instead caps values beyond the given percentile
    thresholds to the threshold value itself. This keeps every
    observation in the dataset (preserving sample size, which matters
    a lot when the whole dataset might only be ~100-200 articles) while
    preventing a single extreme observation (e.g. a stock that dropped
    40% on unrelated M&A news) from dominating an OLS fit, since OLS
    minimizes *squared* error and is therefore highly sensitive to
    extreme values.

    The trade-off: winsorizing is still a judgment call about what
    counts as "extreme" (here, the 1st/99th percentile by default,
    user-adjustable), and it does mechanically shrink the variance of
    the tails. We surface this openly in the UI rather than hiding it,
    and always offer the option to run on raw, unwinsorized data
    instead.
    """
    if series.empty or series.dropna().empty:
        return series
    lower_bound = series.quantile(lower_pct)
    upper_bound = series.quantile(upper_pct)
    return series.clip(lower=lower_bound, upper=upper_bound)


def apply_winsorization(df: pd.DataFrame, columns: list[str], lower_pct: float = 0.01,
                         upper_pct: float = 0.99) -> pd.DataFrame:
    """Return a copy of `df` with winsorized versions of `columns` added
    as new `{col}_wz` columns, leaving the original raw columns intact
    so the UI can toggle between them without re-fetching anything.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_wz"] = winsorize_series(out[col], lower_pct, upper_pct)
    return out
