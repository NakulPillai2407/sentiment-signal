"""
A simple, fully transparent sentiment-threshold long/short backtest.

This is deliberately NOT a full backtesting framework (no order book
simulation, no transaction costs, no position sizing beyond equal
weight, no intraday fills). The goal is to answer one narrow question
as honestly as possible: "if you had mechanically traded the sign of
the sentiment signal, would you have made money over this specific,
small sample?" -- and to make every assumption behind that answer
visible rather than buried inside a library.

=======================================================================
READ THIS BEFORE TRUSTING ANY NUMBER FROM THIS MODULE
=======================================================================
This backtest is illustrative only. It has no transaction costs, no
slippage, no bid-ask spread, no market-impact modelling, no borrow cost
for the short leg, and (most importantly) is very likely run over a
sample of well under a few hundred trades across a handful of weeks.
Sharpe ratios and annualised returns computed from such short, noisy
samples are extremely unstable -- a single lucky or unlucky trade can
swing them enormously. Nothing here should be read as evidence of a
tradeable strategy. See the Limitations & Methodology Notes tab for the
full discussion.
=======================================================================

Mechanics:
  - Each article with a sentiment score above `long_threshold` generates
    a hypothetical long position in that ticker, held for the chosen
    return window (next-day or 3-day).
  - Each article with a sentiment score below `short_threshold`
    (a negative number) generates a hypothetical short position over
    the same window.
  - Articles with sentiment between the thresholds generate no trade.
  - We use *raw* stock returns (not abnormal/benchmark-adjusted returns)
    for position P&L, because abnormal return is an analytical
    construct for isolating firm-specific news effect in a regression --
    it is not a return an investor can actually capture by trading a
    single stock. A real long/short position's P&L is the raw price
    return (times signal direction), full stop.
  - On days with multiple signalled trades (e.g. two different tickers
    both had qualifying articles the same event day), the daily
    strategy return is the equal-weighted average of that day's
    position returns -- consistent with the equal-weighted buy-and-hold
    benchmark used for comparison.
  - On days with no qualifying signal, the strategy is flat (0% return,
    i.e. sitting in cash) -- we do not carry over old positions past
    their defined holding window.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.constants import APPROX_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

BACKTEST_DISCLAIMER = (
    "This is a simplified, illustrative backtest: no transaction costs, no slippage, "
    "no borrow costs on shorts, and typically a small sample of trades over a short "
    "window. It is NOT a validated trading strategy and should not inform real trading "
    "decisions."
)


@dataclass
class BacktestResult:
    strategy_returns: pd.Series      # per-event-day strategy returns
    benchmark_returns: pd.Series     # per-day equal-weighted buy-and-hold returns
    cumulative_strategy: pd.Series
    cumulative_benchmark: pd.Series
    n_trades: int
    n_long: int
    n_short: int
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    error: str | None = None


def _empty_backtest(error: str) -> BacktestResult:
    empty = pd.Series(dtype="float64")
    return BacktestResult(
        strategy_returns=empty, benchmark_returns=empty, cumulative_strategy=empty,
        cumulative_benchmark=empty, n_trades=0, n_long=0, n_short=0, total_return=np.nan,
        annualized_return=np.nan, annualized_vol=np.nan, sharpe_ratio=np.nan,
        max_drawdown=np.nan, hit_rate=np.nan, error=error,
    )


def _max_drawdown(cumulative: pd.Series) -> float:
    """Largest peak-to-trough decline in cumulative (1 + return) terms.

    Computed as running-max minus current value, divided by running-max,
    expressed as a negative number (e.g. -0.15 = a 15% drawdown at the
    worst point).
    """
    if cumulative.empty:
        return np.nan
    wealth = 1.0 + cumulative
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return float(drawdown.min())


def run_backtest(
    articles_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    sentiment_col: str,
    window: str,
    long_threshold: float,
    short_threshold: float,
) -> BacktestResult:
    """Run the sentiment-threshold long/short backtest.

    `articles_df` must already have return columns from
    analysis/returns.py (raw_return_{window}, event_date) and a
    sentiment column (e.g. 'finbert_score' or 'vader_compound').
    `long_threshold` / `short_threshold` are on the same scale as the
    sentiment column (both models here are roughly bounded in [-1, 1]).
    """
    return_col = f"raw_return_{window}"
    if return_col not in articles_df.columns or sentiment_col not in articles_df.columns:
        return _empty_backtest("Required return/sentiment columns not found -- run the returns pipeline first.")

    df = articles_df.dropna(subset=[return_col, sentiment_col, "event_date", "ticker"]).copy()
    if df.empty:
        return _empty_backtest("No articles with valid returns and sentiment scores to backtest.")

    df["signal"] = 0
    df.loc[df[sentiment_col] >= long_threshold, "signal"] = 1
    df.loc[df[sentiment_col] <= short_threshold, "signal"] = -1
    trades = df[df["signal"] != 0].copy()

    if trades.empty:
        return _empty_backtest(
            "No trades generated at the current thresholds -- try widening the long/short bands."
        )

    trades["position_return"] = trades["signal"] * trades[return_col]

    # Multiple qualifying articles for the same ticker on the same event
    # day collapse to one trade (average their signal-weighted return) --
    # we don't want one noisy headline to be double-counted just because
    # a second article about the same stock happened to publish the same
    # day.
    per_ticker_day = (
        trades.groupby(["event_date", "ticker"])
        .agg(position_return=("position_return", "mean"), signal=("signal", "mean"))
        .reset_index()
    )

    # Equal-weighted daily strategy return: average across every ticker
    # with an open signal that day. Days with no signals are implicitly
    # flat (0% return, sitting in cash) once reindexed below.
    strategy_daily = per_ticker_day.groupby("event_date")["position_return"].mean().sort_index()

    # Equal-weighted buy-and-hold benchmark across the whole selected
    # universe, using every stock ticker we have price history for --
    # not just tickers that happened to generate a signal for *this*
    # holding window. Deriving this from `df` (which is already
    # filtered to articles with a usable return for the chosen window)
    # would make the "buy and hold the universe" benchmark shift every
    # time the user toggles the holding window, even though buy-and-hold
    # has nothing to do with article timing -- it should be identical
    # regardless of that choice. Benchmark index symbols (e.g. '^GSPC',
    # '^FTSE') are excluded since they aren't part of the tradeable
    # universe the user selected.
    tickers_in_universe = [
        t for t in price_data if not t.startswith("^") and not price_data[t].empty
    ]
    daily_returns_by_ticker = {}
    for t in tickers_in_universe:
        closes = price_data[t]["Close"]
        daily_returns_by_ticker[t] = closes.pct_change().dropna()
    if daily_returns_by_ticker:
        combined = pd.concat(daily_returns_by_ticker, axis=1)
        benchmark_daily = combined.mean(axis=1, skipna=True).dropna()
    else:
        benchmark_daily = pd.Series(dtype="float64")

    # Reindex the strategy series over the full benchmark trading
    # calendar so "no signal that day" correctly shows up as a flat 0%
    # contribution when compounding, rather than being skipped entirely
    # (which would silently compress the timeline).
    full_index = benchmark_daily.index.union(strategy_daily.index).sort_values()
    strategy_full = strategy_daily.reindex(full_index, fill_value=0.0)
    benchmark_full = benchmark_daily.reindex(full_index, fill_value=0.0)

    cumulative_strategy = (1.0 + strategy_full).cumprod() - 1.0
    cumulative_benchmark = (1.0 + benchmark_full).cumprod() - 1.0

    total_return = float(cumulative_strategy.iloc[-1]) if not cumulative_strategy.empty else np.nan

    n_periods = len(strategy_full)
    # Annualisation here is a rough approximation, not a precise
    # trading-calendar calculation: we scale by the number of periods
    # actually observed relative to a 252-trading-day year. With a
    # short sample (a handful of weeks of news), this number should be
    # read as "what this rate would extrapolate to over a year if it
    # persisted" -- a big if -- not as a real annual return estimate.
    years_elapsed = max(n_periods / TRADING_DAYS_PER_YEAR, 1e-6)
    annualized_return = (1.0 + total_return) ** (1.0 / years_elapsed) - 1.0 if not np.isnan(total_return) else np.nan

    daily_vol = float(strategy_full.std())
    annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    daily_rf = APPROX_RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess_daily = strategy_full - daily_rf
    sharpe_ratio = (
        float(excess_daily.mean() / excess_daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if excess_daily.std() > 0 else np.nan
    )

    max_dd = _max_drawdown(cumulative_strategy)

    # Hit rate: fraction of individual trades where the direction of the
    # signal matched the direction of the realized return (a long that
    # went up, or a short that went down) -- a simple, interpretable
    # measure of "was the sentiment signal directionally right",
    # independent of position sizing or compounding.
    correct = np.sign(trades["signal"]) == np.sign(trades[return_col])
    hit_rate = float(correct.mean())

    return BacktestResult(
        strategy_returns=strategy_full,
        benchmark_returns=benchmark_full,
        cumulative_strategy=cumulative_strategy,
        cumulative_benchmark=cumulative_benchmark,
        n_trades=int(len(trades)),
        n_long=int((trades["signal"] == 1).sum()),
        n_short=int((trades["signal"] == -1).sum()),
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_dd,
        hit_rate=hit_rate,
    )
