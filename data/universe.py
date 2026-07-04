"""
Ticker universe resolution: turning a user's selection (a preset index
subset, or a hand-typed list) into a validated set of tickers the rest
of the pipeline can trust.

The key job of this module is to fail loudly and *early*. Every
downstream module (news, prices, sentiment, regression) assumes the
tickers it receives actually resolve on Yahoo Finance -- if we let a
typo like "AZM.L" (instead of "AZN.L") through, it silently produces
empty news/price frames three modules downstream, which is a much
harder bug to trace than "we told the user AZM.L doesn't exist" at
selection time.
"""

from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

from utils.caching import cache_validation
from utils.constants import INDEX_FILES, MAX_TICKERS


@dataclass
class UniverseResult:
    """Outcome of resolving a user's ticker selection.

    valid_tickers: tickers confirmed to resolve on Yahoo Finance and
        safe to pass to the news/price fetchers.
    dropped: mapping of ticker -> reason, for anything we rejected, so
        the UI can show the user exactly what was excluded and why
        (rather than silently shrinking their universe).
    """

    valid_tickers: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)


def load_index_constituents(index_name: str) -> pd.DataFrame:
    """Load the bundled constituent list for a preset index.

    We ship static CSVs (utils/data/*.csv) rather than scraping an index
    provider's site at runtime: index membership changes slowly (a
    handful of times a year), so a periodically-refreshed static file is
    far more reliable than a live scrape that can break the app the
    moment a source website changes its markup.
    """
    path = INDEX_FILES.get(index_name)
    if path is None or not path.exists():
        raise ValueError(f"No bundled constituent list for index '{index_name}'")
    return pd.read_csv(path)


def parse_custom_tickers(raw_text: str) -> list[str]:
    """Split a comma/newline/whitespace-separated user string into tickers.

    Accepts both a single comma-separated line and a multi-line text
    area, since the spec allows either input widget. Tickers are
    upper-cased and de-duplicated (order-preserving) because Yahoo
    Finance ticker symbols are case-sensitive-looking but conventionally
    always upper-case, and users will inevitably paste duplicates.
    """
    if not raw_text:
        return []
    # Normalise newlines to commas, then split on both.
    normalised = raw_text.replace("\n", ",")
    raw_tokens = [t.strip().upper() for t in normalised.split(",")]
    tickers, seen = [], set()
    for t in raw_tokens:
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)
    return tickers


@cache_validation
def _ticker_is_valid(ticker: str) -> bool:
    """Check whether yfinance can resolve a ticker to real price history.

    We probe with a short history fetch rather than trusting
    `Ticker.info` (which can return a near-empty dict for a dead/invalid
    ticker without raising, giving a false positive). A short recent
    history window is a cheap, decisive test: if it's empty, the ticker
    is not tradeable/listed as far as Yahoo Finance is concerned.
    """
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        return not hist.empty
    except Exception:
        return False


def validate_tickers(tickers: list[str]) -> UniverseResult:
    """Validate a list of candidate tickers against yfinance.

    Returns both the surviving tickers and a reason for every rejection,
    so the caller (Setup & Universe tab) can render an honest "here's
    what we dropped and why" message instead of quietly shrinking the
    universe -- silent data loss is exactly the kind of thing that
    erodes trust in a quant research tool.
    """
    result = UniverseResult()
    for t in tickers:
        if not t:
            continue
        if _ticker_is_valid(t):
            result.valid_tickers.append(t)
        else:
            result.dropped[t] = "Could not resolve on Yahoo Finance (check spelling/suffix)"
    return result


def enforce_max_tickers(tickers: list[str], max_tickers: int = MAX_TICKERS) -> tuple[list[str], bool]:
    """Cap the universe size to respect API rate limits.

    yfinance has no official rate-limit documentation, but in practice
    hammering it with 50-100 sequential ticker requests in a Streamlit
    rerun loop is a reliable way to get throttled or IP-blocked. Capping
    the universe (configurable, default 25) keeps a single "Fetch Data"
    click to a bounded number of requests.
    """
    truncated = len(tickers) > max_tickers
    return tickers[:max_tickers], truncated
