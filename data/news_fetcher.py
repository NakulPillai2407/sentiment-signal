"""
News retrieval via yfinance.

yfinance's `.news` endpoint is an unofficial, undocumented wrapper
around Yahoo Finance's internal API, so its response schema has changed
between library versions (flat dicts with 'title'/'providerPublishTime'
in older releases, a nested 'content' dict with 'title'/'pubDate' in
newer ones). We parse defensively -- trying multiple known field names
-- rather than pinning to one schema, so the app degrades gracefully
(skips a malformed record) instead of crashing outright if Yahoo
changes the payload shape again.
"""

from datetime import timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

from utils.caching import cache_news

NEWS_COLUMNS = ["ticker", "headline", "summary", "publish_date", "link"]


def _extract_field(article: dict, *candidates, default=""):
    """Pull the first present field from a list of possible key names.

    Handles both the flat and nested ('content') yfinance news schemas
    without needing to know in advance which one we've been given.
    """
    content = article.get("content", article)
    for key in candidates:
        if key in content and content[key] not in (None, ""):
            val = content[key]
            # Some fields (title, canonicalUrl) can themselves be dicts
            # in the newer schema, e.g. {'canonicalUrl': {'url': ...}}.
            if isinstance(val, dict):
                val = val.get("url") or val.get("value") or ""
            return val
    return default


def _extract_publish_date(article: dict) -> pd.Timestamp | None:
    """Normalise the publish timestamp to a timezone-aware pandas Timestamp.

    Older yfinance returns a Unix epoch int ('providerPublishTime');
    newer returns an ISO-8601 string ('pubDate'). Both are handled here
    so downstream return-window calculations always get a consistent,
    comparable timestamp type.
    """
    content = article.get("content", article)
    if "providerPublishTime" in content:
        try:
            return pd.Timestamp(content["providerPublishTime"], unit="s", tz="UTC")
        except Exception:
            return None
    for key in ("pubDate", "displayTime"):
        if key in content and content[key]:
            try:
                return pd.Timestamp(content[key], tz="UTC") if pd.Timestamp(content[key]).tzinfo is None \
                    else pd.Timestamp(content[key])
            except Exception:
                continue
    return None


def _parse_articles(raw_articles: list[dict], ticker: str, cutoff: pd.Timestamp) -> list[dict]:
    """Convert yfinance's raw news payload into our flat article schema."""
    rows = []
    for article in raw_articles:
        pub_date = _extract_publish_date(article)
        if pub_date is None or pub_date < cutoff:
            continue
        headline = _extract_field(article, "title")
        summary = _extract_field(article, "summary", "description")
        link = _extract_field(article, "canonicalUrl", "link", "clickThroughUrl")
        if not headline:
            continue  # an article with no headline is not usable for sentiment scoring
        rows.append({
            "ticker": ticker,
            "headline": headline,
            "summary": summary,
            "publish_date": pub_date,
            "link": link,
        })
    return rows


@cache_news
def fetch_news_for_ticker(ticker: str, max_articles: int, lookback_days: int) -> pd.DataFrame:
    """Fetch recent news articles for a single ticker.

    Returns an empty DataFrame (not an exception) when Yahoo Finance has
    no news for the ticker/window -- this is a common, expected outcome
    for smaller or less-covered names, not an error condition.
    """
    cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=lookback_days)
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    rows = _parse_articles(raw, ticker, cutoff)[:max_articles]
    if not rows:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def fetch_news_for_universe(
    tickers: list[str],
    max_articles: int,
    lookback_days: int,
    progress_callback=None,
) -> pd.DataFrame:
    """Fetch news for every ticker in the universe and concatenate.

    Iterates sequentially (yfinance has no bulk news endpoint) and
    reports progress via an optional callback so the caller can drive a
    Streamlit progress bar without this module needing to import
    Streamlit UI code directly -- keeps data-fetching and presentation
    concerns separate.
    """
    frames = []
    failures = []
    for i, ticker in enumerate(tickers):
        try:
            df = fetch_news_for_ticker(ticker, max_articles, lookback_days)
            if df.empty:
                failures.append(ticker)
            else:
                frames.append(df)
        except Exception as e:
            failures.append(ticker)
            st.session_state.setdefault("fetch_errors", []).append(f"{ticker}: {e}")
        if progress_callback:
            progress_callback(i + 1, len(tickers), ticker)

    if failures:
        st.session_state["news_fetch_no_results"] = failures

    if not frames:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return pd.concat(frames, ignore_index=True)
