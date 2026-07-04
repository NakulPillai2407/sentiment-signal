"""
Thin wrappers around Streamlit's caching primitives.

We centralise TTLs here for two reasons:
1. News and prices have very different "freshness" requirements -- a
   headline published an hour ago is stale in a way a closing price from
   today is not, but neither changes meaningfully within a single user
   session, so both are safe to cache aggressively for the session.
2. Keeping the TTL constants in one file means we don't end up with five
   different magic numbers scattered across fetcher modules.

Streamlit's `st.cache_data` keys the cache automatically off the
function's arguments, so as long as callers pass ticker lists /
lookback windows as hashable arguments (tuples, not lists), re-running
the same query is free -- this directly satisfies the "don't re-hit the
API on identical inputs" requirement from the app spec.
"""

import streamlit as st

# News moves faster than prices intraday, but for the purposes of this
# app (a single interactive analysis session) both are effectively
# static once fetched. A 1-hour TTL avoids ever going fully stale across
# a long session while still saving every rerun within that window.
NEWS_CACHE_TTL_SECONDS = 60 * 60
PRICE_CACHE_TTL_SECONDS = 60 * 60
VALIDATION_CACHE_TTL_SECONDS = 60 * 60 * 24  # ticker validity rarely changes

cache_news = st.cache_data(ttl=NEWS_CACHE_TTL_SECONDS, show_spinner=False)
cache_prices = st.cache_data(ttl=PRICE_CACHE_TTL_SECONDS, show_spinner=False)
cache_validation = st.cache_data(ttl=VALIDATION_CACHE_TTL_SECONDS, show_spinner=False)

# FinBERT model weights are large (~400MB) and slow to load -- this must
# be a *resource* cache (st.cache_resource), not a data cache, because we
# are caching an unpicklable, stateful object (the loaded model/tokenizer),
# not a serialisable value.
cache_model = st.cache_resource(show_spinner=False)
