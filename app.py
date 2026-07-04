"""
Sentiment Signal -- main Streamlit entrypoint.

This app investigates whether financial news sentiment (scored with
FinBERT, a finance-tuned transformer) has any measurable relationship
with subsequent abnormal stock returns, for a user-chosen universe of
tickers. It began as a university coursework script (NLTK VADER on a
static 13-stock FTSE100 sample, ~10 headlines each, a single flat
"daily return", and an R^2 of ~0.02) and has been rebuilt here into a
live, interactive tool with:
  - a finance-domain sentiment model (FinBERT) alongside the original
    VADER baseline for direct comparison,
  - benchmark-adjusted ("abnormal") returns instead of raw returns, to
    strip out market-wide noise (see analysis/returns.py),
  - full OLS inferential statistics (t-stats, p-values, confidence
    intervals) instead of a bare slope/R^2,
  - transparent outlier handling (winsorization, not silent row
    deletion),
  - an illustrative signal backtest, and
  - an explicit, permanent limitations panel.

The app is organised as a Setup step (fetch data once) followed by five
read-only analysis tabs, all sharing the fetched data via
st.session_state so switching tabs never re-hits the network.
"""

import pandas as pd
import streamlit as st

from analysis import backtest as backtest_mod
from analysis import regression as regression_mod
from analysis import returns as returns_mod
from components import charts
from components.article_explorer import (
    build_display_table,
    filter_articles,
    flag_disagreements,
    get_spot_check_examples,
)
from data import news_fetcher, price_fetcher, universe
from sentiment import finbert_model, vader_model
from utils.constants import (
    DEFAULT_ARTICLES_PER_TICKER,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_PRESET_SAMPLE,
    LOOKBACK_OPTIONS,
    MAX_ARTICLES_PER_TICKER,
    MAX_TICKERS,
)

st.set_page_config(page_title="Sentiment Signal", page_icon="\U0001F4C8", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "data_fetched": False,
        "articles_df": pd.DataFrame(),
        "price_data": {},
        "dropped_tickers": {},
        "news_fetch_no_results": [],
        "fetch_errors": [],
        "resolved_tickers": [],
        "return_stats": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# Data pipeline (called once, from the Setup tab's "Fetch Data" button)
# ---------------------------------------------------------------------------
def compute_all_returns(articles_df: pd.DataFrame, price_data: dict):
    """Compute both the flat-subtraction and market-model abnormal
    returns in one pass, so the Sentiment vs. Returns tab can toggle
    between them instantly without re-fetching or re-scoring anything.
    """
    df_flat, stats = returns_mod.compute_returns(articles_df, price_data, use_market_model=False)
    if df_flat.empty:
        return df_flat, stats

    df_mm, _ = returns_mod.compute_returns(articles_df, price_data, use_market_model=True)
    df_flat = df_flat.reset_index(drop=True)

    if len(df_mm) == len(df_flat):
        rename_map = {f"abnormal_return_{w}": f"abnormal_return_{w}_mm" for w in returns_mod.WINDOWS}
        rename_map["beta_used"] = "beta_used_mm"
        mm_cols = df_mm[list(rename_map.keys())].rename(columns=rename_map).reset_index(drop=True)
        df_flat = pd.concat([df_flat, mm_cols], axis=1)
    return df_flat, stats


def run_fetch_pipeline(tickers: list[str], articles_per_ticker: int, lookback_days: int):
    """End-to-end: validate tickers -> fetch news -> score sentiment ->
    fetch prices -> compute returns. Drives its own progress UI since
    this is the one place in the app that makes a bounded, deliberate
    batch of external API calls.
    """
    st.session_state["fetch_errors"] = []
    st.session_state["news_fetch_no_results"] = []

    with st.status("Validating tickers...", expanded=True) as status:
        validation = universe.validate_tickers(tickers)
        st.session_state["dropped_tickers"] = validation.dropped
        valid_tickers, truncated = universe.enforce_max_tickers(validation.valid_tickers, MAX_TICKERS)
        if not valid_tickers:
            status.update(label="No valid tickers to fetch.", state="error")
            return
        if truncated:
            st.warning(f"Universe truncated to the first {MAX_TICKERS} valid tickers to respect API limits.")
        st.session_state["resolved_tickers"] = valid_tickers
        status.write(f"Resolved {len(valid_tickers)} valid ticker(s): {', '.join(valid_tickers)}")

        status.update(label="Fetching news articles...")
        news_progress = st.progress(0.0)

        def _news_cb(done, total, ticker):
            news_progress.progress(done / total, text=f"News: {ticker} ({done}/{total})")

        articles_df = news_fetcher.fetch_news_for_universe(
            valid_tickers, articles_per_ticker, lookback_days, progress_callback=_news_cb
        )
        news_progress.empty()

        if articles_df.empty:
            status.update(label="No news articles found for this universe/window.", state="error")
            return
        status.write(f"Retrieved {len(articles_df)} articles.")

        status.update(label="Scoring sentiment (FinBERT + VADER)...")
        with st.spinner("Loading FinBERT (first run downloads the model, ~400MB)..."):
            articles_df = finbert_model.score_articles(articles_df)
        articles_df = vader_model.score_articles(articles_df)
        status.write("Sentiment scoring complete.")

        status.update(label="Fetching price history...")
        price_progress = st.progress(0.0)

        def _price_cb(done, total, symbol):
            price_progress.progress(done / total, text=f"Prices: {symbol} ({done}/{total})")

        price_data = price_fetcher.fetch_prices_for_universe(
            valid_tickers, lookback_days, progress_callback=_price_cb
        )
        price_progress.empty()
        st.session_state["price_data"] = price_data

        status.update(label="Computing raw and abnormal returns...")
        final_df, return_stats = compute_all_returns(articles_df, price_data)
        st.session_state["return_stats"] = return_stats

        if final_df.empty:
            status.update(label="Articles were found, but none had usable price data to compute returns.", state="error")
            return

        st.session_state["articles_df"] = final_df
        st.session_state["data_fetched"] = True
        status.update(label=f"Done -- {len(final_df)} articles ready for analysis.", state="complete")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.title("\U0001F4C8 Sentiment Signal")
st.caption(
    "Does financial news sentiment predict abnormal stock returns? A live research tool -- "
    "rebuilt from a university coursework project into an interactive FinBERT-powered pipeline."
)

tab_setup, tab_explorer, tab_returns, tab_diagnostics, tab_backtest, tab_limitations = st.tabs([
    "1 \U0001F3AF Setup & Universe",
    "2 \U0001F4F0 Article Explorer",
    "3 \U0001F4CA Sentiment vs. Returns",
    "4 \U0001F52C Model Diagnostics",
    "5 \U0001F4B0 Signal Backtest",
    "6 ⚠️ Limitations & Methodology",
])

# ===========================================================================
# TAB 1 -- Setup & Universe
# ===========================================================================
with tab_setup:
    st.subheader("Build your ticker universe")
    st.caption(
        "Yahoo Finance ticker convention: US-listed stocks use a bare symbol (e.g. `AAPL`); "
        "London Stock Exchange listings need the `.L` suffix (e.g. `AZN.L` for AstraZeneca). "
        "Getting the suffix wrong is the most common reason a ticker fails validation below."
    )

    source = st.radio("Ticker source", ["Preset index", "Custom list"], horizontal=True)

    candidate_tickers: list[str] = []

    if source == "Preset index":
        index_name = st.selectbox("Index", ["FTSE100", "S&P500"])
        constituents = universe.load_index_constituents(index_name)
        constituents["display"] = constituents["Ticker"] + " -- " + constituents["Name"]

        default_n = min(DEFAULT_PRESET_SAMPLE, len(constituents))
        default_sample = constituents.sample(n=default_n, random_state=42)["display"].tolist()

        selected_display = st.multiselect(
            f"Select constituents (max {MAX_TICKERS})",
            options=constituents["display"].tolist(),
            default=default_sample,
        )
        candidate_tickers = [s.split(" -- ")[0] for s in selected_display]

    else:
        raw_text = st.text_area(
            "Enter tickers (comma-separated or one per line)",
            placeholder="AZN.L, HSBA.L, AAPL, MSFT",
            height=100,
        )
        candidate_tickers = universe.parse_custom_tickers(raw_text)
        if candidate_tickers:
            st.caption(f"Parsed {len(candidate_tickers)} ticker(s): {', '.join(candidate_tickers)}")

    if len(candidate_tickers) > MAX_TICKERS:
        st.warning(
            f"You've selected {len(candidate_tickers)} tickers, above the cap of {MAX_TICKERS}. "
            f"Only the first {MAX_TICKERS} will be used -- this cap exists to avoid hammering "
            "yfinance's unofficial, unauthenticated API with a huge batch of sequential requests."
        )

    col1, col2 = st.columns(2)
    with col1:
        articles_per_ticker = st.slider(
            "Articles per ticker", min_value=3, max_value=MAX_ARTICLES_PER_TICKER,
            value=DEFAULT_ARTICLES_PER_TICKER,
        )
    with col2:
        lookback_days = st.select_slider(
            "News lookback window (days)", options=LOOKBACK_OPTIONS, value=DEFAULT_LOOKBACK_DAYS,
        )

    fetch_clicked = st.button("\U0001F680 Fetch Data", type="primary", disabled=not candidate_tickers)

    if fetch_clicked:
        run_fetch_pipeline(candidate_tickers, articles_per_ticker, lookback_days)

    if st.session_state["dropped_tickers"]:
        with st.expander(f"⚠️ {len(st.session_state['dropped_tickers'])} ticker(s) dropped during validation", expanded=True):
            for t, reason in st.session_state["dropped_tickers"].items():
                st.write(f"- **{t}**: {reason}")

    if st.session_state["news_fetch_no_results"]:
        st.info(
            "No news found (within the lookback window) for: "
            + ", ".join(st.session_state["news_fetch_no_results"])
        )

    if st.session_state["data_fetched"]:
        st.success(
            f"Universe ready: {len(st.session_state['resolved_tickers'])} tickers, "
            f"{len(st.session_state['articles_df'])} articles with computed returns. "
            "Head to the other tabs to explore the analysis."
        )
        stats = st.session_state["return_stats"]
        if stats is not None and (stats.skipped_no_price_data or stats.skipped_insufficient_window):
            st.caption(
                f"({stats.skipped_no_price_data} articles skipped for missing price data, "
                f"{stats.skipped_insufficient_window} skipped for insufficient surrounding price history "
                "e.g. very recent news with no forward-looking prices yet.)"
            )

# Guard clause used by every downstream tab.
data_ready = st.session_state["data_fetched"] and not st.session_state["articles_df"].empty
articles_df = st.session_state["articles_df"]

# ===========================================================================
# TAB 2 -- Article Explorer
# ===========================================================================
with tab_explorer:
    if not data_ready:
        st.info("Fetch data in the **Setup & Universe** tab first.")
    else:
        st.subheader("Browse individual articles")

        f1, f2, f3, f4 = st.columns([1.2, 1, 1.4, 1])
        with f1:
            ticker_filter = st.multiselect("Ticker", sorted(articles_df["ticker"].unique()))
        with f2:
            label_filter = st.selectbox("FinBERT label", ["All", "positive", "negative", "neutral"])
        with f3:
            min_date = pd.to_datetime(articles_df["publish_date"]).min().date()
            max_date = pd.to_datetime(articles_df["publish_date"]).max().date()
            date_filter = st.date_input("Date range", value=(min_date, max_date))
        with f4:
            disagreements_only = st.checkbox("Show disagreements only")

        st.caption(
            "\U0001F4A1 'Disagreements' = FinBERT and VADER assign opposite (positive vs. negative) "
            "directions to the same article. This is a feature of comparing two different models on "
            "ambiguous financial text, not a bug -- see Model Diagnostics for how often it happens."
        )

        window_choice = st.selectbox(
            "Return window to display", returns_mod.WINDOWS,
            format_func=lambda w: returns_mod.WINDOW_LABELS[w], key="explorer_window",
        )

        filtered = filter_articles(
            articles_df,
            tickers=ticker_filter or None,
            sentiment_label=label_filter,
            date_range=date_filter if isinstance(date_filter, tuple) else None,
            disagreements_only=disagreements_only,
        )

        display_table = build_display_table(filtered, return_col=f"raw_return_{window_choice}")
        st.dataframe(
            display_table,
            use_container_width=True,
            column_config={
                "FinBERT Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                "FinBERT Score": st.column_config.NumberColumn(format="%.3f"),
                "VADER Compound": st.column_config.NumberColumn(format="%.3f"),
                "Realized Return": st.column_config.NumberColumn(format="percent"),
                "Summary": st.column_config.TextColumn(width="large"),
            },
            hide_index=True,
        )
        st.caption(f"Showing {len(filtered)} of {len(articles_df)} articles.")

        st.divider()
        st.subheader("Spot checks: most positive / most negative (by FinBERT)")
        st.caption(
            "A manual sanity check in the spirit of the original coursework report: read a handful of "
            "extreme-scored articles and judge whether the score actually matches the tone."
        )
        most_pos, most_neg = get_spot_check_examples(articles_df, n=3)

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("**Most positive (FinBERT)**")
            for _, row in most_pos.iterrows():
                with st.expander(f"{row['ticker']}: {row['headline'][:70]}"):
                    st.write(row["summary"] or "*(no summary available)*")
                    st.write(f"FinBERT: **{row['finbert_label']}** (score={row['finbert_score']:.3f}, "
                              f"confidence={row['finbert_confidence']:.2f}) | VADER compound: {row['vader_compound']:.3f}")
        with col_neg:
            st.markdown("**Most negative (FinBERT)**")
            for _, row in most_neg.iterrows():
                with st.expander(f"{row['ticker']}: {row['headline'][:70]}"):
                    st.write(row["summary"] or "*(no summary available)*")
                    st.write(f"FinBERT: **{row['finbert_label']}** (score={row['finbert_score']:.3f}, "
                              f"confidence={row['finbert_confidence']:.2f}) | VADER compound: {row['vader_compound']:.3f}")

# ===========================================================================
# TAB 3 -- Sentiment vs. Returns
# ===========================================================================
with tab_returns:
    if not data_ready:
        st.info("Fetch data in the **Setup & Universe** tab first.")
    else:
        st.subheader("Does sentiment predict abnormal returns?")

        c1, c2, c3 = st.columns(3)
        with c1:
            window = st.selectbox(
                "Return window", returns_mod.WINDOWS,
                format_func=lambda w: returns_mod.WINDOW_LABELS[w], key="returns_window",
            )
        with c2:
            use_market_model = st.toggle(
                "Use market-model (beta-adjusted) abnormal return",
                value=False,
                help="Off: abnormal return = stock return - benchmark return (assumes beta=1). "
                     "On: abnormal return = stock return - (estimated beta * benchmark return), using a "
                     "trailing OLS beta estimate. See analysis/returns.py for the full rationale.",
            )
        with c3:
            winsorize_on = st.toggle("Winsorize returns (1st/99th pct)", value=True)

        abnormal_col = f"abnormal_return_{window}" + ("_mm" if use_market_model else "")
        if abnormal_col not in articles_df.columns:
            st.error("Market-model returns unavailable for this dataset -- falling back to flat benchmark subtraction.")
            abnormal_col = f"abnormal_return_{window}"

        st.markdown(
            "**Why abnormal, not raw, returns?** A stock's raw return on any given day is dominated by "
            "market-wide moves that have nothing to do with firm-specific news. Abnormal return subtracts "
            "out the benchmark's move (or a beta-scaled version of it) over the same window, isolating the "
            "part of the return that's arguably attributable to the stock's *own* news. This is very likely "
            "why the original coursework's raw-return regression produced an R² of only ~0.02."
        )

        working_df = articles_df.dropna(subset=[abnormal_col, "finbert_score"]).copy()

        if winsorize_on:
            working_df = returns_mod.apply_winsorization(working_df, [abnormal_col], 0.01, 0.99)
            y_col = f"{abnormal_col}_wz"
            st.caption(
                "Winsorized: values beyond the 1st/99th percentile are capped at those percentiles "
                "(not deleted) before regression, to limit the influence of extreme outliers without "
                "discarding data. Toggle off to see the raw, unwinsorized regression."
            )
        else:
            y_col = abnormal_col
            st.caption("**Unwinsorized**: results below use raw, uncapped abnormal returns.")

        st.plotly_chart(
            charts.return_distribution_histogram(working_df[abnormal_col], title="Abnormal return distribution (raw, pre-winsorization)"),
            use_container_width=True,
        )

        reg = regression_mod.run_ols(working_df["finbert_score"], working_df[y_col])

        st.plotly_chart(
            charts.sentiment_return_scatter(
                working_df["finbert_score"], working_df[y_col],
                working_df["ticker"], working_df["headline"], reg,
                x_label="FinBERT sentiment score (positive - negative probability)",
                y_label=f"Abnormal return ({returns_mod.WINDOW_LABELS[window]})",
            ),
            use_container_width=True,
        )

        if reg.error:
            st.error(reg.error)
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Slope", f"{reg.slope:.4f}")
            m2.metric("Intercept", f"{reg.intercept:.4f}")
            m3.metric("R²", f"{reg.r_squared:.4f}")
            m4.metric("N (articles)", f"{reg.n_obs}")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("t-stat (slope)", f"{reg.slope_tstat:.3f}")
            m6.metric("p-value (slope)", f"{reg.slope_pvalue:.4f}")
            m7.metric("95% CI (slope)", f"[{reg.slope_ci_low:.4f}, {reg.slope_ci_high:.4f}]")
            m8.metric("F-stat p-value", f"{reg.f_pvalue:.4f}")

            if reg.is_significant:
                st.success(regression_mod.significance_statement(reg))
            else:
                st.warning(regression_mod.significance_statement(reg))

# ===========================================================================
# TAB 4 -- Model Diagnostics
# ===========================================================================
with tab_diagnostics:
    if not data_ready:
        st.info("Fetch data in the **Setup & Universe** tab first.")
    else:
        st.subheader("FinBERT vs. VADER: does the 'better' model actually fit better?")

        window_d = st.selectbox(
            "Return window", returns_mod.WINDOWS,
            format_func=lambda w: returns_mod.WINDOW_LABELS[w], key="diag_window",
        )
        abnormal_col_d = f"abnormal_return_{window_d}"
        diag_df = articles_df.dropna(subset=[abnormal_col_d, "finbert_score", "vader_compound"])

        reg_finbert = regression_mod.run_ols(diag_df["finbert_score"], diag_df[abnormal_col_d])
        reg_vader = regression_mod.run_ols(diag_df["vader_compound"], diag_df[abnormal_col_d])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**FinBERT-based regression**")
            if reg_finbert.error:
                st.error(reg_finbert.error)
            else:
                st.write(f"Slope: `{reg_finbert.slope:.4f}` | R²: `{reg_finbert.r_squared:.4f}` | "
                         f"p-value: `{reg_finbert.slope_pvalue:.4f}` | N: `{reg_finbert.n_obs}`")
        with col_b:
            st.markdown("**VADER-based regression**")
            if reg_vader.error:
                st.error(reg_vader.error)
            else:
                st.write(f"Slope: `{reg_vader.slope:.4f}` | R²: `{reg_vader.r_squared:.4f}` | "
                         f"p-value: `{reg_vader.slope_pvalue:.4f}` | N: `{reg_vader.n_obs}`")

        if not reg_finbert.error and not reg_vader.error:
            metric_labels = {"r_squared": "R²", "slope_tstat_abs": "|t-stat|"}
            finbert_stats = {"r_squared": reg_finbert.r_squared, "slope_tstat_abs": abs(reg_finbert.slope_tstat)}
            vader_stats = {"r_squared": reg_vader.r_squared, "slope_tstat_abs": abs(reg_vader.slope_tstat)}
            st.plotly_chart(
                charts.sentiment_comparison_bar(finbert_stats, vader_stats, metric_labels),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Agreement between FinBERT and VADER labels")
        crosstab = pd.crosstab(articles_df["finbert_label"], articles_df["vader_label"])
        st.plotly_chart(charts.agreement_heatmap(crosstab), use_container_width=True)
        agree_rate = (articles_df["finbert_label"] == articles_df["vader_label"]).mean()
        disagree_rate = flag_disagreements(articles_df).mean()
        st.caption(
            f"Exact label agreement: **{agree_rate:.1%}**. Directional (positive/negative) disagreement: "
            f"**{disagree_rate:.1%}**. Disagreement is expected -- VADER is a general-purpose lexicon model "
            "with no notion of financial-domain meaning, while FinBERT was fine-tuned on financial text."
        )

        st.divider()
        st.subheader("Residual plot (FinBERT regression)")
        st.caption(
            "What to look for: residuals should form a structureless, roughly constant-width band around "
            "zero. A funnel shape (variance changing with fitted value) suggests heteroskedasticity, which "
            "undermines the standard errors -- and therefore the p-values and confidence intervals -- "
            "reported in the Sentiment vs. Returns tab, even though it wouldn't bias the slope estimate "
            "itself. A curved pattern would suggest the true relationship isn't linear."
        )
        if not reg_finbert.error:
            st.plotly_chart(charts.residual_plot(reg_finbert.fitted_values, reg_finbert.residuals), use_container_width=True)

# ===========================================================================
# TAB 5 -- Signal Backtest
# ===========================================================================
with tab_backtest:
    st.error(f"⚠️ **{backtest_mod.BACKTEST_DISCLAIMER}**")

    if not data_ready:
        st.info("Fetch data in the **Setup & Universe** tab first.")
    else:
        st.subheader("Sentiment-threshold long/short backtest")

        b1, b2, b3 = st.columns(3)
        with b1:
            signal_model = st.selectbox("Sentiment signal source", ["FinBERT", "VADER"])
        with b2:
            bt_window = st.selectbox(
                "Holding window", ["next_day", "3day"],
                format_func=lambda w: returns_mod.WINDOW_LABELS[w],
            )
        with b3:
            st.write("")

        s1, s2 = st.columns(2)
        with s1:
            long_threshold = st.slider("Long threshold (sentiment ≥)", 0.0, 1.0, 0.3, 0.05)
        with s2:
            short_threshold = st.slider("Short threshold (sentiment ≤)", -1.0, 0.0, -0.3, 0.05)

        sentiment_col = "finbert_score" if signal_model == "FinBERT" else "vader_compound"

        result = backtest_mod.run_backtest(
            articles_df, st.session_state["price_data"], sentiment_col, bt_window,
            long_threshold, short_threshold,
        )

        if result.error:
            st.warning(result.error)
        else:
            st.plotly_chart(
                charts.cumulative_return_chart(result.cumulative_strategy, result.cumulative_benchmark),
                use_container_width=True,
            )

            p1, p2, p3, p4, p5, p6 = st.columns(6)
            p1.metric("Total return", f"{result.total_return:.2%}")
            p2.metric("Annualized return*", f"{result.annualized_return:.2%}")
            p3.metric("Annualized vol*", f"{result.annualized_vol:.2%}")
            p4.metric("Sharpe ratio*", f"{result.sharpe_ratio:.2f}")
            p5.metric("Max drawdown", f"{result.max_drawdown:.2%}")
            p6.metric("Hit rate", f"{result.hit_rate:.1%}")
            st.caption(
                f"*Annualized figures extrapolate a short sample ({result.n_trades} trades: "
                f"{result.n_long} long, {result.n_short} short) to a full year and should be read as "
                "illustrative scale, not a forecast. Sharpe uses a flat ~4% approximate risk-free rate."
            )

# ===========================================================================
# TAB 6 -- Limitations & Methodology Notes
# ===========================================================================
with tab_limitations:
    st.subheader("Limitations & Methodology Notes")
    st.caption("Read this before drawing any conclusions from the other tabs.")

    st.markdown("""
**Sample size.** News-driven event studies in academic finance typically use thousands of
firm-events to get statistically reliable estimates. This tool typically works with a few
dozen to a few hundred articles across a user-chosen universe -- enough to illustrate the
methodology, not enough to draw confident conclusions about a true population relationship.
Treat every regression's confidence interval width, not just its point estimate, as part of
the answer.

**Correlation is not causation.** Even a statistically significant sentiment-return
relationship does not establish that sentiment *causes* returns. Both could be driven by a
third factor (e.g. the underlying news event itself moves the price *and* is what the
sentiment model is scoring -- sentiment is a noisy proxy for "the news," not an independent
causal input).

**Lookahead bias risk in news summaries.** Yahoo Finance article summaries are sometimes
generated or edited after initial publication, and can occasionally reference price action
that happened *after* the nominal publish timestamp (e.g. "shares fell 3% following the
announcement"). If so, we would be scoring sentiment on text that already encodes the price
move we're trying to predict, which would inflate the apparent relationship. We use the
publish timestamp reported by the API at face value; we cannot independently verify exactly
when summary text was finalized.

**Event-day alignment is a convention, not a certainty.** We anchor each article to the
first trading day on or after its publish date, and define same-day/next-day/3-day windows
from there (see analysis/returns.py). This is a standard, defensible event-study convention,
but it doesn't account for exact intraday timing (e.g. news published 5 minutes before the
close vs. 5 hours before) or which exchange's trading hours actually apply.

**Benchmark selection is a heuristic.** We infer the relevant market benchmark from the
ticker's exchange suffix (`.L` → FTSE 100, else → S&P 500). This will misclassify
any stock whose primary listing/currency exposure doesn't match that simple rule (e.g. a
UK-headquartered but US-listed ADR).

**Both sentiment models are imperfect proxies.** FinBERT is a substantial improvement over a
general-purpose lexicon model for financial text, but it was trained on analyst-labelled
phrases and Reuters newswire text -- it can still misread irony, complex conditional
statements ("would have missed estimates if not for..."), or company-specific context it
wasn't trained on. Neither model has access to information beyond the text itself (e.g. how
the market actually reacted, or whether the news was already priced in).

**The backtest is illustrative only.** No transaction costs, slippage, bid-ask spread, market
impact, or short-borrow costs are modelled. Position sizing is a flat equal weight per
signal, not risk-adjusted. Annualized return/Sharpe figures extrapolate from a short sample
and are highly sensitive to a handful of trades. See the Signal Backtest tab's own on-page
disclaimer.

**Index constituent lists are static snapshots.** The bundled FTSE 100 / S&P 500 lists are
a point-in-time snapshot bundled with the app, not a live feed -- index membership changes
periodically (additions, removals, ticker changes) and the bundled list will drift out of
date without a manual refresh.

**yfinance is an unofficial API.** It has no official rate-limit documentation or support
guarantee; its response schemas have changed between versions before and may again. The app
handles missing/malformed data defensively, but a broader Yahoo Finance outage or schema
change could degrade functionality until patched.

**Winsorization is a judgment call.** Capping returns at the 1st/99th percentile (adjustable)
is a defensible, transparent way to limit outlier influence versus silently deleting rows,
but the specific percentile threshold is still a choice that shrinks tail variance -- it is
not a "correct" answer, just a documented one.
""")

st.divider()
st.caption(
    "Sentiment Signal -- built as a portfolio project. Originally a university coursework "
    "script; rebuilt as a live research tool. Not investment advice."
)
