# Sentiment Signal

**Does financial news sentiment predict abnormal stock returns?**

An interactive Streamlit app that pulls live news for a user-chosen set of tickers, scores it with FinBERT (a finance-tuned transformer), pulls matching price data, and rigorously tests — with proper OLS inference, not just a bare correlation — whether sentiment has any measurable relationship with subsequent abnormal returns. It also ships a transparent, illustrative long/short backtest of the sentiment signal, and is explicit throughout about the statistical limits of what any of this can actually claim.

Built as a portfolio project for Quantitative Analyst / Financial Technology roles, rebuilt from a university Computational Economics coursework script into a live, interactive tool.

## Live Demo
* *

## Overview

This project started as a university coursework script for a Computational Economics module. The original version:

- used NLTK's VADER (a general-purpose, social-media-tuned lexicon model) to score financial headlines,
- ran on a static, one-time pull of 13 hand-picked FTSE 100 tickers with 10 articles each,
- computed a single "daily return" as the percentage change between the closing price ~2 days before and ~2 days after an article's publish date, with no separation of market-wide vs. firm-specific price movement,
- regressed sentiment against that raw return with `sklearn`, getting a slope of ~1.04 and an **R² of ~0.02** — a very weak fit,
- and handled one obvious outlier by hardcoding `df.drop(80)`.

Rather than write that up as a one-off report and move on, I rebuilt it as a live tool, because the original's weak result raised a much more interesting question than the coursework asked: *is that R² of 0.02 weak because sentiment genuinely doesn't matter, or because raw daily returns are dominated by market-wide noise that has nothing to do with firm-specific news?* Answering that properly meant upgrading almost every stage of the pipeline — a finance-tuned sentiment model, a real event-study return methodology, and honest statistical inference instead of a slope and an R² — which is what this repo is.

I'm keeping that origin story here deliberately rather than hiding it: the ability to look at a weak academic result, diagnose *why* it might be weak, and rebuild the methodology to test that diagnosis, is a more useful signal than pretending the project started at its current state.

## Key Features

- Pulls live news for any user-chosen tickers (FTSE100 / S&P500 preset or custom list) via `yfinance`, with defensive parsing that degrades gracefully across Yahoo's changing news schema
- Scores every headline with **FinBERT** (`ProsusAI/finbert`), a finance-domain transformer, alongside **VADER** as a comparison baseline
- Computes both raw and **benchmark-adjusted ("abnormal") returns** — flat or beta-scaled market-model — to isolate firm-specific price impact from market-wide moves
- Runs full **OLS inference** (`statsmodels`) on sentiment vs. abnormal return: t-statistics, p-values, 95% confidence intervals, and an F-test — not just a slope and an R²
- User-toggleable winsorization (1st/99th percentile capping, not silent deletion) for outlier handling
- Ships an illustrative sentiment-threshold **long/short backtest**, explicit about its lack of transaction costs, slippage, and small sample size
- A **Model Diagnostics** tab comparing FinBERT vs. VADER head-to-head (agreement rates, relative regression strength)
- A dedicated, permanent **Limitations & Methodology Notes** tab rather than a buried caveat paragraph
- One-time fetch per session via `st.session_state` — switching tabs or tweaking a slider never re-hits the network

## Methodology

| Stage | Original coursework | This app |
|---|---|---|
| Sentiment model | VADER only | FinBERT (primary) + VADER (kept for comparison) |
| Universe | 13 fixed FTSE100 tickers | User-chosen, FTSE100 / S&P500 preset or custom list |
| Return | Raw % change, ~2 days before/after | Same-day / next-day / 3-day windows, event-day aligned |
| Return type | Raw only | Raw **and** benchmark-adjusted ("abnormal") return |
| Outliers | Hardcoded `df.drop(80)` | User-toggleable winsorization (1st/99th pct, capped not deleted) |
| Regression | `sklearn` (slope, intercept, R² only) | `statsmodels` OLS: t-stats, p-values, 95% CIs, F-test |
| Backtest | None | Illustrative sentiment-threshold long/short backtest |
| Limitations | Brief conclusion paragraph | Dedicated, permanent tab |

**Why abnormal returns?** A stock's raw daily return is mostly driven by how the whole market moved that day, not by firm-specific news. A market-wide 2% rally will show up as a "positive return" on every stock in the universe regardless of whether that stock had any news at all. Abnormal return subtracts out the market's move (either flat subtraction of the benchmark's return, or a beta-scaled version via a simple rolling market-model regression) to isolate the part of the move plausibly attributable to the stock's own news — see `analysis/returns.py` for the full derivation and the trade-offs between the two approaches.

**Why FinBERT over VADER?** VADER is a lexicon model built for social media text with no notion of financial-domain meaning — phrases like "cuts costs by reducing headcount" can score as neutral-to-positive under a general lexicon while being bearish in context. FinBERT (`ProsusAI/finbert`) is BERT further pre-trained on financial text and fine-tuned on analyst-labelled sentiment, so it has actually seen domain-specific usage. The app keeps VADER around specifically so the **Model Diagnostics** tab can show this gap empirically (agreement rates, side-by-side regression strength) rather than just asserting it.

**Why `statsmodels` over `sklearn`?** `sklearn.LinearRegression` gives you a fitted line and nothing else — it's built for prediction. `statsmodels.OLS` gives standard errors, t-statistics, p-values, and confidence intervals, which is what's actually needed to answer "is this relationship distinguishable from noise, given a small sample?" rather than just "what's the best-fit line through these points."

Full narrative on every methodological choice — event-day alignment convention, benchmark-selection heuristic, winsorization rationale, backtest assumptions — lives as inline commentary in the relevant module (`analysis/returns.py`, `analysis/regression.py`, `analysis/backtest.py`) and is summarized for a non-technical reader in the app's own **Limitations & Methodology Notes** tab.

## Repo Structure

```
app.py                          # Streamlit entrypoint, tab routing, session_state pipeline
data/
  universe.py                   # Ticker resolution: preset index / custom list, yfinance validation
  news_fetcher.py                # yfinance news retrieval, defensive schema parsing, caching
  price_fetcher.py                # yfinance price retrieval, benchmark alignment, caching
sentiment/
  finbert_model.py                 # FinBERT loading + batched scoring
  vader_model.py                    # VADER scoring (comparison baseline)
analysis/
  returns.py                        # Raw/abnormal return computation, event-day alignment, winsorizing
  regression.py                      # statsmodels OLS wrapper with full inferential statistics
  backtest.py                         # Vectorized sentiment-threshold long/short backtest
components/
  article_explorer.py                  # Filtering/formatting helpers for the Article Explorer tab
  charts.py                              # All Plotly chart builders
utils/
  caching.py                              # st.cache_data / st.cache_resource wrappers + TTLs
  constants.py                             # Benchmark mapping, rate-limit caps, bundled index CSV paths
  data/                                     # Bundled FTSE100 / S&P500 constituent CSVs
```

## Installation & Running Locally

```bash
git clone https://github.com/NakulPillai2407/sentiment-signal.git
cd sentiment-signal
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

First run will download FinBERT weights (~400MB) and the NLTK VADER lexicon; both are cached locally afterwards (`st.cache_resource` / `~/nltk_data`) and won't re-download on subsequent runs.

**Usage:** open the **Setup & Universe** tab, pick a preset index subset or type your own tickers, set the article count and lookback window, and click **Fetch Data**. Every other tab reads from that one fetch via `st.session_state`.

## Tech Stack

- **Python**, **Streamlit**, **Plotly**, **Pandas / NumPy**
- **[yfinance](https://github.com/ranaroussi/yfinance)** — live news and price data, no API key required
- **`transformers` + `torch`** — FinBERT (`ProsusAI/finbert`) sentiment scoring
- **NLTK (VADER)** — comparison baseline sentiment model
- **`statsmodels`** — OLS regression with full inferential statistics

## What I'd Do With More Time / Data

- **Sector-level breakdown.** The original coursework's FTSE 100 industry-count analysis is a natural follow-on: does the sentiment-return relationship differ by sector (e.g. cyclicals vs. defensives)? Deliberately out of scope here to keep the core pipeline focused, but a clean next iteration.
- **A real risk-free rate series** for the Sharpe ratio instead of a flat approximate rate, and proper trading-calendar-aware annualization instead of a simple period-count scaling.
- **Transaction-cost-adjusted backtesting** with a real position-sizing model, instead of the current flat equal-weight, cost-free illustration.
- **A larger, longer-running sample** collected over weeks/months rather than a single on-demand pull, which would materially improve the statistical power of every regression in the app — the single biggest limitation of the current version is simply sample size.
- **Independent verification of news timestamp integrity** to rule out the lookahead-bias risk flagged in the Limitations tab (i.e. checking whether Yahoo Finance summaries are ever edited post-publication).

## Limitations

This backtest and regression analysis is illustrative only — small sample sizes (a single on-demand pull, typically well under a few hundred articles/trades), no transaction costs or slippage in the backtest, and no causal identification in the regression (a statistically significant slope shows correlation, not that sentiment *causes* returns). See the in-app **Limitations & Methodology Notes** tab for the full set of caveats before drawing any conclusions from its output.

**This is a research/portfolio tool, not investment advice.**

## Author

**Nakul Pillai**
BSc Economics & Data Science, University of Southampton · Incoming MSc Financial Technology, Imperial College London

- LinkedIn: [linkedin.com/in/nakul-pillai](https://www.linkedin.com/in/nakul-pillai)
- GitHub: [@NakulPillai2407](https://github.com/NakulPillai2407)
