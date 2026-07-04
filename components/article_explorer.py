"""
Helper functions for the Article Explorer tab: filtering, formatting,
and identifying model disagreements.

Kept separate from app.py so the filtering/formatting logic is testable
and reusable independent of Streamlit's widget/rerun model.
"""

import pandas as pd


def filter_articles(
    df: pd.DataFrame,
    tickers: list[str] | None = None,
    sentiment_label: str | None = None,
    date_range: tuple | None = None,
    disagreements_only: bool = False,
) -> pd.DataFrame:
    """Apply the Article Explorer's filter controls to the articles frame.

    Every filter is optional/no-op when left at its default, so this
    can be called unconditionally on every rerun without special-casing
    "no filter selected".
    """
    out = df.copy()

    if tickers:
        out = out[out["ticker"].isin(tickers)]

    if sentiment_label and sentiment_label != "All":
        out = out[out["finbert_label"].str.lower() == sentiment_label.lower()]

    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        pub = pd.to_datetime(out["publish_date"]).dt.tz_localize(None)
        out = out[(pub >= start) & (pub <= end)]

    if disagreements_only:
        out = out[flag_disagreements(out)]

    return out


def flag_disagreements(df: pd.DataFrame) -> pd.Series:
    """Boolean mask for articles where FinBERT and VADER disagree on
    sentiment *direction* (positive vs. negative -- we don't count a
    'neutral' vs. 'positive' call as a disagreement, since that's a
    much softer discrepancy than a positive/negative flip).

    This is deliberately surfaced as a feature, not hidden: two
    reasonable sentiment models disagreeing on ambiguous financial text
    is expected and informative (it shows where lexicon-based scoring
    breaks down versus a domain-tuned transformer), not a bug in either
    model.
    """
    if "finbert_label" not in df.columns or "vader_label" not in df.columns:
        return pd.Series(False, index=df.index)

    def _direction(label: str) -> int:
        if label == "positive":
            return 1
        if label == "negative":
            return -1
        return 0

    finbert_dir = df["finbert_label"].map(_direction)
    vader_dir = df["vader_label"].map(_direction)
    return (finbert_dir != 0) & (vader_dir != 0) & (finbert_dir != vader_dir)


def get_spot_check_examples(df: pd.DataFrame, n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the top-n most positive and top-n most negative articles
    by FinBERT score, for manual spot-checking in the UI.

    Mirrors the manual "read a few articles and sanity-check the score"
    exercise in the original coursework report -- a cheap, honest way
    to catch a sentiment model doing something obviously wrong (e.g.
    scoring a headline about a competitor's bad news as positive for
    the covered company) that aggregate statistics alone wouldn't show.
    """
    if "finbert_score" not in df.columns or df.empty:
        return df.head(0), df.head(0)
    most_positive = df.nlargest(n, "finbert_score")
    most_negative = df.nsmallest(n, "finbert_score")
    return most_positive, most_negative


def build_display_table(df: pd.DataFrame, return_col: str | None = None) -> pd.DataFrame:
    """Select and rename columns for a clean st.dataframe presentation.

    Kept separate from the underlying analysis dataframe (which keeps
    verbose/raw column names for programmatic use) so the UI can show a
    human-friendly view without mutating the data other tabs rely on.
    """
    cols = {
        "ticker": "Ticker",
        "headline": "Headline",
        "summary": "Summary",
        "publish_date": "Published",
        "finbert_label": "FinBERT Label",
        "finbert_confidence": "FinBERT Confidence",
        "finbert_score": "FinBERT Score",
        "vader_compound": "VADER Compound",
    }
    if return_col and return_col in df.columns:
        cols[return_col] = "Realized Return"

    available = {k: v for k, v in cols.items() if k in df.columns}
    out = df[list(available.keys())].rename(columns=available)
    if "Published" in out.columns:
        out["Published"] = pd.to_datetime(out["Published"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
    return out
