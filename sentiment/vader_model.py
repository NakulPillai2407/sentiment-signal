"""
VADER sentiment scoring -- retained as the comparison/baseline model.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a
rule-based, lexicon-driven sentiment scorer originally built for social
media text. It's fast (no GPU/model download needed) and was the model
used in the original coursework version of this project. We keep it
here deliberately, not to power the main analysis, but so the Model
Diagnostics tab can show a real, empirical comparison of a
domain-general lexicon model against a finance-tuned transformer
(FinBERT) -- i.e. to demonstrate *why* FinBERT is the better choice
here, rather than just asserting it.
"""

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def _ensure_vader_lexicon():
    """Download the VADER lexicon on first use if it isn't already present.

    NLTK ships its data separately from the package itself, so a fresh
    environment needs this one-time download. We guard it with a
    try/except rather than checking a version flag because nltk's own
    `nltk.data.find` is the most reliable way to check without
    duplicating its internal path logic.
    """
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def score_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Score every article's headline+summary with VADER.

    Adds:
      - vader_compound: VADER's compound score in [-1, 1], the
        standard single-number summary of overall sentiment polarity
        VADER's own documentation recommends using for this purpose.
      - vader_label: a discretised positive/negative/neutral label using
        VADER's own suggested thresholds (>=0.05 positive, <=-0.05
        negative, else neutral), so it's directly comparable to
        FinBERT's discrete label in the agreement/disagreement crosstab.
    """
    _ensure_vader_lexicon()
    analyzer = SentimentIntensityAnalyzer()

    out = df.copy()
    if out.empty:
        out["vader_compound"] = pd.Series(dtype="float64")
        out["vader_label"] = pd.Series(dtype="object")
        return out

    def _compound(headline, summary):
        # NaN-safe: a missing (float NaN) headline/summary must not be
        # formatted into the literal string "nan", which would silently
        # feed bogus tokens into the lexicon scorer.
        headline = "" if pd.isna(headline) else str(headline)
        summary = "" if pd.isna(summary) else str(summary)
        text = f"{headline}. {summary}".strip(". ")
        return analyzer.polarity_scores(text)["compound"] if text else 0.0

    out["vader_compound"] = [
        _compound(h, s) for h, s in zip(out["headline"], out["summary"])
    ]
    out["vader_label"] = out["vader_compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    )
    return out
