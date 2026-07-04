"""
FinBERT sentiment scoring.

Why FinBERT over a general-purpose lexicon model (VADER) as the primary
signal: VADER was trained on social-media text and scores words against
a fixed, hand-built lexicon -- it has no notion of financial-domain
meaning. Phrases that are neutral-to-positive in everyday English can be
bearish in a financial context (e.g. "cuts costs by reducing headcount",
"raises prices amid inflation"), and VADER has no way to learn that.
FinBERT (Araci, 2019 -- ProsusAI/finbert) is a BERT model further
pre-trained on financial text (Reuters TRC2) and fine-tuned on analyst
sentiment-labelled financial phrases, so it has actually seen
domain-specific usage patterns. It is not perfect -- see the
Limitations tab -- but it is the more defensible choice for scoring
financial news headlines/summaries than a general lexicon model.

We keep VADER in the app anyway (sentiment/vader_model.py) precisely so
we can *demonstrate* this gap empirically (Model Diagnostics tab) rather
than just asserting it.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.caching import cache_model

MODEL_NAME = "ProsusAI/finbert"

# FinBERT's classification head outputs logits in this fixed order --
# this ordering comes from the model's config (id2label) and must match
# it exactly, otherwise positive/negative probabilities get swapped.
LABELS = ["positive", "negative", "neutral"]


@cache_model
def load_finbert():
    """Load and cache the FinBERT tokenizer + model.

    Cached via st.cache_resource (not st.cache_data) because the return
    value is a live PyTorch model object, not serialisable data. Without
    caching, Streamlit's rerun-on-every-interaction model would reload
    ~400MB of weights on every single widget change, which is both slow
    and wasteful.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def _prepare_text(headline: str, summary: str) -> str:
    """Build the text FinBERT scores for one article.

    We concatenate headline + summary rather than scoring only the
    summary: headlines are often written to be attention-grabbing and
    can carry sentiment the summary body doesn't restate (the original
    coursework version noted headlines are "brief" and summaries
    "richer" -- concatenating gets the richness of the summary while not
    discarding headline-only signal). If a summary is missing (some
    Yahoo Finance news records have none), we fall back to the headline
    alone rather than producing an empty-string score.
    """
    # Guard against NaN (a float) rather than relying on `x or ""`:
    # float('nan') is truthy in Python, so `nan or ""` evaluates to nan
    # and a subsequent `.strip()` call would raise AttributeError.
    headline = "" if pd.isna(headline) else str(headline).strip()
    summary = "" if pd.isna(summary) else str(summary).strip()
    if summary:
        return f"{headline}. {summary}" if headline else summary
    return headline


@torch.no_grad()
def score_articles(df: pd.DataFrame, batch_size: int = 8) -> pd.DataFrame:
    """Score every article in `df` (needs 'headline' and 'summary' cols).

    Returns the input DataFrame with new columns:
      - finbert_positive, finbert_negative, finbert_neutral: softmax
        probabilities (sum to 1)
      - finbert_label: the arg-max class
      - finbert_confidence: probability of the arg-max class
      - finbert_score: positive - negative probability, our continuous
        sentiment signal used throughout the regression analysis (see
        analysis/regression.py for why a continuous signal is preferred
        over the discrete label for a regression against continuous
        returns)

    Batched inference (default batch_size=8) rather than one
    forward-pass per article: transformer inference has meaningful
    per-call overhead, so batching materially speeds up scoring a
    universe of 100+ articles on CPU.
    """
    if df.empty:
        out = df.copy()
        for col in ["finbert_positive", "finbert_negative", "finbert_neutral",
                    "finbert_label", "finbert_confidence", "finbert_score"]:
            out[col] = pd.Series(dtype="float64" if col != "finbert_label" else "object")
        return out

    tokenizer, model = load_finbert()
    texts = [_prepare_text(h, s) for h, s in zip(df["headline"], df["summary"])]

    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()
        all_probs.append(probs)
    probs = np.vstack(all_probs)

    out = df.copy()
    for idx, label in enumerate(LABELS):
        out[f"finbert_{label}"] = probs[:, idx]
    out["finbert_label"] = out[["finbert_positive", "finbert_negative", "finbert_neutral"]].idxmax(axis=1).str.replace("finbert_", "")
    out["finbert_confidence"] = out[["finbert_positive", "finbert_negative", "finbert_neutral"]].max(axis=1)

    # Continuous sentiment signal for regression: positive minus negative
    # probability, ranging from -1 (fully negative) to +1 (fully
    # positive), with a high-neutral-probability article naturally
    # landing near 0 regardless of which of positive/negative edges it
    # out on. We use this instead of the raw positive probability alone
    # because a P(positive)=0.5 article that's otherwise all-neutral
    # (P(neutral)=0.5) and a P(positive)=0.5 article that's a genuine
    # 50/50 positive-vs-negative split are very different signals, and
    # pos-minus-neg distinguishes them while a bare P(positive) does not.
    out["finbert_score"] = out["finbert_positive"] - out["finbert_negative"]

    return out
