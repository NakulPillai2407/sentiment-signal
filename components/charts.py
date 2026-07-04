"""
Plotly chart builders.

All charts here are interactive (hover tooltips, zoom/pan) by
construction because we use Plotly rather than Matplotlib -- a
deliberate upgrade from the original coursework's static Matplotlib
scatter plot. For a research tool, being able to hover a point and see
*which headline* produced an unusual return is far more useful than a
static image, especially when investigating outliers.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis.regression import RegressionResult

# A colourblind-friendly, high-contrast palette used consistently across
# charts so "positive/long" and "negative/short" always mean the same
# colour throughout the app.
COLOR_POSITIVE = "#2E7D32"
COLOR_NEGATIVE = "#C62828"
COLOR_NEUTRAL = "#757575"
COLOR_STRATEGY = "#1565C0"
COLOR_BENCHMARK = "#9E9E9E"
COLOR_FIT_LINE = "#EF6C00"


def sentiment_return_scatter(
    x: pd.Series,
    y: pd.Series,
    hover_ticker: pd.Series,
    hover_headline: pd.Series,
    regression: RegressionResult,
    x_label: str = "Sentiment score",
    y_label: str = "Abnormal return",
) -> go.Figure:
    """Scatter of sentiment vs. return with an overlaid OLS fit line.

    The fit line is drawn using the *same* fitted intercept/slope
    returned by analysis.regression.run_ols on this exact x/y pair, so
    the line on the chart is guaranteed to match the statistics printed
    alongside it (rather than being a separately-fit trendline that
    could silently drift out of sync).
    """
    hover_headline_trunc = hover_headline.astype(str).str.slice(0, 90) + hover_headline.astype(str).str.len().gt(90).map({True: "...", False: ""})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=8, color=COLOR_STRATEGY, opacity=0.65, line=dict(width=0.5, color="white")),
        text=[f"<b>{t}</b><br>{h}" for t, h in zip(hover_ticker, hover_headline_trunc)],
        hovertemplate="%{text}<br>Sentiment: %{x:.3f}<br>Return: %{y:.2%}<extra></extra>",
        name="Articles",
    ))

    if not regression.error and len(x.dropna()) > 0:
        x_range = np.linspace(x.min(), x.max(), 50)
        y_fit = regression.intercept + regression.slope * x_range
        fig.add_trace(go.Scatter(
            x=x_range, y=y_fit, mode="lines",
            line=dict(color=COLOR_FIT_LINE, width=3),
            name=f"OLS fit (slope={regression.slope:.4f})",
            hoverinfo="skip",
        ))

    fig.update_layout(
        xaxis_title=x_label, yaxis_title=y_label,
        yaxis_tickformat=".1%",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, l=10, r=10, b=10),
        hovermode="closest",
    )
    return fig


def residual_plot(fitted: pd.Series, residuals: pd.Series) -> go.Figure:
    """Residuals vs. fitted values, for a quick visual heteroskedasticity
    / non-linearity check.

    What to look for: a "healthy" OLS residual plot looks like a
    structureless, randomly-scattered horizontal band centred on zero.
    A funnel shape (residual spread widening or narrowing as fitted
    values increase) suggests heteroskedasticity -- non-constant error
    variance, which doesn't bias the slope estimate but does make the
    standard errors (and therefore the p-values and confidence
    intervals we rely on for significance testing) unreliable. A curved
    or clearly patterned scatter suggests the true relationship isn't
    linear, so a straight-line OLS fit may be the wrong model entirely.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_NEUTRAL)
    fig.add_trace(go.Scatter(
        x=fitted, y=residuals, mode="markers",
        marker=dict(size=8, color=COLOR_STRATEGY, opacity=0.65),
        hovertemplate="Fitted: %{x:.4f}<br>Residual: %{y:.4f}<extra></extra>",
        name="Residuals",
    ))
    fig.update_layout(
        xaxis_title="Fitted value", yaxis_title="Residual",
        template="plotly_white",
        margin=dict(t=30, l=10, r=10, b=10),
        showlegend=False,
    )
    return fig


def return_distribution_histogram(series: pd.Series, title: str = "") -> go.Figure:
    """Histogram of a return series, used to make outliers visually
    obvious before the user decides whether to winsorize.
    """
    fig = px.histogram(series.dropna(), nbins=40, template="plotly_white")
    fig.update_traces(marker_color=COLOR_STRATEGY, opacity=0.8)
    fig.update_layout(
        title=title, xaxis_title="Return", yaxis_title="Count",
        xaxis_tickformat=".1%",
        showlegend=False,
        margin=dict(t=40, l=10, r=10, b=10),
    )
    return fig


def cumulative_return_chart(
    cumulative_strategy: pd.Series,
    cumulative_benchmark: pd.Series,
    strategy_name: str = "Sentiment strategy",
    benchmark_name: str = "Buy & hold (equal-weight)",
) -> go.Figure:
    """Cumulative return line chart: strategy vs. buy-and-hold benchmark."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_strategy.index, y=cumulative_strategy.values,
        mode="lines", name=strategy_name,
        line=dict(color=COLOR_STRATEGY, width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Cumulative: %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cumulative_benchmark.index, y=cumulative_benchmark.values,
        mode="lines", name=benchmark_name,
        line=dict(color=COLOR_BENCHMARK, width=2, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>Cumulative: %{y:.2%}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#BDBDBD", line_width=1)
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Cumulative return",
        yaxis_tickformat=".1%",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, l=10, r=10, b=10),
        hovermode="x unified",
    )
    return fig


def agreement_heatmap(crosstab: pd.DataFrame) -> go.Figure:
    """Heatmap of the FinBERT-vs-VADER label crosstab (a lightweight
    stand-in for a confusion matrix -- there's no "ground truth" label
    here, just two models that may or may not agree).
    """
    fig = px.imshow(
        crosstab.values,
        x=list(crosstab.columns),
        y=list(crosstab.index),
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(
        xaxis_title="VADER label", yaxis_title="FinBERT label",
        template="plotly_white",
        margin=dict(t=30, l=10, r=10, b=10),
        coloraxis_showscale=False,
    )
    return fig


def sentiment_comparison_bar(finbert_stats: dict, vader_stats: dict, metric_labels: dict) -> go.Figure:
    """Side-by-side bar chart comparing key regression stats (e.g. R^2,
    |t-stat|) between the FinBERT-based and VADER-based regressions.
    """
    metrics = list(metric_labels.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="FinBERT", x=[metric_labels[m] for m in metrics],
        y=[finbert_stats.get(m, np.nan) for m in metrics],
        marker_color=COLOR_STRATEGY,
    ))
    fig.add_trace(go.Bar(
        name="VADER", x=[metric_labels[m] for m in metrics],
        y=[vader_stats.get(m, np.nan) for m in metrics],
        marker_color=COLOR_BENCHMARK,
    ))
    fig.update_layout(
        barmode="group", template="plotly_white",
        margin=dict(t=40, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig
