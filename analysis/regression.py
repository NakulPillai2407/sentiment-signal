"""
OLS regression wrapper (statsmodels) with full inferential statistics.

We use statsmodels.OLS rather than sklearn's LinearRegression (as the
original coursework script did) because sklearn is built for
*prediction* -- it gives you a fitted line and an R^2 and nothing else.
statsmodels is built for *inference* -- it gives you standard errors,
t-statistics, p-values, and confidence intervals, which is what you
actually need to answer the real question here: "is this relationship
distinguishable from noise, or could a slope this large arise by chance
even if sentiment has zero true effect on returns?" A slope and an R^2
alone cannot answer that question.

---------------------------------------------------------------------
How to read the statistics this module returns
---------------------------------------------------------------------
slope (beta_1)         : the estimated change in abnormal return for a
                          one-unit increase in sentiment score. Because
                          our sentiment score is bounded in [-1, 1], a
                          more intuitive read is "moving from maximally
                          negative to maximally positive sentiment is
                          associated with a 2 * slope change in
                          abnormal return".
slope_se               : standard error of the slope estimate -- how
                          much the slope would vary if we re-sampled the
                          data. Small relative to the slope itself is
                          good; large means the slope is imprecisely
                          estimated (often due to a small sample, which
                          is a real risk here -- see the Limitations
                          tab).
slope_tstat            : slope / slope_se. Roughly, "how many standard
                          errors is the slope away from zero".
slope_pvalue           : probability of observing a t-statistic this
                          extreme *if the true slope were zero* (the
                          null hypothesis of "no relationship"). Small
                          p-values (conventionally < 0.05) suggest the
                          observed relationship is unlikely to be pure
                          sampling noise -- NOT that the relationship is
                          large, important, or causal.
slope_ci_95            : the range of slope values consistent with the
                          data at 95% confidence. If this interval
                          contains zero, we cannot reject "no
                          relationship" at conventional significance.
r_squared              : fraction of variance in abnormal returns
                          explained by sentiment alone. Expect this to
                          be small (a few percent) even for a genuine
                          effect -- daily stock returns are dominated by
                          noise/other factors, so a "real" news-based
                          signal explaining, say, 2-5% of return
                          variance would not be unusual in the
                          academic event-study literature.
f_pvalue               : joint-significance test for the whole model
                          (here, equivalent to the slope's own p-value
                          since we have exactly one regressor, but
                          computed and shown for completeness/rigor).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class RegressionResult:
    n_obs: int
    intercept: float
    slope: float
    intercept_se: float
    slope_se: float
    intercept_pvalue: float
    slope_pvalue: float
    slope_tstat: float
    slope_ci_low: float
    slope_ci_high: float
    r_squared: float
    adj_r_squared: float
    f_stat: float
    f_pvalue: float
    fitted_values: pd.Series
    residuals: pd.Series
    x: pd.Series
    y: pd.Series
    is_significant: bool
    alpha: float = 0.05
    error: str | None = None


def _empty_result(error: str) -> RegressionResult:
    empty = pd.Series(dtype="float64")
    return RegressionResult(
        n_obs=0, intercept=np.nan, slope=np.nan, intercept_se=np.nan, slope_se=np.nan,
        intercept_pvalue=np.nan, slope_pvalue=np.nan, slope_tstat=np.nan,
        slope_ci_low=np.nan, slope_ci_high=np.nan, r_squared=np.nan, adj_r_squared=np.nan,
        f_stat=np.nan, f_pvalue=np.nan, fitted_values=empty, residuals=empty,
        x=empty, y=empty, is_significant=False, error=error,
    )


def run_ols(x: pd.Series, y: pd.Series, alpha: float = 0.05) -> RegressionResult:
    """Fit y = intercept + slope * x via OLS and return full diagnostics.

    Rows with a NaN in either x or y are dropped pairwise (common when a
    return window near the edge of available price history couldn't be
    computed for some articles -- see analysis/returns.py). We require
    at least 3 valid observations to fit anything meaningful (technically
    OLS needs >= 2 to be non-degenerate, but standard errors are only
    informative with more residual degrees of freedom than that).
    """
    paired = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(paired) < 3:
        return _empty_result(
            f"Not enough paired observations to run a regression (have {len(paired)}, need >= 3)."
        )

    X = sm.add_constant(paired["x"])
    model = sm.OLS(paired["y"], X).fit()

    slope_pvalue = float(model.pvalues.get("x", np.nan))
    ci = model.conf_int(alpha=alpha)
    slope_ci = ci.loc["x"] if "x" in ci.index else (np.nan, np.nan)

    return RegressionResult(
        n_obs=int(model.nobs),
        intercept=float(model.params.get("const", np.nan)),
        slope=float(model.params.get("x", np.nan)),
        intercept_se=float(model.bse.get("const", np.nan)),
        slope_se=float(model.bse.get("x", np.nan)),
        intercept_pvalue=float(model.pvalues.get("const", np.nan)),
        slope_pvalue=slope_pvalue,
        slope_tstat=float(model.tvalues.get("x", np.nan)),
        slope_ci_low=float(slope_ci[0]),
        slope_ci_high=float(slope_ci[1]),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_stat=float(model.fvalue) if model.fvalue is not None else np.nan,
        f_pvalue=float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        fitted_values=model.fittedvalues,
        residuals=model.resid,
        x=paired["x"],
        y=paired["y"],
        is_significant=bool(slope_pvalue < alpha) if not np.isnan(slope_pvalue) else False,
        alpha=alpha,
    )


def significance_statement(result: RegressionResult) -> str:
    """Plain-language, honest summary of the regression's statistical
    significance -- written so it cannot be quoted out of context as
    "sentiment predicts returns" when the data doesn't support that.
    """
    if result.error:
        return result.error
    if result.is_significant:
        direction = "positive" if result.slope > 0 else "negative"
        return (
            f"Statistically significant at the {result.alpha:.0%} level "
            f"(p = {result.slope_pvalue:.4f}). The estimated relationship is {direction}, "
            f"but statistical significance alone does not establish economic significance, "
            f"causality, or predictive value out-of-sample -- see the Limitations tab."
        )
    return (
        f"NOT statistically significant at the {result.alpha:.0%} level "
        f"(p = {result.slope_pvalue:.4f}). We cannot reject the possibility that the true "
        f"relationship between sentiment and abnormal returns is zero. Any apparent trend in "
        f"the scatter plot should be treated as noise unless this changes with a larger sample."
    )
