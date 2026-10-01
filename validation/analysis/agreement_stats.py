"""
agreement_stats.py — shared agreement statistics for the PyZebArdYolo validation
=================================================================================
Functions used by 01_pairing_metrics.py (per video) and 03_agreement_summary.py
(pooled), so that every ICC, confidence interval and Bland-Altman figure in the
paper comes from a single implementation.

    icc_a1(ratings)            ICC(2,1) / ICC(A,1), two-way random, absolute
                               agreement, single measure, with the exact 95% CI
                               of McGraw & Wong (1996). Full precision (no rounding).
    bland_altman(diff)         bias, SD, 95% limits of agreement, % inside the limits
    proportional_bias(m, d)    OLS of difference on mean + Breusch-Pagan test

Dependencies: numpy, scipy, statsmodels
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def icc_a1(ratings: np.ndarray, alpha: float = 0.05) -> dict:
    """ICC(A,1) (Shrout & Fleiss ICC(2,1)) with the McGraw & Wong (1996) CI.

    Parameters
    ----------
    ratings : array (n_targets, k_raters); rows with any NaN are dropped.
    alpha   : 1 - confidence level.

    Returns
    -------
    dict with icc, ci_lo, ci_hi, n (targets used).
    """
    r = np.asarray(ratings, float)
    r = r[~np.isnan(r).any(axis=1)]
    n, k = r.shape
    out = {"icc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": n}
    if n < 3:
        return out
    grand = r.mean()
    ms_r = k * ((r.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((r.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = r - r.mean(axis=1, keepdims=True) - r.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    est = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)

    a = k * est / (n * (1 - est))
    b = 1 + k * est * (n - 1) / (n * (1 - est))
    v = (a * ms_c + b * ms_e) ** 2 / (
        (a * ms_c) ** 2 / (k - 1) + (b * ms_e) ** 2 / ((n - 1) * (k - 1))
    )
    f_lower = stats.f.ppf(1 - alpha / 2, n - 1, v)
    f_upper = stats.f.ppf(1 - alpha / 2, v, n - 1)
    lo = n * (ms_r - f_lower * ms_e) / (
        f_lower * (k * ms_c + (k * n - k - n) * ms_e) + n * ms_r
    )
    hi = n * (f_upper * ms_r - ms_e) / (
        k * ms_c + (k * n - k - n) * ms_e + n * f_upper * ms_r
    )
    out.update(icc=float(est), ci_lo=float(lo), ci_hi=float(hi))
    return out


def bland_altman(diff: np.ndarray) -> dict:
    """Bias, SD (ddof=1), 95% limits of agreement and % of differences inside them."""
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    bias, sd = d.mean(), d.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    return {"n": d.size, "bias": bias, "sd": sd, "loa_lo": lo, "loa_hi": hi,
            "pct_within_loa": float(np.mean((d >= lo) & (d <= hi)) * 100)}


def proportional_bias(mean: np.ndarray, diff: np.ndarray) -> dict:
    """Bland-Altman regression (diff ~ mean) and Breusch-Pagan test on its residuals.

    Also returns the change in predicted bias across the useful range of the axis,
    |slope| x (P99 - P1 of the means), used in the text as the practical size of any
    proportional bias (percentiles discard a few extreme points without cutting the
    area the animal actually explored).
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan

    m = np.asarray(mean, float)
    d = np.asarray(diff, float)
    ok = ~(np.isnan(m) | np.isnan(d))
    m, d = m[ok], d[ok]
    fit = sm.OLS(d, sm.add_constant(m)).fit()
    _, bp_p, _, _ = het_breuschpagan(fit.resid, fit.model.exog)
    ci = fit.conf_int()[1]
    return {"slope": float(fit.params[1]), "slope_ci_lo": float(ci[0]),
            "slope_ci_hi": float(ci[1]), "slope_p": float(fit.pvalues[1]),
            "bp_p": float(bp_p),
            "bias_change_across_range": float(abs(fit.params[1]) *
                                              (np.percentile(m, 99) - np.percentile(m, 1)))}
