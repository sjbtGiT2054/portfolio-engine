"""Macro sensitivity estimation and Black Litterman view construction.

For each active macro variable we estimate, per holding, an OLS
sensitivity of the holding's MONTHLY return to the MONTHLY CHANGE in that
variable, over up to ten years of overlapping monthly data. The slope is
the holding's historical response; the t statistic tells us whether it is
distinguishable from zero.

These are noisy historical relationships, NOT causal constants. Regressing
a handful of years of monthly equity returns against many macro series
recovers mostly noise: signs flip across samples, and most coefficients
are not significant. Everything downstream is framed as the portfolio's
implied response IF these relationships held, never as a forecast. The
t stat and the low confidence flag exist precisely so the weak ones can
be discounted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNIFICANCE = 0.10           # two sided; |t| ~ 1.64 at 10% for large n
LOW_CONF_ABS_T = 1.645
CONFIDENCE_MAP = {"low": 0.2, "medium": 0.5, "high": 0.8}


def ols_sensitivity(y: pd.Series, x: pd.Series) -> dict:
    """OLS of y on a constant and x, aligned on their common index.
    Returns coef (slope), tstat, n, and a low_confidence flag (|t| below
    the 10% threshold, or too few points)."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(df)
    if n < 12:
        return {"coef": 0.0, "tstat": 0.0, "n": n, "low_confidence": True}
    xv = df["x"].values
    yv = df["y"].values
    X = np.column_stack([np.ones(n), xv])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = n - 2
    sigma2 = float(resid @ resid) / dof if dof > 0 else np.inf
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(sigma2 * xtx_inv[1, 1])) if sigma2 < np.inf else np.inf
    tstat = float(beta[1] / se_slope) if se_slope > 0 else 0.0
    return {"coef": float(beta[1]), "tstat": tstat, "n": n,
            "low_confidence": bool(abs(tstat) < LOW_CONF_ABS_T)}


def monthly_var_change(series: pd.Series) -> pd.Series:
    """Month on month change in a macro variable (resampled to month end,
    first difference). For a rate this is the change in percentage points;
    for an index or price it is the change in level."""
    return series.resample("ME").last().diff().dropna()


def monthly_returns(prices_usd: pd.DataFrame) -> pd.DataFrame:
    """Month on month simple returns per holding."""
    return prices_usd.resample("ME").last().pct_change().dropna(how="all")


def estimate_sensitivities(prices_usd: pd.DataFrame,
                           macro_series: dict[str, pd.Series],
                           max_years: int = 10) -> dict[str, dict[str, dict]]:
    """For each macro key in macro_series and each holding column, estimate
    the OLS sensitivity. Returns {var_key: {ticker: {coef, tstat, n,
    low_confidence}}}. Series that are None are skipped (their slider is
    greyed out upstream)."""
    rets = monthly_returns(prices_usd)
    cutoff = rets.index.max() - pd.DateOffset(years=max_years)
    rets = rets[rets.index >= cutoff]
    out: dict[str, dict[str, dict]] = {}
    for key, series in macro_series.items():
        if series is None or len(series) == 0:
            continue
        dvar = monthly_var_change(series)
        out[key] = {t: ols_sensitivity(rets[t], dvar) for t in rets.columns}
    return out


def build_view_tilts(sensitivities: dict[str, dict[str, dict]],
                     deviations: dict[str, float],
                     horizon_years: int,
                     var_confidence: dict[str, str],
                     tickers: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """Turn macro path deviations into per holding annualised return tilts
    and per holding confidences.

    For holding i:
      tilt_i = sum_v (coef_iv * deviation_v) / horizon_years

    deviation_v is the variable's end of horizon gap from its baseline
    (same units the sensitivity was estimated in). coef_iv maps a unit
    change to a monthly return, and a cumulative path deviation telescopes
    to that total change, so coef_iv * deviation_v is the cumulative return
    impact; dividing by the horizon annualises it for the optimiser.

    The per holding confidence is the tilt magnitude weighted average of
    the contributing variables' confidences, so the variables that move a
    holding most also set how firmly its view is held.
    Returns (tilts, confidences), both keyed by ticker.
    """
    tilts: dict[str, float] = {}
    confs: dict[str, float] = {}
    h = max(int(horizon_years), 1)
    for t in tickers:
        tilt = 0.0
        wconf = 0.0
        wsum = 0.0
        for key, dev in deviations.items():
            if dev == 0 or key not in sensitivities:
                continue
            coef = sensitivities[key].get(t, {}).get("coef", 0.0)
            contrib = coef * dev / h
            tilt += contrib
            mag = abs(contrib)
            wconf += mag * CONFIDENCE_MAP.get(var_confidence.get(key, "medium"), 0.5)
            wsum += mag
        tilts[t] = tilt
        confs[t] = (wconf / wsum) if wsum > 1e-12 else CONFIDENCE_MAP["medium"]
    return tilts, confs


def build_bl_views(pi: pd.Series, tilts: dict[str, float],
                   confs: dict[str, float], eps: float = 1e-4) -> list[dict]:
    """Construct absolute Black Litterman views from per holding tilts.
    Each view says holding i's excess return is its equilibrium prior plus
    the macro tilt, held at the blended confidence. Holdings with a
    negligible tilt get no view (so a flat macro path leaves the posterior
    equal to the prior). Format matches engine.black_litterman.view_matrices.
    """
    views = []
    for t, tilt in tilts.items():
        if abs(tilt) < eps:
            continue
        views.append({"name": f"macro_{t}", "assets": [t],
                      "excess_return": float(pi.get(t, 0.0) + tilt),
                      "confidence": float(confs.get(t, CONFIDENCE_MAP["medium"]))})
    return views
