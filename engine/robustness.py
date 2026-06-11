"""Robustness layer: estimation error is the biggest risk in mean
variance optimisation, so this module measures and dampens it.

Three tools:

1. Ledoit Wolf shrinkage covariance (2004, "A well conditioned
   estimator for large dimensional covariance matrices"). Shrinks the
   sample covariance toward a scaled identity with an analytically
   optimal intensity, instead of the fixed shrinkage fraction in
   stats.covariance_matrix. Config switch; the simple estimator stays
   the default.

2. Resampled efficiency (Michaud style). Bootstrap the daily returns,
   re-optimise on each resample, average the weights. The dispersion of
   weights across resamples shows which positions are conviction and
   which are estimation noise.

3. Sensitivity report. Bump each ETF's expected return by plus and
   minus a small amount (default 1%) and re-optimise. Positions whose
   bump moves the whole portfolio a lot are fragile: the optimiser is
   leaning on a return estimate that small data changes would overturn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimiser, stats

FRAGILE_TURNOVER = 0.10  # a 1% return bump moving >10% of the book = fragile


# ----------------------------------------------------------------------
# Ledoit Wolf covariance
# ----------------------------------------------------------------------

def ledoit_wolf_cov(returns: pd.DataFrame, periods: int = 252) -> tuple[pd.DataFrame, float]:
    """Ledoit Wolf (2004) shrinkage toward mu*I where mu is the average
    sample eigenvalue. Returns (annualised covariance, shrink intensity
    in [0, 1]). Pure numpy, no sklearn dependency.
    """
    X = returns.values - returns.values.mean(axis=0)
    t, n = X.shape
    sample = X.T @ X / t

    mu = np.trace(sample) / n
    target = mu * np.eye(n)
    d2 = np.linalg.norm(sample - target, "fro") ** 2 / n

    b2_sum = 0.0
    for k in range(t):
        xk = X[k][:, None]
        b2_sum += np.linalg.norm(xk @ xk.T - sample, "fro") ** 2 / n
    b2 = min(b2_sum / t**2, d2)

    intensity = b2 / d2 if d2 > 0 else 0.0
    shrunk = intensity * target + (1.0 - intensity) * sample
    cov = pd.DataFrame(shrunk * periods, index=returns.columns, columns=returns.columns)
    return cov, float(intensity)


# ----------------------------------------------------------------------
# Resampled efficiency
# ----------------------------------------------------------------------

def resampled_weights(returns: pd.DataFrame, rf: float, settings: dict,
                      baskets: list[dict], w_opt: pd.Series,
                      n_boot: int = 100, seed: int = 11) -> pd.DataFrame:
    """Bootstrap daily returns (iid rows with replacement), re-optimise
    max Sharpe on each resample under the SAME constraints, and report
    per ETF weight mean and std across resamples next to the point
    estimate optimal. Failed solves are skipped.

    A position whose resampled mean is far below its optimal weight, or
    whose std is large, is being driven by sampling luck rather than a
    stable risk return trade off.
    """
    rng = np.random.default_rng(seed)
    periods = settings["annualisation"]
    max_pos = settings["max_position"]
    shrink = settings.get("shrinkage", 0.0)
    t = len(returns)

    samples = []
    for _ in range(n_boot):
        boot = returns.iloc[rng.integers(0, t, t)]
        exp_ret = stats.expected_returns(boot, periods)
        cov = stats.covariance_matrix(boot, periods, shrink)
        try:
            w = optimiser.max_sharpe(exp_ret, cov, rf, max_pos, baskets)
            samples.append(w)
        except RuntimeError:
            continue
    if not samples:
        raise RuntimeError("All bootstrap re-optimisations failed.")

    W = np.array(samples)
    out = pd.DataFrame({
        "optimal": w_opt.reindex(returns.columns).fillna(0.0),
        "boot_mean": W.mean(axis=0),
        "boot_std": W.std(axis=0),
    }, index=returns.columns)
    out["n_samples"] = len(samples)
    return out.sort_values("optimal", ascending=False)


# ----------------------------------------------------------------------
# Sensitivity to expected return estimates
# ----------------------------------------------------------------------

def sensitivity_report(exp_ret: pd.Series, cov: pd.DataFrame, rf: float,
                       settings: dict, baskets: list[dict], w_base: np.ndarray,
                       bump: float = 0.01) -> pd.DataFrame:
    """For each ETF, shift its expected return up and down by `bump` and
    re-optimise. Reports the portfolio turnover each shift causes
    (0.5 * sum |w_new - w_base|, the fraction of the book that trades)
    and the ETF's own weight change. fragile = True when either
    direction moves more than FRAGILE_TURNOVER of the portfolio.
    """
    max_pos = settings["max_position"]
    rows = []
    for i, t in enumerate(exp_ret.index):
        row = {"ticker": t, "weight": float(w_base[i])}
        for sign, label in ((+1, "up"), (-1, "down")):
            bumped = exp_ret.copy()
            bumped.iloc[i] += sign * bump
            try:
                w_new = optimiser.max_sharpe(bumped, cov, rf, max_pos, baskets,
                                             x0=w_base)
                row[f"turnover_{label}"] = float(0.5 * np.abs(w_new - w_base).sum())
                row[f"own_change_{label}"] = float(w_new[i] - w_base[i])
            except RuntimeError:
                row[f"turnover_{label}"] = np.nan
                row[f"own_change_{label}"] = np.nan
        row["fragile"] = bool(max(row["turnover_up"] or 0, row["turnover_down"] or 0)
                              > FRAGILE_TURNOVER)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("ticker")
    return df.sort_values(["fragile", "turnover_up"], ascending=False)
