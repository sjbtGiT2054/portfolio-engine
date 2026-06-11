"""Black Litterman expected returns (Phase 2).

The model blends two sources of information:

1. Equilibrium returns (the prior). Reverse optimisation of the CAPM
   market portfolio: pi = delta * Sigma * w_mkt, where delta is the
   market's risk aversion implied by the benchmark's excess return over
   its variance. This is "what returns must the market expect for the
   observed weights to be optimal", and it replaces noisy historical
   means as the neutral starting point.

2. User views (the tilt), read from views.yaml. Each view says a set of
   assets will out or underperform by some annual amount, with a
   confidence in (0, 1]. Views are OPTIONAL and the file ships with
   commented examples only. With zero active views the posterior equals
   the equilibrium exactly, so switching the engine to black_litterman
   without views simply swaps noisy historical means for equilibrium
   means. Views are the owner's macro judgement; this module never
   invents them.

Market weights: ETF total assets (AUM) from Yahoo Finance is used as a
free proxy for market cap weights. AUM measures fund size, not the
float of the underlying index, so it is an imperfect but documented and
freely available stand in. When AUM cannot be fetched (offline mode, or
fewer than 80% of tickers resolve) the module falls back to inverse
volatility weights and labels the source accordingly; inverse vol is a
neutral, view free heuristic, not a market implied quantity.

All return vectors inside the maths are EXCESS returns; rf is added
back at the boundary so the rest of the engine keeps working in total
return space.
"""
from __future__ import annotations

import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import yaml

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
AUM_CACHE = os.path.join(DATA_DIR, "market_weights.csv")

DEFAULT_DELTA = 2.5          # standard literature value, used when the estimate is unusable
DELTA_BOUNDS = (1.0, 10.0)   # sanity band for estimated risk aversion
MIN_AUM_COVERAGE = 0.8       # below this fraction of tickers resolved, fall back


# ----------------------------------------------------------------------
# Market weights
# ----------------------------------------------------------------------

def _inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    iv = 1.0 / returns.std()
    return iv / iv.sum()


def _fetch_aum(tickers: list[str], cache_days: int) -> pd.Series | None:
    """ETF total assets from Yahoo, cached in data/market_weights.csv so a
    monthly rerun doesn't redo 18 slow info calls. Returns None on poor
    coverage; the caller falls back to inverse vol.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(AUM_CACHE):
        cached = pd.read_csv(AUM_CACHE, index_col=0)
        age = (date.today() - date.fromisoformat(cached["asof"].iloc[0])).days
        if age <= cache_days and set(tickers) <= set(cached.index):
            return cached.loc[tickers, "total_assets"]

    import yfinance as yf
    aum = {}
    for t in tickers:
        try:
            ta = yf.Ticker(t).get_info().get("totalAssets")
            if ta:
                aum[t] = float(ta)
        except Exception:
            continue
    if len(aum) < MIN_AUM_COVERAGE * len(tickers):
        return None
    s = pd.Series(aum).reindex(tickers)
    s = s.fillna(s.median())  # the odd missing fund gets a neutral middle size
    pd.DataFrame({"total_assets": s, "asof": date.today().isoformat()}).to_csv(AUM_CACHE)
    return s


def market_weights(tickers: list[str], returns: pd.DataFrame,
                   offline: bool = False, cache_days: int = 7) -> tuple[pd.Series, str]:
    """Return (weights summing to one, source label)."""
    if not offline:
        try:
            aum = _fetch_aum(tickers, cache_days)
            if aum is not None:
                return aum / aum.sum(), "ETF AUM (Yahoo totalAssets, cap weight proxy)"
            warnings.warn("AUM coverage too thin; using inverse volatility weights.")
        except Exception as exc:
            warnings.warn(f"AUM fetch failed ({exc}); using inverse volatility weights.")
    return _inverse_vol_weights(returns[tickers]), "inverse volatility (fallback proxy)"


# ----------------------------------------------------------------------
# Equilibrium (the prior)
# ----------------------------------------------------------------------

def implied_risk_aversion(benchmark: pd.Series, rf: float, periods: int = 252) -> float:
    """delta = market excess return / market variance, clamped to a sane
    band. Falls back to the literature default 2.5 if the estimate is
    degenerate (e.g. negative mean over the sample).
    """
    mu = float(benchmark.mean()) * periods
    var = float(benchmark.var()) * periods
    if var <= 0:
        return DEFAULT_DELTA
    delta = (mu - rf) / var
    if not np.isfinite(delta) or delta <= 0:
        return DEFAULT_DELTA
    return float(np.clip(delta, *DELTA_BOUNDS))


def equilibrium_returns(cov: pd.DataFrame, w_mkt: pd.Series, delta: float) -> pd.Series:
    """pi = delta * Sigma * w_mkt (annualised EXCESS returns)."""
    w = w_mkt.reindex(cov.index).fillna(0.0)
    return pd.Series(delta * (cov.values @ w.values), index=cov.index, name="equilibrium")


# ----------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------

def load_views(path: str, tickers: list[str]) -> list[dict]:
    """Parse views.yaml. Unknown tickers fail loudly rather than silently
    distorting the posterior. Returns [] when the file is absent or has
    no active views (the shipped default).
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    views = doc.get("views") or []
    for v in views:
        for key in ("name", "confidence"):
            if key not in v:
                raise ValueError(f"View missing required field '{key}': {v}")
        involved = (v.get("assets") or []) + (v.get("long") or []) + (v.get("short") or [])
        unknown = [t for t in involved if t not in tickers]
        if unknown:
            raise ValueError(f"View '{v['name']}' references tickers not in the "
                             f"universe: {unknown}")
        if not involved:
            raise ValueError(f"View '{v['name']}' names no assets.")
    return views


def view_matrices(views: list[dict], tickers: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build P (k x n pick matrix), Q (k expected excess returns) and the
    per view confidences. Absolute views: equal positive weights over
    'assets', Q = 'excess_return'. Relative views: 'long' minus 'short'
    legs (equal weights within each leg, +1 and -1 totals), Q =
    'outperformance'. A 'short' leg only expresses a relative VIEW; the
    portfolio itself stays long only.
    """
    n, k = len(tickers), len(views)
    idx = {t: i for i, t in enumerate(tickers)}
    P = np.zeros((k, n))
    Q = np.zeros(k)
    conf = np.zeros(k)
    for r, v in enumerate(views):
        if v.get("assets"):  # absolute
            for t in v["assets"]:
                P[r, idx[t]] = 1.0 / len(v["assets"])
            Q[r] = float(v["excess_return"])
        else:  # relative
            for t in v["long"]:
                P[r, idx[t]] = 1.0 / len(v["long"])
            for t in v["short"]:
                P[r, idx[t]] = -1.0 / len(v["short"])
            Q[r] = float(v["outperformance"])
        conf[r] = float(np.clip(v["confidence"], 0.01, 0.99))
    return P, Q, conf


def posterior_returns(pi: pd.Series, cov: pd.DataFrame, views: list[dict],
                      tau: float) -> pd.Series:
    """Black Litterman posterior excess returns.

    mu = pi + tau*Sigma*P' (P*tau*Sigma*P' + Omega)^-1 (Q - P*pi)

    Omega is diagonal with omega_i = (P tau Sigma P')_ii * (1 - c_i)/c_i,
    the He Litterman convention where confidence c in (0, 1) scales the
    view's error variance: c -> 1 means the view is near certain, c -> 0
    means it barely moves the prior. With no views the posterior IS the
    prior, exactly.
    """
    if not views:
        return pi.copy()
    tickers = list(cov.index)
    P, Q, conf = view_matrices(views, tickers)
    ts = tau * cov.values
    pts = P @ ts @ P.T
    omega = np.diag(np.diag(pts) * (1.0 - conf) / conf)
    adj = ts @ P.T @ np.linalg.solve(pts + omega, Q - P @ pi.values)
    return pd.Series(pi.values + adj, index=pi.index, name="bl_posterior")


# ----------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------

def black_litterman_returns(returns: pd.DataFrame, cov: pd.DataFrame,
                            benchmark: pd.Series, rf: float, cfg: dict,
                            offline: bool = False, periods: int = 252) -> dict:
    """Full BL pipeline. Returns total (not excess) return Series so the
    optimiser and reporting layers stay in their usual space.
    """
    bl_cfg = cfg.get("black_litterman", {})
    tau = float(bl_cfg.get("tau", 0.05))
    views_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              bl_cfg.get("views_file", "views.yaml"))
    tickers = list(cov.index)

    w_mkt, source = market_weights(tickers, returns, offline,
                                   int(bl_cfg.get("market_weights_cache_days", 7)))
    delta = implied_risk_aversion(benchmark, rf, periods)
    pi = equilibrium_returns(cov, w_mkt, delta)
    views = load_views(views_file, tickers)
    mu = posterior_returns(pi, cov, views, tau)
    return {
        "expected": mu + rf,          # total returns for the optimiser
        "equilibrium": pi + rf,       # total returns for reporting
        "market_weights": w_mkt,
        "weights_source": source,
        "delta": delta,
        "tau": tau,
        "n_views": len(views),
        "views": views,
    }
