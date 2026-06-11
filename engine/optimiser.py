"""Constrained optimisation: Monte Carlo simulation plus exact SLSQP.

The Monte Carlo cloud is the visual, intuitive layer (50k random
portfolios). SLSQP then solves the same problem exactly so the reported
optimal weights aren't hostage to sampling luck. Both respect:
  - long only bounds with a per ETF maximum
  - full investment (weights sum to one)
  - macro basket min/max ranges
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .stats import portfolio_performance


# ----------------------------------------------------------------------
# Constraint construction
# ----------------------------------------------------------------------

def basket_matrices(tickers: list[str], baskets: dict) -> list[dict]:
    """For each basket return its indicator vector and min/max."""
    out = []
    for name, b in baskets.items():
        ind = np.array([1.0 if t in b["tickers"] else 0.0 for t in tickers])
        if ind.sum() == 0:
            continue  # basket has no tickers present in the universe
        out.append({"name": name, "ind": ind,
                    "min": float(b.get("min", 0.0)), "max": float(b.get("max", 1.0))})
    return out


def feasible_mask(weights: np.ndarray, baskets: list[dict]) -> np.ndarray:
    """Vectorised basket feasibility check for an (n_sims, n_assets) array."""
    ok = np.ones(weights.shape[0], dtype=bool)
    for b in baskets:
        tot = weights @ b["ind"]
        ok &= (tot >= b["min"] - 1e-9) & (tot <= b["max"] + 1e-9)
    return ok


# ----------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------

def monte_carlo(exp_ret: pd.Series, cov: pd.DataFrame, rf: float,
                n_sims: int, max_pos: float, baskets: list[dict],
                seed: int = 7) -> dict:
    """Sample random long only portfolios, cap single positions, and
    evaluate return/vol/Sharpe. Returns the full cloud plus the best
    basket feasible portfolio found by simulation.
    """
    rng = np.random.default_rng(seed)
    n = len(exp_ret)
    w = rng.dirichlet(np.ones(n), size=n_sims)

    # Enforce the per ETF cap by iterative clip and renormalise
    for _ in range(50):
        over = w > max_pos
        if not over.any():
            break
        w = np.minimum(w, max_pos)
        deficit = 1.0 - w.sum(axis=1, keepdims=True)
        room = np.maximum(max_pos - w, 0.0)
        room_tot = room.sum(axis=1, keepdims=True)
        w = w + room * np.divide(deficit, room_tot, out=np.zeros_like(room_tot),
                                 where=room_tot > 0)
    w = w / w.sum(axis=1, keepdims=True)

    mu, sigma = exp_ret.values, cov.values
    rets = w @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, sigma, w))
    sharpes = np.where(vols > 0, (rets - rf) / vols, 0.0)
    feas = feasible_mask(w, baskets)

    best_idx = int(np.argmax(np.where(feas, sharpes, -np.inf))) if feas.any() else None
    return {
        "weights": w, "returns": rets, "vols": vols, "sharpes": sharpes,
        "feasible": feas,
        "best_mc_weights": w[best_idx] if best_idx is not None else None,
        "n_feasible": int(feas.sum()),
    }


# ----------------------------------------------------------------------
# Exact optimisation (SLSQP)
# ----------------------------------------------------------------------

def _scipy_constraints(n: int, baskets: list[dict]) -> list[dict]:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    for b in baskets:
        ind = b["ind"]
        cons.append({"type": "ineq", "fun": lambda w, i=ind, lo=b["min"]: w @ i - lo})
        cons.append({"type": "ineq", "fun": lambda w, i=ind, hi=b["max"]: hi - w @ i})
    return cons


def _solve(objective, exp_ret: pd.Series, cov: pd.DataFrame, max_pos: float,
           baskets: list[dict], x0: np.ndarray | None = None) -> np.ndarray:
    n = len(exp_ret)
    bounds = [(0.0, max_pos)] * n
    cons = _scipy_constraints(n, baskets)
    if x0 is None:
        x0 = np.full(n, 1.0 / n)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    if not res.success:
        # Retry from a feasible-ish corner before giving up
        res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 2000, "ftol": 1e-9})
    if not res.success:
        raise RuntimeError(f"Optimiser failed: {res.message}. "
                           "Check basket mins don't exceed 1 or conflict with max_position.")
    w = np.clip(res.x, 0, max_pos)
    return w / w.sum()


def max_sharpe(exp_ret: pd.Series, cov: pd.DataFrame, rf: float, max_pos: float,
               baskets: list[dict], x0: np.ndarray | None = None) -> np.ndarray:
    def neg_sharpe(w):
        r, v, s = portfolio_performance(w, exp_ret.values, cov.values, rf)
        return -s
    return _solve(neg_sharpe, exp_ret, cov, max_pos, baskets, x0)


def min_volatility(exp_ret: pd.Series, cov: pd.DataFrame, max_pos: float,
                   baskets: list[dict]) -> np.ndarray:
    def vol(w):
        return float(np.sqrt(w @ cov.values @ w))
    return _solve(vol, exp_ret, cov, max_pos, baskets)


def efficient_frontier(exp_ret: pd.Series, cov: pd.DataFrame, max_pos: float,
                       baskets: list[dict], n_points: int = 40) -> pd.DataFrame:
    """Minimum volatility at each target return between the min vol
    portfolio's return and the highest achievable constrained return.
    """
    w_minv = min_volatility(exp_ret, cov, max_pos, baskets)
    r_low = float(w_minv @ exp_ret.values)

    def neg_ret(w):
        return -float(w @ exp_ret.values)
    w_maxr = _solve(neg_ret, exp_ret, cov, max_pos, baskets)
    r_high = float(w_maxr @ exp_ret.values)

    rows = []
    for target in np.linspace(r_low, r_high, n_points):
        def vol(w):
            return float(np.sqrt(w @ cov.values @ w))
        n = len(exp_ret)
        cons = _scipy_constraints(n, baskets)
        cons.append({"type": "eq", "fun": lambda w, t=target: w @ exp_ret.values - t})
        res = minimize(vol, np.full(n, 1.0 / n), method="SLSQP",
                       bounds=[(0.0, max_pos)] * n, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-10})
        if res.success:
            rows.append({"ret": target, "vol": float(np.sqrt(res.x @ cov.values @ res.x))})
    return pd.DataFrame(rows)
