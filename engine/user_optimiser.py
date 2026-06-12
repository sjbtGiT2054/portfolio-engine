"""Phase 6: optimisation over a user entered universe.

Takes the holdings the Portfolio Analyzer already fetched and USD
converted, and solves four alternative weightings over exactly those
tickers, reusing the SLSQP machinery in engine/optimiser.py (no
duplicated maths, just an empty basket list):

- max Sharpe          highest (return - rf) / volatility in sample
- min volatility      lowest historical volatility
- max diversification minimises w' Corr w, the correlation weighted
                      concentration: the min vol portfolio you would get
                      if every asset had identical volatility, so it
                      loads on the least correlated names
- equal weight        the naive 1/n baseline every optimiser must beat

Constraints everywhere: long only, fully invested, user adjustable
single position cap. All outputs are historical in sample analytics
under stated assumptions; estimation error means they are illustrations,
not forecasts, and the wording downstream must keep saying so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt_mod
from . import optimiser, robustness, stats

MIN_TICKERS = 2
MIN_DAYS = 252            # one year of common history to optimise
BACKTEST_MIN_DAYS = 504   # two years before the walk forward is worth running
BACKTEST_MAX_TICKERS = 15 # runtime cap for the public app
DEFAULT_MAX_POS = 0.25
MC_SIMS = 10_000


def validate_universe(rets: pd.DataFrame) -> None:
    """Raise ValueError with a user readable message when the universe is
    too small or too short to optimise."""
    if rets.shape[1] < MIN_TICKERS:
        raise ValueError("Optimisation needs at least 2 valid tickers; this "
                         f"universe has {rets.shape[1]}.")
    if len(rets) < MIN_DAYS:
        raise ValueError("Optimisation needs at least one year of common price "
                         f"history across all tickers; these share only "
                         f"{len(rets)} trading days.")


def weighted_avg_correlation(w: np.ndarray, corr: pd.DataFrame) -> float:
    """Weight weighted average pairwise correlation:
    sum_{i!=j} w_i w_j rho_ij / sum_{i!=j} w_i w_j. For a single nonzero
    weight there are no pairs and the result is nan."""
    W = np.outer(w, w)
    off = ~np.eye(len(w), dtype=bool)
    denom = W[off].sum()
    return float((W * corr.values)[off].sum() / denom) if denom > 1e-12 else float("nan")


def optimise(prices_usd: pd.DataFrame, rf: float, max_pos: float = DEFAULT_MAX_POS,
             periods: int = 252, shrinkage: float = 0.10,
             n_sims: int = MC_SIMS) -> dict:
    """Solve the four portfolios plus the Monte Carlo cloud and the
    universe's own efficient frontier. Returns weights as Series indexed
    by ticker. If the cap is infeasible for n tickers (max_pos < 1/n) it
    is raised to 1/n and noted."""
    rets = prices_usd.pct_change().dropna()
    validate_universe(rets)
    n = rets.shape[1]

    note = None
    if max_pos < 1.0 / n - 1e-9:
        note = (f"A {max_pos:.0%} cap cannot sum to 100% across {n} tickers; "
                f"the cap was raised to {1.0 / n:.1%} (equal weight is then the "
                "only feasible portfolio at exactly that cap).")
        max_pos = 1.0 / n

    exp_ret = stats.expected_returns(rets, periods)
    cov = stats.covariance_matrix(rets, periods, shrinkage)
    corr = rets.corr()

    w_ms = optimiser.max_sharpe(exp_ret, cov, rf, max_pos, [])
    w_mv = optimiser.min_volatility(exp_ret, cov, max_pos, [])

    def corr_concentration(w):
        return float(w @ corr.values @ w)
    w_md = optimiser._solve(corr_concentration, exp_ret, cov, max_pos, [])

    weights = {
        "max_sharpe": pd.Series(w_ms, index=rets.columns),
        "min_vol": pd.Series(w_mv, index=rets.columns),
        "max_div": pd.Series(w_md, index=rets.columns),
        "equal_weight": pd.Series(np.full(n, 1.0 / n), index=rets.columns),
    }
    mc = optimiser.monte_carlo(exp_ret, cov, rf, n_sims, max_pos, [])
    frontier = optimiser.efficient_frontier(exp_ret, cov, max_pos, [])

    return {"weights": weights, "rets": rets, "exp_ret": exp_ret, "cov": cov,
            "corr": corr, "mc": mc, "frontier": frontier,
            "max_pos": max_pos, "cap_note": note, "rf": rf, "periods": periods}


def sensitivity(opt: dict) -> pd.DataFrame:
    """Plus/minus 1% expected return sensitivity of the user's max Sharpe
    solution, reusing the engine robustness layer (empty baskets)."""
    return robustness.sensitivity_report(
        opt["exp_ret"], opt["cov"], opt["rf"],
        {"max_position": opt["max_pos"]}, [],
        opt["weights"]["max_sharpe"].values)


def highest_corr_pair(corr: pd.DataFrame) -> tuple[str, str, float]:
    c = corr.where(~np.eye(len(corr), dtype=bool))
    t1 = c.max().idxmax()
    t2 = c[t1].idxmax()
    return t1, t2, float(c.loc[t1, t2])


def backtest_user(opt: dict, w_current: pd.Series, bench_daily: pd.Series,
                  cost_bps: int = 10) -> dict:
    """Walk forward backtest over the user's tickers, reusing
    engine/backtest.py: the max Sharpe strategy re-optimised monthly on
    trailing data only, vs the current weights held static, vs the
    benchmark. Gated for runtime on the public app."""
    rets = opt["rets"]
    if rets.shape[1] > BACKTEST_MAX_TICKERS:
        return {"skipped": f"Backtest is capped at {BACKTEST_MAX_TICKERS} tickers "
                           f"for runtime; this universe has {rets.shape[1]}."}
    if len(rets) < BACKTEST_MIN_DAYS:
        return {"skipped": "Backtest needs at least two years of common history; "
                           f"these tickers share only {len(rets)} trading days."}

    cfg = {"settings": {"annualisation": opt["periods"],
                        "max_position": opt["max_pos"],
                        "shrinkage": 0.10,
                        "transaction_cost_bps": cost_bps}}
    try:
        bt = bt_mod.run_backtest(rets, bench_daily, opt["rf"], cfg, [])
    except RuntimeError as exc:
        return {"skipped": f"Backtest could not run: {exc}"}

    idx = bt["daily"].index
    w = (w_current / w_current.sum()).reindex(rets.columns).fillna(0.0)
    static = pd.Series(rets.loc[idx].values @ w.values, index=idx)
    bench = bench_daily.reindex(idx).dropna()

    def row(daily: pd.Series) -> dict:
        ann = stats.annualise_series(daily, opt["periods"])
        mdd, _ = stats.max_drawdown(daily)
        sharpe = (ann["cagr"] - opt["rf"]) / ann["vol"] if ann["vol"] > 0 else 0.0
        return {"cagr": ann["cagr"], "vol": ann["vol"], "sharpe": sharpe,
                "max_drawdown": mdd}

    summary = pd.DataFrame({
        "current weights (static)": row(static),
        "max Sharpe (monthly re-optimised)": row(bt["daily"]),
        "benchmark": row(bench),
    }).T
    return {"daily": pd.DataFrame({"current weights (static)": static,
                                   "max Sharpe (monthly re-optimised)": bt["daily"],
                                   "benchmark": bench}),
            "summary": summary,
            "n_rebalances": bt["summary"]["n_rebalances"],
            "start": bt["summary"]["start"], "end": bt["summary"]["end"]}
