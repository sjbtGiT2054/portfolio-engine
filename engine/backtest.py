"""Walk forward backtest with monthly rebalancing.

Each month end the engine re-estimates returns and covariance on a
trailing window (training data only), re-optimises max Sharpe under the
same basket constraints, and holds those weights for the following month.
Every decision uses only information available at the time, so the whole
equity curve is out of sample. Transaction costs are charged on turnover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimiser, stats


def run_backtest(returns: pd.DataFrame, benchmark: pd.Series, rf: float,
                 cfg: dict, baskets: list[dict],
                 train_years: float = 3.0) -> dict:
    periods = cfg["settings"]["annualisation"]
    max_pos = cfg["settings"]["max_position"]
    shrink = cfg["settings"].get("shrinkage", 0.0)
    cost = cfg["settings"].get("transaction_cost_bps", 0) / 10000.0

    month_ends = returns.resample("ME").last().index
    train_len = int(train_years * periods)

    port_rets, weight_log, prev_w = [], [], None
    for i, me in enumerate(month_ends[:-1]):
        train = returns.loc[:me].tail(train_len)
        if len(train) < periods:  # need at least a year of training data
            continue
        exp_ret = stats.expected_returns(train, periods)
        cov = stats.covariance_matrix(train, periods, shrink)
        try:
            w = optimiser.max_sharpe(exp_ret, cov, rf, max_pos, baskets, x0=prev_w)
        except RuntimeError:
            w = prev_w if prev_w is not None else np.full(returns.shape[1], 1 / returns.shape[1])

        nxt = returns.loc[(returns.index > me) & (returns.index <= month_ends[i + 1])]
        if nxt.empty:
            continue
        month_series = pd.Series(nxt.values @ w, index=nxt.index)
        turnover = float(np.abs(w - prev_w).sum()) if prev_w is not None else 1.0
        if not month_series.empty:
            month_series.iloc[0] -= turnover * cost
        port_rets.append(month_series)
        weight_log.append(pd.Series(w, index=returns.columns, name=me))
        prev_w = w

    if not port_rets:
        raise RuntimeError("Backtest produced no periods. Need more history than the training window.")

    daily = pd.concat(port_rets).sort_index()
    bench = benchmark.reindex(daily.index).dropna()
    daily = daily.reindex(bench.index)

    port_stats = stats.annualise_series(daily, periods)
    bench_stats = stats.annualise_series(bench, periods)
    mdd, dd_series = stats.max_drawdown(daily)
    bench_mdd, _ = stats.max_drawdown(bench)

    summary = {
        "start": daily.index[0].date(), "end": daily.index[-1].date(),
        "portfolio": {**port_stats,
                      "sharpe": (port_stats["cagr"] - rf) / port_stats["vol"] if port_stats["vol"] > 0 else 0,
                      "sortino": stats.sortino_ratio(daily, rf, periods),
                      "max_drawdown": mdd},
        "benchmark": {**bench_stats,
                      "sharpe": (bench_stats["cagr"] - rf) / bench_stats["vol"] if bench_stats["vol"] > 0 else 0,
                      "max_drawdown": bench_mdd},
        "n_rebalances": len(weight_log),
        "avg_turnover_cost_bps": cfg["settings"].get("transaction_cost_bps", 0),
    }
    return {"daily": daily, "benchmark_daily": bench, "drawdown": dd_series,
            "weights_history": pd.DataFrame(weight_log), "summary": summary}
