"""Machine readable exports for the Streamlit dashboard.

Everything the dashboard needs lands in outputs/dashboard/ as JSON and
CSV. The dashboard only ever reads these files (plus config.yaml and
current_holdings.csv); it never imports the optimisation pipeline, so
dashboard work can't break the engine.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

MC_EXPORT_ROWS = 8000  # downsample the 50k cloud so the CSV stays light


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)  # dates and anything else


def write_dashboard_data(ctx: dict, cfg: dict, frontier: pd.DataFrame,
                         capm: pd.DataFrame, minv_point: tuple,
                         corr: pd.DataFrame, out_dir: str) -> str:
    dash_dir = os.path.join(out_dir, "dashboard")
    os.makedirs(dash_dir, exist_ok=True)

    # ----- CSV artifacts -------------------------------------------------
    frontier.to_csv(os.path.join(dash_dir, "frontier.csv"), index=False)
    capm.round(6).to_csv(os.path.join(dash_dir, "capm.csv"))
    corr.round(4).to_csv(os.path.join(dash_dir, "correlation.csv"))

    mc = ctx["mc"]
    n = len(mc["returns"])
    rng = np.random.default_rng(7)
    keep = rng.choice(n, size=min(MC_EXPORT_ROWS, n), replace=False)
    if mc["n_feasible"]:  # make sure the best feasible portfolio survives sampling
        best = int(np.argmax(np.where(mc["feasible"], mc["sharpes"], -np.inf)))
        keep = np.unique(np.append(keep, best))
    pd.DataFrame({
        "ret": mc["returns"][keep], "vol": mc["vols"][keep],
        "sharpe": mc["sharpes"][keep], "feasible": mc["feasible"][keep],
    }).round(6).to_csv(os.path.join(dash_dir, "mc_cloud.csv"), index=False)

    if ctx.get("backtest"):
        bt = ctx["backtest"]
        pd.DataFrame({
            "portfolio": bt["daily"], "benchmark": bt["benchmark_daily"],
            "drawdown": bt["drawdown"],
        }).round(6).to_csv(os.path.join(dash_dir, "backtest_daily.csv"),
                           index_label="date")

    # ----- Phase 2: BL comparison and robustness tables -------------------
    if ctx.get("bl"):
        bl = ctx["bl"]
        pd.DataFrame({
            "historical": ctx["exp_ret_hist"],
            "equilibrium": bl["equilibrium"],
            "bl_posterior": bl["expected"],
            "market_weight": bl["market_weights"],
        }).round(6).to_csv(os.path.join(dash_dir, "bl_comparison.csv"),
                           index_label="ticker")
    if ctx.get("stability") is not None:
        ctx["stability"].round(6).to_csv(os.path.join(dash_dir, "weight_stability.csv"),
                                         index_label="ticker")
    if ctx.get("sensitivity") is not None:
        ctx["sensitivity"].round(6).to_csv(os.path.join(dash_dir, "sensitivity.csv"),
                                           index_label="ticker")

    # ----- results.json ---------------------------------------------------
    w = ctx["weights"]
    bl = ctx.get("bl")
    results = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_note": ctx["data_note"],
        "benchmark": ctx["benchmark"],
        "estimation": {
            "expected_returns": ctx.get("ret_mode", "historical"),
            "covariance_estimator": ctx.get("cov_estimator", "simple"),
            "lw_intensity": ctx.get("lw_intensity"),
            "bl_available": bl is not None,
            "bl_n_views": bl["n_views"] if bl else None,
            "bl_delta": bl["delta"] if bl else None,
            "bl_tau": bl["tau"] if bl else None,
            "bl_weights_source": bl["weights_source"] if bl else None,
        },
        "risk_free": {"rate": ctx["rf"]["rate"], "source": ctx["rf"]["source"],
                      "ten_year": ctx["rf"].get("ten_year")},
        "universe": cfg["universe"],
        "metrics": {
            "exp_return": ctx["opt_ret"], "volatility": ctx["opt_vol"],
            "sharpe": ctx["opt_sharpe"], "sortino": ctx["sortino"],
            "beta": ctx["port_beta"], "max_drawdown": ctx["mdd"],
            "var_95": ctx["var"]["var_hist"], "cvar_95": ctx["var"]["cvar"],
            "mc_best_sharpe": ctx["mc_best_sharpe"],
            "mc_n_feasible": ctx["mc"]["n_feasible"],
        },
        "weights": {t: float(v) for t, v in w.items() if v > 0.0005},
        "min_vol_point": {"ret": minv_point[0], "vol": minv_point[1]},
        "baskets": [{
            "name": b["name"], "min": b["min"], "max": b["max"],
            "allocation": float(w.values @ b["ind"]),
            "tickers": [t for t, f in zip(w.index, b["ind"]) if f],
        } for b in ctx["baskets"]],
        "limit_checks": ctx["limit_checks"],
        "stress": ctx["stress"].to_dict(orient="records"),
        "worst": ctx["worst"],
        "correlation_monitor": ctx["corrmon"],
        "capm_attrs": {"market_return": capm.attrs.get("market_return"),
                       "rf": capm.attrs.get("rf")},
        "backtest_summary": ctx["backtest"]["summary"] if ctx.get("backtest") else None,
    }
    path = os.path.join(dash_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    return path
