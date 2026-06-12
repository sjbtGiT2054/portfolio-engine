"""Portfolio Engine — entry point.

Usage (from the portfolio_engine folder):
    python main.py              full run: optimise, charts, backtest, tear sheet
    python main.py --no-backtest    skip the walk forward backtest (faster)
    python main.py --offline        synthetic data demo, no internet needed

Outputs land in outputs/charts/ and outputs/reports/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from engine import backtest as bt_mod
from engine import black_litterman as bl_mod
from engine import capm as capm_mod
from engine import charts, data, export, optimiser, report, risk, robustness, stats
from engine.riskfree import get_risk_free_rate

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")
CHARTS_DIR = os.path.join(OUT_DIR, "charts")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")


def load_config() -> dict:
    with open(os.path.join(PROJECT_DIR, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Macro constrained Sharpe maximising portfolio engine")
    ap.add_argument("--offline", action="store_true", help="synthetic data demo, no internet")
    ap.add_argument("--no-backtest", action="store_true", help="skip walk forward backtest")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config()
    s = cfg["settings"]
    tickers = list(cfg["universe"].keys())
    benchmark_ticker = s["benchmark"]

    print(f"Universe: {len(tickers)} ETFs | lookback {s['lookback_years']}y | "
          f"{s['num_simulations']:,} simulations | mode: {'OFFLINE DEMO' if args.offline else 'live data'}")

    # ----- Data ---------------------------------------------------------
    all_tickers = sorted(set(tickers + [benchmark_ticker]))
    prices = data.fetch_prices(all_tickers, s["lookback_years"], offline=args.offline)
    rets_all = data.daily_returns(prices)
    universe = [t for t in tickers if t in rets_all.columns]
    rets = rets_all[universe]
    bench = rets_all[benchmark_ticker] if benchmark_ticker in rets_all.columns else rets.mean(axis=1)
    print(f"Data: {len(rets)} trading days, {rets.index[0].date()} to {rets.index[-1].date()}, "
          f"{len(universe)} usable tickers")

    rf = get_risk_free_rate(cfg, offline=args.offline)
    print(f"Risk free rate: {rf['rate']:.2%} ({rf['source']})")

    # ----- Estimation ---------------------------------------------------
    cov_estimator = s.get("covariance_estimator", "simple")
    if cov_estimator == "ledoit_wolf":
        cov, lw_intensity = robustness.ledoit_wolf_cov(rets, s["annualisation"])
        print(f"Covariance: Ledoit Wolf shrinkage, intensity {lw_intensity:.2f}")
    else:
        cov = stats.covariance_matrix(rets, s["annualisation"], s.get("shrinkage", 0.0))
        lw_intensity = None

    exp_ret_hist = stats.expected_returns(rets, s["annualisation"])
    ret_mode = s.get("expected_returns", "historical")
    bl = None
    try:
        bl = bl_mod.black_litterman_returns(rets, cov, bench, rf["rate"], cfg,
                                            offline=args.offline,
                                            periods=s["annualisation"])
    except Exception as exc:
        if ret_mode == "black_litterman":
            raise  # explicitly requested mode must not silently degrade
        print(f"  (Black Litterman comparison unavailable: {exc})")
    if ret_mode == "black_litterman":
        exp_ret = bl["expected"]
        print(f"Expected returns: Black Litterman | {bl['n_views']} active views | "
              f"delta {bl['delta']:.2f} | market weights: {bl['weights_source']}")
        if bl["n_views"] == 0:
            print("  No active views in views.yaml, so these are pure market "
                  "implied equilibrium returns.")
    else:
        exp_ret = exp_ret_hist

    baskets = optimiser.basket_matrices(universe, cfg.get("baskets", {}))

    # ----- Monte Carlo + exact optimisation -----------------------------
    print("Running Monte Carlo simulation...")
    mc = optimiser.monte_carlo(exp_ret, cov, rf["rate"], s["num_simulations"],
                               s["max_position"], baskets)
    print(f"  {mc['n_feasible']:,} of {s['num_simulations']:,} simulated portfolios satisfy basket limits")

    print("Solving exact constrained max Sharpe (SLSQP)...")
    w_opt = optimiser.max_sharpe(exp_ret, cov, rf["rate"], s["max_position"], baskets,
                                 x0=mc["best_mc_weights"])
    w_minv = optimiser.min_volatility(exp_ret, cov, s["max_position"], baskets)
    frontier = optimiser.efficient_frontier(exp_ret, cov, s["max_position"], baskets)

    opt_ret, opt_vol, opt_sharpe = stats.portfolio_performance(
        w_opt, exp_ret.values, cov.values, rf["rate"])
    minv_ret, minv_vol, _ = stats.portfolio_performance(
        w_minv, exp_ret.values, cov.values, rf["rate"])
    mc_best_sharpe = float(mc["sharpes"][mc["feasible"]].max()) if mc["n_feasible"] else float("nan")

    weights = pd.Series(w_opt, index=universe).round(6)
    daily = stats.portfolio_daily_series(w_opt, rets)

    # ----- Robustness diagnostics ----------------------------------------
    rob_cfg = cfg.get("robustness", {})
    stability, sensitivity = None, None
    if rob_cfg.get("enabled", True):
        print("Robustness: bootstrap resampling and sensitivity analysis...")
        n_boot = int(rob_cfg.get("n_bootstrap", 100))
        bump = float(rob_cfg.get("sensitivity_bump", 0.01))
        try:
            stability = robustness.resampled_weights(rets, rf["rate"], s, baskets,
                                                     weights, n_boot=n_boot)
            sensitivity = robustness.sensitivity_report(exp_ret, cov, rf["rate"],
                                                        s, baskets, w_opt, bump=bump)
            n_frag = int(sensitivity["fragile"].sum())
            print(f"  {stability['n_samples'].iloc[0]} of {n_boot} bootstrap solves "
                  f"succeeded | {n_frag} position(s) flagged fragile")
        except RuntimeError as exc:
            print(f"  Robustness diagnostics skipped: {exc}")

    # ----- CAPM, risk ----------------------------------------------------
    capm = capm_mod.capm_table(rets, bench, exp_ret, rf["rate"], s["annualisation"])
    port_beta = capm_mod.portfolio_beta(w_opt, capm)
    mdd, _ = stats.max_drawdown(daily)
    ctx = {
        "weights": weights, "baskets": baskets, "benchmark": benchmark_ticker,
        "opt_ret": opt_ret, "opt_vol": opt_vol, "opt_sharpe": opt_sharpe,
        "sortino": stats.sortino_ratio(daily, rf["rate"], s["annualisation"]),
        "mdd": mdd, "var": stats.value_at_risk(daily),
        "mc": mc, "mc_best_sharpe": mc_best_sharpe,
        "rf": rf, "port_beta": port_beta,
        "limit_checks": risk.limit_checks(weights, s["max_position"], baskets),
        "stress": risk.beta_stress_tests(port_beta),
        "worst": risk.historical_worst(daily),
        "corrmon": risk.correlation_monitor(rets),
        "data_note": ("synthetic demo data" if args.offline else
                      f"Yahoo Finance adjusted closes, {rets.index[0].date()} to {rets.index[-1].date()}"),
        "rebalance": report.rebalance_recommendation(weights, PROJECT_DIR),
        "backtest": None, "daily": daily,
        "ret_mode": ret_mode, "cov_estimator": cov_estimator,
        "lw_intensity": lw_intensity,
        "exp_ret_hist": exp_ret_hist, "bl": bl,
        "stability": stability, "sensitivity": sensitivity,
    }

    # ----- Charts ---------------------------------------------------------
    print("Drawing charts...")
    charts.frontier_chart(mc, frontier, (opt_ret, opt_vol), (minv_ret, minv_vol),
                          rf["rate"], CHARTS_DIR)
    charts.sml_chart(capm, port_beta, opt_ret, CHARTS_DIR)
    charts.weights_chart(weights, baskets, CHARTS_DIR)
    charts.basket_chart(weights, baskets, CHARTS_DIR)
    charts.correlation_chart(rets, CHARTS_DIR)

    # ----- Backtest --------------------------------------------------------
    if not args.no_backtest:
        print("Running walk forward backtest (this is the slow part)...")
        try:
            bt = bt_mod.run_backtest(rets, bench, rf["rate"], cfg, baskets)
            ctx["backtest"] = bt
            charts.backtest_chart(bt, CHARTS_DIR)
            bt["weights_history"].round(4).to_csv(os.path.join(OUT_DIR, "backtest_weights_history.csv"))
        except RuntimeError as exc:
            print(f"  Backtest skipped: {exc}")

    # ----- Report -----------------------------------------------------------
    export.write_dashboard_data(ctx, cfg, frontier, capm, (minv_ret, minv_vol),
                                rets.corr(), OUT_DIR)
    weights[weights > 0.001].rename("weight").to_csv(os.path.join(OUT_DIR, "optimal_weights.csv"))
    path = report.write_report(ctx, REPORTS_DIR)

    print("\n" + "=" * 62)
    print(f"OPTIMAL PORTFOLIO   Sharpe {opt_sharpe:.2f} | ret {opt_ret:.1%} | vol {opt_vol:.1%} | beta {port_beta:.2f}")
    print("=" * 62)
    for t, v in weights[weights > 0.001].sort_values(ascending=False).items():
        print(f"  {t:<6} {v:>7.1%}   {cfg['universe'].get(t, '')}")
    print("=" * 62)
    fails = [c for c in ctx["limit_checks"] if not c["pass"]]
    print(f"Risk limit checks: {'ALL PASS' if not fails else f'{len(fails)} FAILED'}")
    print(f"Tear sheet: {os.path.relpath(path, PROJECT_DIR)}")
    print(f"Charts:     outputs/charts/   |   Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
