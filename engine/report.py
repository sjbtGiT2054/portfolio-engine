"""Written tear sheet: a dated markdown report assembling every output,
plus a rebalance recommendation vs current_holdings.csv if present.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

HOLDINGS_FILE = "current_holdings.csv"


def rebalance_recommendation(new_weights: pd.Series, project_dir: str) -> pd.DataFrame | None:
    """Compare optimal weights with current_holdings.csv (columns: ticker,weight).
    Returns a trade list, or None if no holdings file exists yet.
    """
    path = os.path.join(project_dir, HOLDINGS_FILE)
    if not os.path.exists(path):
        return None
    cur = pd.read_csv(path).set_index("ticker")["weight"]
    all_t = sorted(set(cur.index) | set(new_weights.index))
    cur = cur.reindex(all_t).fillna(0.0)
    new = new_weights.reindex(all_t).fillna(0.0)
    diff = (new - cur)
    out = pd.DataFrame({"current": cur, "target": new, "trade": diff})
    out["action"] = out["trade"].apply(
        lambda x: "BUY" if x > 0.005 else ("SELL" if x < -0.005 else "HOLD"))
    return out.sort_values("trade", ascending=False)


def write_report(ctx: dict, reports_dir: str) -> str:
    """ctx carries every computed object from main.py."""
    os.makedirs(reports_dir, exist_ok=True)
    today = date.today().isoformat()
    path = os.path.join(reports_dir, f"tear_sheet_{today}.md")

    w = ctx["weights"]
    held = w[w > 0.001].sort_values(ascending=False)
    lines = [
        f"# Portfolio Engine Tear Sheet — {today}",
        "",
        "**Mandate:** long only, equity ETF, macro constrained, monthly rebalanced, "
        "maximising the Sharpe ratio within stated basket limits.",
        f"**Data:** {ctx['data_note']}",
        f"**Risk free rate:** {ctx['rf']['rate']:.2%} — source: {ctx['rf']['source']}"
        + (f" (10y context: {ctx['rf']['ten_year']:.2%})" if ctx['rf'].get('ten_year') else ""),
        "",
        "## Optimal portfolio (exact SLSQP solution)",
        "",
        f"- Expected return: **{ctx['opt_ret']:.2%}**",
        f"- Volatility: **{ctx['opt_vol']:.2%}**",
        f"- Sharpe ratio: **{ctx['opt_sharpe']:.2f}**",
        f"- Sortino ratio: {ctx['sortino']:.2f}",
        f"- Portfolio beta vs {ctx['benchmark']}: {ctx['port_beta']:.2f}",
        f"- Max drawdown (in sample, static weights): {ctx['mdd']:.1%}",
        f"- Daily VaR 95% (historical): {ctx['var']['var_hist']:.2%}  |  CVaR: {ctx['var']['cvar']:.2%}",
        f"- Monte Carlo check: best of {ctx['mc']['n_feasible']:,} feasible simulations "
        f"reached Sharpe {ctx['mc_best_sharpe']:.2f} vs exact {ctx['opt_sharpe']:.2f}",
        "",
        "### Weights",
        "",
        "| ETF | Weight |",
        "|---|---|",
        *[f"| {t} | {v:.1%} |" for t, v in held.items()],
        "",
        "### Macro basket allocation",
        "",
        "| Basket | Allocation | Range |",
        "|---|---|---|",
        *[f"| {b['name']} | {float(w.values @ b['ind']):.1%} | {b['min']:.0%} to {b['max']:.0%} |"
          for b in ctx["baskets"]],
        "",
        "## Risk limit checks",
        "",
        "| Check | Value | Status |",
        "|---|---|---|",
        *[f"| {c['check']} | {c['value']} | {'PASS' if c['pass'] else '**FAIL**'} |"
          for c in ctx["limit_checks"]],
        "",
        "## Stress tests (beta approximation)",
        "",
        "| Scenario | Market shock | Est. portfolio impact |",
        "|---|---|---|",
        *[f"| {r.scenario} | {r.market_shock:.0%} | {r.est_portfolio_impact:.1%} |"
          for r in ctx["stress"].itertuples()],
        "",
        f"Historical worst day: {ctx['worst']['worst_day']:.2%} ({ctx['worst']['worst_day_date']}); "
        f"worst month: {ctx['worst']['worst_month']:.2%} ({ctx['worst']['worst_month_date']})",
        "",
        "## Correlation monitor",
        "",
        f"- Average pairwise correlation, full sample: {ctx['corrmon']['avg_corr_full']:.2f}",
        f"- Recent {ctx['corrmon']['window_days']} days: {ctx['corrmon']['avg_corr_recent']:.2f}"
        + ("  — **WARNING: correlations elevated, diversification weakening**"
           if ctx["corrmon"]["warning"] else ""),
        "",
    ]

    lines += [
        "## Estimation inputs (Phase 2)",
        "",
        f"- Expected returns mode: **{ctx.get('ret_mode', 'historical')}**"
        + (" (no active views, pure market implied equilibrium)"
           if ctx.get("bl") and ctx["bl"]["n_views"] == 0
           and ctx.get("ret_mode") == "black_litterman" else ""),
        f"- Covariance estimator: **{ctx.get('cov_estimator', 'simple')}**"
        + (f" (Ledoit Wolf intensity {ctx['lw_intensity']:.2f})"
           if ctx.get("lw_intensity") is not None else ""),
    ]
    if ctx.get("bl"):
        bl = ctx["bl"]
        lines += [
            f"- BL prior: delta {bl['delta']:.2f}, tau {bl['tau']}, "
            f"market weights from {bl['weights_source']}",
            f"- Active views: {bl['n_views']} "
            + ("(views.yaml is the placeholder awaiting owner input)"
               if bl["n_views"] == 0 else ""),
            "",
            "### Historical vs Black Litterman expected returns",
            "",
            "| ETF | Mkt weight | Historical | Equilibrium | BL posterior |",
            "|---|---|---|---|---|",
            *[f"| {t} | {bl['market_weights'][t]:.1%} | {ctx['exp_ret_hist'][t]:.1%} "
              f"| {bl['equilibrium'][t]:.1%} | {bl['expected'][t]:.1%} |"
              for t in ctx["exp_ret_hist"].index],
            "",
            "Equilibrium returns are reverse optimised from market weights "
            "(pi = delta * Sigma * w). With zero views the posterior equals the "
            "equilibrium; the gap vs historical means shows how much noise the "
            "sample mean carries.",
            "",
        ]
    else:
        lines += ["- Black Litterman comparison unavailable this run.", ""]

    if ctx.get("stability") is not None:
        st = ctx["stability"]
        shown = st[(st["optimal"] > 0.001) | (st["boot_mean"] > 0.01)]
        lines += [
            "## Robustness: bootstrap weight stability",
            "",
            f"{int(st['n_samples'].iloc[0])} bootstrap resamples of the daily "
            "returns, each re-optimised under identical constraints.",
            "",
            "| ETF | Optimal | Bootstrap mean | Bootstrap std |",
            "|---|---|---|---|",
            *[f"| {t} | {r.optimal:.1%} | {r.boot_mean:.1%} | {r.boot_std:.1%} |"
              for t, r in shown.iterrows()],
            "",
            "A large gap between optimal and bootstrap mean, or a large std, "
            "means that position is estimation noise rather than conviction "
            "(Michaud resampled efficiency).",
            "",
        ]
    if ctx.get("sensitivity") is not None:
        se = ctx["sensitivity"]
        shown = se[se["fragile"] | (se["weight"] > 0.001)]
        lines += [
            "## Robustness: expected return sensitivity",
            "",
            "Each ETF's expected return shifted up and down 1%, portfolio "
            "re-optimised; turnover = fraction of the book that trades.",
            "",
            "| ETF | Weight | Turnover +1% | Turnover -1% | Fragile |",
            "|---|---|---|---|---|",
            *[f"| {t} | {r.weight:.1%} | {r.turnover_up:.1%} | {r.turnover_down:.1%} "
              f"| {'**YES**' if r.fragile else 'no'} |"
              for t, r in shown.iterrows()],
            "",
            "Fragile = a 1% estimate change moves over 10% of the portfolio. "
            "Those weights deserve the least trust and the basket caps exist "
            "to contain exactly this.",
            "",
        ]

    if ctx.get("backtest"):
        s = ctx["backtest"]["summary"]
        p, b = s["portfolio"], s["benchmark"]
        lines += [
            "## Walk forward backtest (out of sample)",
            "",
            f"Period {s['start']} to {s['end']}, {s['n_rebalances']} monthly rebalances, "
            f"costs {s['avg_turnover_cost_bps']} bps per unit turnover.",
            "",
            "| Metric | Engine | Benchmark |",
            "|---|---|---|",
            f"| CAGR | {p['cagr']:.2%} | {b['cagr']:.2%} |",
            f"| Volatility | {p['vol']:.2%} | {b['vol']:.2%} |",
            f"| Sharpe | {p['sharpe']:.2f} | {b['sharpe']:.2f} |",
            f"| Sortino | {p['sortino']:.2f} | — |",
            f"| Max drawdown | {p['max_drawdown']:.1%} | {b['max_drawdown']:.1%} |",
            "",
        ]

    if ctx.get("rebalance") is not None:
        rb = ctx["rebalance"]
        lines += [
            "## Rebalance recommendation vs current holdings",
            "",
            "| ETF | Current | Target | Trade | Action |",
            "|---|---|---|---|---|",
            *[f"| {t} | {r.current:.1%} | {r.target:.1%} | {r.trade:+.1%} | {r.action} |"
              for t, r in rb.iterrows()],
            "",
        ]
    else:
        lines += ["## Rebalance recommendation",
                  "",
                  "No `current_holdings.csv` found, so target weights above are the full buy list. "
                  "Create that file (columns: ticker,weight) to get monthly trade recommendations.",
                  ""]

    lines += [
        "## Limitations (standing register)",
        "",
        "- Expected returns use historical means, which are noisy forecasts. Weights are guidance, not advice.",
        "- Equity ETF only universe means high positive correlation; diversification is limited in selloffs.",
        "- Monte Carlo and optimiser both overfit history; basket constraints and the position cap are the guardrails.",
        "- Free Yahoo Finance data may contain gaps or survivorship effects.",
        "- Personal learning and research tool only. Not investment advice and not FCA authorised activity.",
        "",
        "Charts saved alongside this report in `outputs/charts/`.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
