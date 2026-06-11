"""All visual output. Charts are saved as PNG files into outputs/charts/."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, charts_dir: str, name: str) -> str:
    os.makedirs(charts_dir, exist_ok=True)
    path = os.path.join(charts_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def frontier_chart(mc: dict, frontier: pd.DataFrame, opt_point: tuple,
                   minv_point: tuple, rf: float, charts_dir: str) -> str:
    """Monte Carlo cloud, efficient frontier, CML, optimal and min vol points."""
    fig, ax = plt.subplots(figsize=(11, 7))
    feas, infeas = mc["feasible"], ~mc["feasible"]
    ax.scatter(mc["vols"][infeas], mc["returns"][infeas], s=2, c="lightgrey",
               alpha=0.4, label="Simulated (violates basket limits)")
    sc = ax.scatter(mc["vols"][feas], mc["returns"][feas], s=3,
                    c=mc["sharpes"][feas], cmap="viridis", alpha=0.6,
                    label="Simulated (feasible)")
    fig.colorbar(sc, ax=ax, label="Sharpe ratio")
    if not frontier.empty:
        ax.plot(frontier["vol"], frontier["ret"], "r-", lw=2.2, label="Efficient frontier (constrained)")
    o_ret, o_vol = opt_point
    x = np.linspace(0, max(mc["vols"].max(), o_vol) * 1.05, 50)
    sharpe_opt = (o_ret - rf) / o_vol
    ax.plot(x, rf + sharpe_opt * x, "k--", lw=1.5, label="Capital Market Line")
    ax.scatter([o_vol], [o_ret], marker="*", s=420, c="red", edgecolors="black",
               zorder=5, label="Max Sharpe (optimal)")
    ax.scatter([minv_point[1]], [minv_point[0]], marker="D", s=120, c="orange",
               edgecolors="black", zorder=5, label="Min volatility")
    ax.scatter([0], [rf], marker="o", s=80, c="black", zorder=5, label=f"Risk free ({rf:.2%})")
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised expected return")
    ax.set_title("Efficient Frontier, Monte Carlo Cloud and Capital Market Line")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, charts_dir, "01_efficient_frontier.png")


def sml_chart(capm: pd.DataFrame, port_beta: float, port_ret: float,
              charts_dir: str) -> str:
    rf, mkt = capm.attrs["rf"], capm.attrs["market_return"]
    fig, ax = plt.subplots(figsize=(11, 7))
    betas = np.linspace(0, max(capm["beta"].max(), port_beta) * 1.15, 50)
    ax.plot(betas, rf + betas * (mkt - rf), "k-", lw=2, label="Security Market Line")
    ax.scatter(capm["beta"], capm["realised"], s=60, c=np.where(capm["alpha"] > 0, "green", "red"),
               edgecolors="black", zorder=4)
    for t, row in capm.iterrows():
        ax.annotate(t, (row["beta"], row["realised"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.scatter([port_beta], [port_ret], marker="*", s=420, c="blue",
               edgecolors="black", zorder=5, label="Optimal portfolio")
    ax.scatter([1.0], [mkt], marker="s", s=110, c="grey", edgecolors="black",
               zorder=5, label="Market (benchmark)")
    ax.set_xlabel("Beta vs benchmark")
    ax.set_ylabel("Annualised return (realised)")
    ax.set_title("Security Market Line: green = positive alpha vs CAPM, red = negative")
    ax.legend(fontsize=9)
    return _save(fig, charts_dir, "02_security_market_line.png")


def weights_chart(weights: pd.Series, baskets: list[dict], charts_dir: str) -> str:
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(baskets), 1)))
    tick_col = {}
    for i, b in enumerate(baskets):
        for j, flag in enumerate(b["ind"]):
            if flag and weights.index[j] not in tick_col:
                tick_col[weights.index[j]] = colours[i]
    w = weights[weights > 0.001].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(w))))
    ax.barh(w.index, w.values,
            color=[tick_col.get(t, "grey") for t in w.index], edgecolor="black", lw=0.4)
    for i, v in enumerate(w.values):
        ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[i]) for i in range(len(baskets))]
    ax.legend(handles, [b["name"] for b in baskets], fontsize=9, loc="lower right")
    ax.set_xlabel("Weight")
    ax.set_title("Optimal Weights by ETF, coloured by macro basket")
    return _save(fig, charts_dir, "03_optimal_weights.png")


def basket_chart(weights: pd.Series, baskets: list[dict], charts_dir: str) -> str:
    names = [b["name"] for b in baskets]
    allocs = [float(weights.values @ b["ind"]) for b in baskets]
    mins = [b["min"] for b in baskets]
    maxs = [b["max"] for b in baskets]
    fig, ax = plt.subplots(figsize=(10, 6))
    xpos = np.arange(len(names))
    ax.bar(xpos, allocs, color="steelblue", edgecolor="black", label="Allocation")
    for i in xpos:
        ax.plot([i - 0.4, i + 0.4], [mins[i]] * 2, "g--", lw=1.5)
        ax.plot([i - 0.4, i + 0.4], [maxs[i]] * 2, "r--", lw=1.5)
        ax.text(i, allocs[i] + 0.01, f"{allocs[i]:.1%}", ha="center", fontsize=10)
    ax.plot([], [], "g--", label="Basket minimum")
    ax.plot([], [], "r--", label="Basket maximum")
    ax.set_xticks(xpos, names, rotation=15)
    ax.set_ylabel("Portfolio weight")
    ax.set_title("Macro Basket Allocation vs Constraint Ranges")
    ax.legend(fontsize=9)
    return _save(fig, charts_dir, "04_basket_allocation.png")


def backtest_chart(bt: dict, charts_dir: str) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    port_curve = (1 + bt["daily"]).cumprod()
    bench_curve = (1 + bt["benchmark_daily"]).cumprod()
    ax1.plot(port_curve.index, port_curve.values, lw=1.8, label="Engine (monthly rebalanced, net of costs)")
    ax1.plot(bench_curve.index, bench_curve.values, lw=1.5, ls="--", label="Benchmark buy and hold")
    ax1.set_ylabel("Growth of 1")
    ax1.set_title("Walk Forward Backtest (out of sample)")
    ax1.legend(fontsize=9)
    ax2.fill_between(bt["drawdown"].index, bt["drawdown"].values, 0, color="firebrick", alpha=0.6)
    ax2.set_ylabel("Drawdown")
    return _save(fig, charts_dir, "05_backtest.png")


def correlation_chart(returns: pd.DataFrame, charts_dir: str) -> str:
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Daily Return Correlation Matrix")
    return _save(fig, charts_dir, "06_correlation_matrix.png")
