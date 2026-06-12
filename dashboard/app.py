"""Portfolio Engine dashboard — Phase 4.

Local Streamlit app. It reads the artifacts main.py writes to
outputs/dashboard/, plus config.yaml and current_holdings.csv. It never
runs the optimisation pipeline itself; the only engine code it imports
is the read only rebalance helper. Run from the project root:

    streamlit run dashboard/app.py

Personal learning and research tool only. Not investment advice.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from ruamel.yaml import YAML

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")
DASH_DIR = os.path.join(OUT_DIR, "dashboard")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")
CHARTS_DIR = os.path.join(OUT_DIR, "charts")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.yaml")
HOLDINGS_PATH = os.path.join(PROJECT_DIR, "current_holdings.csv")

sys.path.insert(0, PROJECT_DIR)
from engine import analyzer  # noqa: E402  (read only calculations, permitted by CLAUDE.md)
from engine import user_optimiser  # noqa: E402  (Phase 6, read only calculations)
from engine.report import rebalance_recommendation  # noqa: E402  (read only helper)

st.set_page_config(page_title="Portfolio Engine", page_icon="📈", layout="wide")


def is_public() -> bool:
    """Deployment mode toggle. True on Streamlit Community Cloud when
    PUBLIC_MODE is set in app secrets (or as an environment variable for
    local testing). Public mode never touches the owner's local files:
    no current_holdings.csv reads or writes, no config.yaml editor, no
    engine reruns."""
    if str(os.environ.get("PUBLIC_MODE", "")).lower() in ("1", "true", "yes"):
        return True
    try:
        return str(st.secrets.get("PUBLIC_MODE", "")).lower() in ("1", "true", "yes")
    except Exception:  # no secrets.toml configured locally
        return False


# ----------------------------------------------------------------------
# Cached file loaders (mtime in the key so saves invalidate the cache)
# ----------------------------------------------------------------------

@st.cache_data
def _read_json(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _read_csv(path: str, mtime: float, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


def load_json(path: str) -> dict | None:
    return _read_json(path, os.path.getmtime(path)) if os.path.exists(path) else None


def load_csv(path: str, **kw) -> pd.DataFrame | None:
    return _read_csv(path, os.path.getmtime(path), **kw) if os.path.exists(path) else None


def pct(x: float, dp: int = 1) -> str:
    return f"{x:.{dp}%}"


# ----------------------------------------------------------------------
# Sidebar: run status and engine runner
# ----------------------------------------------------------------------

def sidebar(results: dict | None, public: bool = False) -> None:
    st.sidebar.title("Portfolio Engine")
    if results:
        st.sidebar.caption(f"Last engine run: {results['generated']}")
        st.sidebar.caption(f"Data: {results['data_note']}")
        st.sidebar.caption(f"Risk free: {pct(results['risk_free']['rate'], 2)} "
                           f"({results['risk_free']['source']})")
    elif not public:
        st.sidebar.warning("No engine output found yet. Run the engine below.")

    if public:
        st.sidebar.divider()
        st.sidebar.caption("Public demo. The analytics are computed from the "
                           "repository's committed engine outputs; the maintainer "
                           "refreshes them periodically. Educational analytics "
                           "tool, not investment advice.")
        return

    st.sidebar.divider()
    st.sidebar.subheader("Run engine")
    offline = st.sidebar.toggle("Offline demo data", value=False,
                                help="Synthetic data, no internet needed")
    skip_bt = st.sidebar.toggle("Skip backtest (faster)", value=False)
    if st.sidebar.button("Run optimisation", type="primary", width="stretch"):
        cmd = [sys.executable, os.path.join(PROJECT_DIR, "main.py")]
        if offline:
            cmd.append("--offline")
        if skip_bt:
            cmd.append("--no-backtest")
        with st.spinner("Engine running, the backtest is the slow part..."):
            res = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True,
                                 text=True, timeout=900)
        st.session_state["run_log"] = (res.stdout + res.stderr)[-4000:]
        st.session_state["run_ok"] = res.returncode == 0
        st.cache_data.clear()
        st.rerun()

    if "run_log" in st.session_state:
        if st.session_state.get("run_ok"):
            st.sidebar.success("Engine run complete")
        else:
            st.sidebar.error("Engine run failed")
        with st.sidebar.expander("Last run output"):
            st.code(st.session_state["run_log"], language=None)

    st.sidebar.divider()
    st.sidebar.caption("Learning and research tool only. Long only, fully "
                       "invested, macro constrained. Not investment advice.")


# ----------------------------------------------------------------------
# Tab: overview
# ----------------------------------------------------------------------

def tab_overview(results: dict) -> None:
    m = results["metrics"]
    r1 = st.columns(4)
    r1[0].metric("Sharpe ratio", f"{m['sharpe']:.2f}")
    r1[1].metric("Expected return", pct(m["exp_return"]))
    r1[2].metric("Volatility", pct(m["volatility"]))
    r1[3].metric(f"Beta vs {results['benchmark']}", f"{m['beta']:.2f}")
    r2 = st.columns(4)
    r2[0].metric("Sortino ratio", f"{m['sortino']:.2f}")
    r2[1].metric("Max drawdown", pct(m["max_drawdown"]))
    r2[2].metric("Daily VaR 95%", pct(m["var_95"], 2))
    r2[3].metric("Daily CVaR 95%", pct(m["cvar_95"], 2))
    st.caption(f"Monte Carlo cross check: best of {m['mc_n_feasible']:,} feasible "
               f"simulations reached Sharpe {m['mc_best_sharpe']:.2f} vs exact "
               f"SLSQP {m['sharpe']:.2f}. The exact solver should always match "
               "or beat sampling, which is why both are run.")

    ticker_basket = {t: b["name"] for b in results["baskets"] for t in b["tickers"]}
    wdf = pd.DataFrame([
        {"ticker": t, "weight": w, "basket": ticker_basket.get(t, "none"),
         "name": results["universe"].get(t, "")}
        for t, w in sorted(results["weights"].items(), key=lambda kv: -kv[1])
    ])

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Optimal weights")
        chart = alt.Chart(wdf).mark_bar().encode(
            x=alt.X("weight:Q", axis=alt.Axis(format=".0%"), title="Weight"),
            y=alt.Y("ticker:N", sort="-x", title=None),
            color=alt.Color("basket:N", title="Macro basket"),
            tooltip=["ticker", "name",
                     alt.Tooltip("weight", format=".1%"), "basket"],
        ).properties(height=max(220, 28 * len(wdf)))
        st.altair_chart(chart, width="stretch")

    with c2:
        st.subheader("Basket allocation vs ranges")
        bdf = pd.DataFrame(results["baskets"])
        bars = alt.Chart(bdf).mark_bar(color="steelblue").encode(
            x=alt.X("name:N", title=None, axis=alt.Axis(labelAngle=-15)),
            y=alt.Y("allocation:Q", axis=alt.Axis(format=".0%"), title="Weight"),
            tooltip=["name", alt.Tooltip("allocation", format=".1%"),
                     alt.Tooltip("min", format=".0%"), alt.Tooltip("max", format=".0%")],
        )
        lo = alt.Chart(bdf).mark_tick(color="green", thickness=3, size=40).encode(
            x="name:N", y="min:Q")
        hi = alt.Chart(bdf).mark_tick(color="red", thickness=3, size=40).encode(
            x="name:N", y="max:Q")
        st.altair_chart((bars + lo + hi).properties(height=320), width="stretch")
        st.caption("Green tick = basket minimum, red tick = maximum. An allocation "
                   "pinned to a tick means that constraint is binding.")

    checks = pd.DataFrame(results["limit_checks"])
    fails = (~checks["pass"]).sum()
    st.subheader("Risk limit checks " + ("(all pass)" if fails == 0 else f"({fails} FAILED)"))
    checks["status"] = checks["pass"].map({True: "PASS", False: "FAIL"})
    st.dataframe(checks[["check", "value", "status"]], hide_index=True,
                 width="stretch")


# ----------------------------------------------------------------------
# Tab: efficient frontier
# ----------------------------------------------------------------------

def tab_frontier(results: dict) -> None:
    cloud = load_csv(os.path.join(DASH_DIR, "mc_cloud.csv"))
    frontier = load_csv(os.path.join(DASH_DIR, "frontier.csv"))
    if cloud is None:
        st.info("No frontier data yet. Run the engine from the sidebar.")
        return

    m, rf = results["metrics"], results["risk_free"]["rate"]
    vmax = float(cloud["vol"].max()) * 1.05
    cml = pd.DataFrame({"vol": [0.0, vmax],
                        "ret": [rf, rf + m["sharpe"] * vmax]})
    points = pd.DataFrame([
        {"vol": m["volatility"], "ret": m["exp_return"], "label": "Max Sharpe (optimal)"},
        {"vol": results["min_vol_point"]["vol"], "ret": results["min_vol_point"]["ret"],
         "label": "Min volatility"},
        {"vol": 0.0, "ret": rf, "label": f"Risk free ({pct(rf, 2)})"},
    ])

    tt = [alt.Tooltip("ret", format=".1%", title="Return"),
          alt.Tooltip("vol", format=".1%", title="Volatility"),
          alt.Tooltip("sharpe", format=".2f", title="Sharpe")]
    infeas = alt.Chart(cloud[~cloud["feasible"]]).mark_circle(
        size=6, color="lightgrey", opacity=0.35).encode(
        x=alt.X("vol:Q", axis=alt.Axis(format=".0%"), title="Annualised volatility"),
        y=alt.Y("ret:Q", axis=alt.Axis(format=".0%"), title="Annualised expected return"),
        tooltip=tt)
    feas = alt.Chart(cloud[cloud["feasible"]]).mark_circle(size=10, opacity=0.6).encode(
        x="vol:Q", y="ret:Q",
        color=alt.Color("sharpe:Q", scale=alt.Scale(scheme="viridis"), title="Sharpe"),
        tooltip=tt)
    layers = [infeas, feas]
    if frontier is not None and not frontier.empty:
        layers.append(alt.Chart(frontier).mark_line(color="red", strokeWidth=2.5)
                      .encode(x="vol:Q", y="ret:Q"))
    layers.append(alt.Chart(cml).mark_line(color="black", strokeDash=[6, 4])
                  .encode(x="vol:Q", y="ret:Q"))
    layers.append(alt.Chart(points).mark_point(size=260, filled=True,
                                               stroke="black", strokeWidth=1).encode(
        x="vol:Q", y="ret:Q",
        shape=alt.Shape("label:N", title=None),
        color=alt.value("red"),
        tooltip=["label", alt.Tooltip("ret", format=".1%"), alt.Tooltip("vol", format=".1%")]))

    st.altair_chart(alt.layer(*layers).resolve_scale(shape="independent")
                    .properties(height=560).interactive(), width="stretch")
    st.caption("Grey points violate at least one basket range; coloured points are "
               "feasible. The red line is the constrained efficient frontier and the "
               "dashed line is the Capital Market Line through the optimal portfolio. "
               "Scroll to zoom, drag to pan, hover for exact figures.")


# ----------------------------------------------------------------------
# Tab: CAPM / Security Market Line
# ----------------------------------------------------------------------

def tab_capm(results: dict) -> None:
    capm = load_csv(os.path.join(DASH_DIR, "capm.csv"), index_col=0)
    if capm is None:
        st.info("No CAPM data yet. Run the engine from the sidebar.")
        return
    rf = results["capm_attrs"]["rf"]
    mkt = results["capm_attrs"]["market_return"]
    m = results["metrics"]

    df = capm.reset_index()
    df["name"] = df["ticker"].map(results["universe"]).fillna("")
    df["alpha_sign"] = (df["alpha"] > 0).map({True: "positive alpha", False: "negative alpha"})
    bmax = max(float(df["beta"].max()), m["beta"]) * 1.15
    sml = pd.DataFrame({"beta": [0.0, bmax], "ret": [rf, rf + bmax * (mkt - rf)]})

    line = alt.Chart(sml).mark_line(color="black").encode(
        x=alt.X("beta:Q", title=f"Beta vs {results['benchmark']}"),
        y=alt.Y("ret:Q", axis=alt.Axis(format=".0%"), title="Annualised return"))
    pts = alt.Chart(df).mark_circle(size=110, stroke="black", strokeWidth=0.5).encode(
        x="beta:Q", y=alt.Y("realised:Q"),
        color=alt.Color("alpha_sign:N", title=None,
                        scale=alt.Scale(domain=["positive alpha", "negative alpha"],
                                        range=["green", "red"])),
        tooltip=["ticker", "name", alt.Tooltip("beta", format=".2f"),
                 alt.Tooltip("realised", format=".1%", title="Realised return"),
                 alt.Tooltip("capm_expected", format=".1%", title="CAPM expected"),
                 alt.Tooltip("alpha", format=".1%")])
    labels = alt.Chart(df).mark_text(dx=8, dy=-6, fontSize=10).encode(
        x="beta:Q", y="realised:Q", text="ticker")
    special = pd.DataFrame([
        {"beta": m["beta"], "ret": m["exp_return"], "label": "Optimal portfolio"},
        {"beta": 1.0, "ret": mkt, "label": f"Market ({results['benchmark']})"}])
    stars = alt.Chart(special).mark_point(size=300, filled=True, stroke="black").encode(
        x="beta:Q", y="ret:Q", shape=alt.Shape("label:N", title=None),
        color=alt.value("blue"),
        tooltip=["label", alt.Tooltip("beta", format=".2f"), alt.Tooltip("ret", format=".1%")])

    st.altair_chart((line + pts + labels + stars).resolve_scale(shape="independent")
                    .properties(height=520).interactive(), width="stretch")
    st.caption("Points above the SML earned more than CAPM predicts for their beta "
               "(positive alpha from OLS on daily returns); points below underperformed. "
               "Alpha here is historical, not a forecast.")

    st.subheader("Per ETF CAPM table")
    show = df[["ticker", "name", "beta", "capm_expected", "realised", "alpha"]].copy()
    st.dataframe(show.style.format({"beta": "{:.2f}", "capm_expected": "{:.1%}",
                                    "realised": "{:.1%}", "alpha": "{:+.1%}"}),
                 hide_index=True, width="stretch")


# ----------------------------------------------------------------------
# Tab: risk
# ----------------------------------------------------------------------

def tab_risk(results: dict) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Beta stress tests")
        sdf = pd.DataFrame(results["stress"])
        st.dataframe(sdf.style.format({"market_shock": "{:.0%}",
                                       "est_portfolio_impact": "{:.1%}"}),
                     hide_index=True, width="stretch")
        w = results["worst"]
        st.caption(f"Historical worst day {pct(float(w['worst_day']), 2)} "
                   f"({w['worst_day_date']}); worst month "
                   f"{pct(float(w['worst_month']), 2)} ({w['worst_month_date']}). "
                   "Stress figures are first order CAPM approximations: "
                   "portfolio loss is roughly beta times the market shock.")
    with c2:
        st.subheader("Correlation monitor")
        cm = results["correlation_monitor"]
        a, b = st.columns(2)
        a.metric("Avg pairwise corr (full sample)", f"{cm['avg_corr_full']:.2f}")
        b.metric(f"Recent {cm['window_days']} days", f"{cm['avg_corr_recent']:.2f}")
        if cm["warning"]:
            st.error("Correlations elevated vs history. Diversification is weakening "
                     "exactly when it matters, the classic equity only failure mode.")
        else:
            st.success("Recent correlations in line with history.")

    corr = load_csv(os.path.join(DASH_DIR, "correlation.csv"), index_col=0)
    if corr is not None:
        st.subheader("Daily return correlation matrix")
        cdf = corr.stack().reset_index()  # name agnostic, live data has a named index
        cdf.columns = ["t1", "t2", "corr"]
        heat = alt.Chart(cdf).mark_rect().encode(
            x=alt.X("t1:N", title=None, sort=list(corr.columns)),
            y=alt.Y("t2:N", title=None, sort=list(corr.columns)),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redyellowgreen",
                                                      reverse=True, domain=[-1, 1])),
            tooltip=["t1", "t2", alt.Tooltip("corr", format=".2f")],
        ).properties(height=520)
        st.altair_chart(heat, width="stretch")


# ----------------------------------------------------------------------
# Tab: backtest
# ----------------------------------------------------------------------

def tab_backtest(results: dict) -> None:
    daily = load_csv(os.path.join(DASH_DIR, "backtest_daily.csv"),
                     index_col=0, parse_dates=True)
    if daily is None or results.get("backtest_summary") is None:
        st.info("No backtest data. Run the engine without 'Skip backtest'.")
        return
    s = results["backtest_summary"]
    p, b = s["portfolio"], s["benchmark"]

    st.caption(f"Walk forward, out of sample: {s['start']} to {s['end']}, "
               f"{s['n_rebalances']} monthly rebalances, "
               f"{s['avg_turnover_cost_bps']} bps per unit turnover. Each month the "
               "engine re-optimises on trailing data only, so no look ahead.")
    cols = st.columns(5)
    cols[0].metric("CAGR", pct(p["cagr"]), delta=pct(p["cagr"] - b["cagr"]))
    cols[1].metric("Volatility", pct(p["vol"]),
                   delta=pct(p["vol"] - b["vol"]), delta_color="inverse")
    cols[2].metric("Sharpe", f"{p['sharpe']:.2f}", delta=f"{p['sharpe'] - b['sharpe']:.2f}")
    cols[3].metric("Sortino", f"{p['sortino']:.2f}")
    cols[4].metric("Max drawdown", pct(p["max_drawdown"]),
                   delta=pct(p["max_drawdown"] - b["max_drawdown"]))
    st.caption(f"Deltas are vs {results['benchmark']} buy and hold over the same period.")

    curves = pd.DataFrame({
        "Engine (net of costs)": (1 + daily["portfolio"]).cumprod(),
        f"{results['benchmark']} buy and hold": (1 + daily["benchmark"]).cumprod(),
    }).reset_index().melt(id_vars="date", var_name="series", value_name="growth")
    line = alt.Chart(curves).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("growth:Q", title="Growth of 1", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None),
        tooltip=[alt.Tooltip("date:T"), "series", alt.Tooltip("growth", format=".3f")],
    ).properties(height=380).interactive()
    st.altair_chart(line, width="stretch")

    dd = daily.reset_index()[["date", "drawdown"]]
    area = alt.Chart(dd).mark_area(color="firebrick", opacity=0.6).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("drawdown:Q", axis=alt.Axis(format=".0%"), title="Drawdown"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("drawdown", format=".1%")],
    ).properties(height=160)
    st.altair_chart(area, width="stretch")

    wh = load_csv(os.path.join(OUT_DIR, "backtest_weights_history.csv"),
                  index_col=0, parse_dates=True)
    if wh is not None:
        st.subheader("Weights through time (monthly re-optimisation)")
        whm = wh.reset_index().melt(id_vars=wh.index.name or "index",
                                    var_name="ticker", value_name="weight")
        whm.columns = ["date", "ticker", "weight"]
        stack = alt.Chart(whm).mark_area().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("weight:Q", stack="normalize", axis=alt.Axis(format=".0%"), title="Weight"),
            color=alt.Color("ticker:N", title=None),
            tooltip=["ticker", alt.Tooltip("date:T"), alt.Tooltip("weight", format=".1%")],
        ).properties(height=340)
        st.altair_chart(stack, width="stretch")


# ----------------------------------------------------------------------
# Tab: robustness and Black Litterman (Phase 2)
# ----------------------------------------------------------------------

def tab_robustness(results: dict) -> None:
    est = results.get("estimation", {})
    c = st.columns(4)
    c[0].metric("Expected returns", est.get("expected_returns", "historical"))
    c[1].metric("Covariance", est.get("covariance_estimator", "simple"))
    c[2].metric("Active BL views", str(est.get("bl_n_views", "n/a")))
    c[3].metric("Implied risk aversion",
                f"{est['bl_delta']:.2f}" if est.get("bl_delta") else "n/a")
    if est.get("bl_n_views") == 0:
        msg = ("views.yaml has no active views, so the BL posterior below equals "
               "the market implied equilibrium exactly.")
        if not is_public():
            msg += (" Add your views (with confidences) to views.yaml when you "
                    "have macro judgement to encode.")
        st.info(msg)

    blc = load_csv(os.path.join(DASH_DIR, "bl_comparison.csv"), index_col=0)
    if blc is not None:
        st.subheader("Historical vs Black Litterman expected returns")
        melt = blc.reset_index().melt(
            id_vars=["ticker", "market_weight"],
            value_vars=["historical", "equilibrium", "bl_posterior"],
            var_name="estimate", value_name="ret")
        bar = alt.Chart(melt).mark_bar().encode(
            x=alt.X("ticker:N", title=None, sort=list(blc.index)),
            xOffset=alt.XOffset("estimate:N"),
            y=alt.Y("ret:Q", axis=alt.Axis(format=".0%"), title="Annualised return"),
            color=alt.Color("estimate:N", title=None),
            tooltip=["ticker", "estimate", alt.Tooltip("ret", format=".1%"),
                     alt.Tooltip("market_weight", format=".1%")],
        ).properties(height=380)
        st.altair_chart(bar, width="stretch")
        st.caption(f"Market weights source: {est.get('bl_weights_source', 'n/a')}. "
                   "Equilibrium returns are reverse optimised (pi = delta Sigma w), "
                   "so high beta assets get high implied returns by construction. "
                   "Where the historical bar towers over the equilibrium bar, the "
                   "sample mean is making a claim the market's pricing does not "
                   "support — that is the estimation noise BL is designed to damp.")

    stab = load_csv(os.path.join(DASH_DIR, "weight_stability.csv"), index_col=0)
    if stab is not None:
        st.subheader("Bootstrap weight stability")
        shown = stab[(stab["optimal"] > 0.001) | (stab["boot_mean"] > 0.01)].reset_index()
        shown["lo"] = (shown["boot_mean"] - shown["boot_std"]).clip(lower=0)
        shown["hi"] = shown["boot_mean"] + shown["boot_std"]
        base = alt.Chart(shown).encode(
            x=alt.X("ticker:N", title=None, sort=list(shown["ticker"])))
        bars = base.mark_bar(color="steelblue", opacity=0.55).encode(
            y=alt.Y("optimal:Q", axis=alt.Axis(format=".0%"), title="Weight"),
            tooltip=["ticker", alt.Tooltip("optimal", format=".1%"),
                     alt.Tooltip("boot_mean", format=".1%"),
                     alt.Tooltip("boot_std", format=".1%")])
        err = base.mark_errorbar(color="black").encode(
            y=alt.Y("lo:Q", title=""), y2="hi:Q")
        pts = base.mark_point(color="black", filled=True, size=60).encode(y="boot_mean:Q")
        st.altair_chart((bars + err + pts).properties(height=340), width="stretch")
        st.caption(f"Bars = optimal weights. Dots = mean across "
                   f"{int(stab['n_samples'].iloc[0])} bootstrap re-optimisations, "
                   "whiskers = one std. Dots far from bars, or wide whiskers, mark "
                   "positions the optimiser holds on sampling luck (Michaud "
                   "resampled efficiency).")

    sens = load_csv(os.path.join(DASH_DIR, "sensitivity.csv"), index_col=0)
    if sens is not None:
        st.subheader("Expected return sensitivity")
        n_frag = int(sens["fragile"].sum())
        if n_frag:
            st.warning(f"{n_frag} position(s) flagged fragile: a 1% change in one "
                       "return estimate moves over 10% of the portfolio.")
        else:
            st.success("No fragile positions: weights survive a 1% shift in any "
                       "single return estimate.")
        show = sens.reset_index()[["ticker", "weight", "turnover_up",
                                   "turnover_down", "own_change_up", "fragile"]]
        st.dataframe(show.style.format({
            "weight": "{:.1%}", "turnover_up": "{:.1%}", "turnover_down": "{:.1%}",
            "own_change_up": "{:+.1%}"}).map(
                lambda v: "color: red; font-weight: bold" if v is True else "",
                subset=["fragile"]),
            hide_index=True, width="stretch")

    if blc is None and stab is None and sens is None:
        st.info("No Phase 2 outputs yet. Re-run the engine from the sidebar.")


# ----------------------------------------------------------------------
# Tab: basket editor (rewrites config.yaml, comments preserved)
# ----------------------------------------------------------------------

def tab_baskets(results: dict | None) -> None:
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    yaml_rt.width = 4096  # don't wrap long thesis lines on save
    with open(CONFIG_PATH, encoding="utf-8") as f:
        doc = yaml_rt.load(f)

    st.caption("Adjust the macro basket ranges and the single ETF cap, then save. "
               "config.yaml is rewritten in place with comments preserved; the engine "
               "stays untouched. Re-run the optimisation afterwards to see the effect. "
               "These ranges are your active views, the Markowitz machinery only "
               "optimises within them.")

    alloc_now = ({b["name"]: b["allocation"] for b in results["baskets"]}
                 if results else {})

    max_pos = st.slider("Single ETF cap (max_position)", 5, 50,
                        int(round(doc["settings"]["max_position"] * 100)),
                        step=1, format="%d%%") / 100

    new_ranges = {}
    for name, b in doc["baskets"].items():
        st.markdown(f"**{name}** | {b['thesis']}")
        tick_list = ", ".join(b["tickers"])
        cur = f" | current allocation {pct(alloc_now[name])}" if name in alloc_now else ""
        st.caption(f"{tick_list}{cur}")
        lo, hi = st.slider(f"range_{name}", 0, 100,
                           (int(round(float(b["min"]) * 100)),
                            int(round(float(b["max"]) * 100))),
                           step=1, format="%d%%", label_visibility="collapsed")
        new_ranges[name] = (lo / 100, hi / 100)

    # Feasibility checks before allowing a save
    sum_min = sum(lo for lo, _ in new_ranges.values())
    sum_max = sum(hi for _, hi in new_ranges.values())
    problems = []
    if sum_min > 1.0 + 1e-9:
        problems.append(f"Basket minimums sum to {pct(sum_min)} which exceeds 100%. "
                        "No fully invested portfolio can satisfy that.")
    if sum_max < 1.0 - 1e-9:
        problems.append(f"Basket maximums sum to {pct(sum_max)}. The baskets cover the "
                        "whole universe, so a fully invested portfolio needs them to "
                        "sum to at least 100%.")
    for name, (lo, _) in new_ranges.items():
        cap = len(doc["baskets"][name]["tickers"]) * max_pos
        if lo > cap + 1e-9:
            problems.append(f"'{name}' minimum {pct(lo)} is unreachable: its "
                            f"{len(doc['baskets'][name]['tickers'])} tickers at the "
                            f"{pct(max_pos)} cap only allow {pct(cap)}.")
    for p in problems:
        st.error(p)

    if st.button("Save to config.yaml", type="primary", disabled=bool(problems)):
        doc["settings"]["max_position"] = round(max_pos, 4)
        for name, (lo, hi) in new_ranges.items():
            doc["baskets"][name]["min"] = round(lo, 4)
            doc["baskets"][name]["max"] = round(hi, 4)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml_rt.dump(doc, f)
        st.success("config.yaml updated. Re-run the engine (sidebar) to re-optimise "
                   "under the new constraints.")


# ----------------------------------------------------------------------
# Tab: holdings and rebalance
# ----------------------------------------------------------------------

def tab_holdings() -> None:
    st.caption("Holdings are entered manually (Trading 212 has no public API). "
               "Edit the table and save; the trade list below compares your holdings "
               "with the latest optimal weights.")
    if os.path.exists(HOLDINGS_PATH):
        cur = pd.read_csv(HOLDINGS_PATH)
    else:
        cur = pd.DataFrame({"ticker": pd.Series(dtype=str),
                            "weight": pd.Series(dtype=float)})
    edited = st.data_editor(
        cur, num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "ticker": st.column_config.TextColumn("ticker", required=True),
            "weight": st.column_config.NumberColumn(
                "weight", min_value=0.0, max_value=1.0, step=0.001, format="%.3f"),
        })
    total = float(edited["weight"].fillna(0).sum()) if len(edited) else 0.0
    st.caption(f"Weights sum to {pct(total)}." +
               ("" if abs(total - 1) < 0.01 or total == 0 else
                " That is away from 100%, check for missing positions or cash."))
    if st.button("Save current_holdings.csv"):
        clean = edited.dropna(subset=["ticker"])
        clean = clean[clean["ticker"].str.strip() != ""]
        clean.to_csv(HOLDINGS_PATH, index=False)
        st.success(f"Saved {len(clean)} holdings.")

    st.subheader("Rebalance recommendation")
    opt_path = os.path.join(OUT_DIR, "optimal_weights.csv")
    if not os.path.exists(opt_path):
        st.info("No optimal weights yet. Run the engine from the sidebar.")
        return
    if not os.path.exists(HOLDINGS_PATH):
        st.info("Save your holdings above to get a trade list. Until then the "
                "optimal weights are the full buy list.")
        return
    target = pd.read_csv(opt_path, index_col=0)["weight"]
    rb = rebalance_recommendation(target, PROJECT_DIR)

    def colour(a: str) -> str:
        return {"BUY": "color: green; font-weight: bold",
                "SELL": "color: red; font-weight: bold"}.get(a, "color: grey")
    st.dataframe(rb.style.format({"current": "{:.1%}", "target": "{:.1%}",
                                  "trade": "{:+.1%}"}).map(colour, subset=["action"]),
                 width="stretch")
    st.caption("Trades smaller than 0.5% either way show as HOLD to avoid churning "
               "costs. Guidance only, not investment advice.")


# ----------------------------------------------------------------------
# Tab: portfolio analyzer (Phase 5)
# ----------------------------------------------------------------------

def _fmt_stat(key: str, v) -> str:
    if v is None:
        return "n/a"
    pct_keys = {"ann_return", "volatility", "max_drawdown", "var_95",
                "cvar_95", "alpha"}
    if key in pct_keys:
        return f"{v:.2%}"
    if key == "n_holdings":
        return f"{int(v)}"
    return f"{v:.2f}"


def tab_analyzer(results: dict | None, public: bool = False) -> None:
    st.caption("Paste any portfolio (any Yahoo tickers, any quote currency) and "
               "compare it with the engine optimal, the constrained frontier and "
               "the benchmark. Everything is converted to USD and every conversion "
               "is documented. Historical analytics only, not advice.")

    if public:
        # session state only: the public app never reads or stores anyone's
        # holdings, the example below is generic
        seed = pd.DataFrame({"ticker": ["VOO", "QQQ"], "weight": [0.6, 0.4]})
        st.caption("Example portfolio shown. Edit it freely; entries live in "
                   "your browser session only and are never stored.")
    elif os.path.exists(HOLDINGS_PATH):
        seed = pd.read_csv(HOLDINGS_PATH)
    else:
        seed = pd.DataFrame({"ticker": pd.Series(dtype=str),
                             "weight": pd.Series(dtype=float)})
    edited = st.data_editor(
        seed, num_rows="dynamic", hide_index=True, width="stretch",
        key="analyzer_editor",
        column_config={
            "ticker": st.column_config.TextColumn("ticker", required=True,
                                                  help="Any Yahoo Finance symbol"),
            "weight": st.column_config.NumberColumn("weight", min_value=0.0,
                                                    step=0.001, format="%.3f"),
        })

    clean = edited.dropna(subset=["ticker"])
    clean = clean[clean["ticker"].str.strip() != ""]
    total = float(clean["weight"].fillna(0).sum()) if len(clean) else 0.0
    if len(clean) and abs(total - 1.0) > 0.005:
        st.warning(f"Weights sum to {pct(total)}; they will be normalised to 100% "
                   "for the analysis and the saved file.")

    if st.button("Validate tickers and analyse", type="primary"):
        if not len(clean):
            st.error("Add at least one holding.")
            return
        holdings = clean.set_index("ticker")["weight"].astype(float)
        holdings = holdings / holdings.sum()
        with st.spinner("Validating tickers on Yahoo..."):
            checks = analyzer.validate_tickers(list(holdings.index))
        bad = {t: c for t, c in checks.items() if not c["ok"]}
        if bad:
            for t, c in bad.items():
                st.error(f"'{t}' did not resolve on Yahoo Finance"
                         + (f": {c.get('error')}" if c.get("error") else "."))
            return
        for t, c in checks.items():
            st.caption(f"{t}: {c['name']} ({c['currency']})")
        if public:
            st.success("Holdings validated (this session only, nothing stored).")
        else:
            holdings.rename("weight").reset_index().to_csv(HOLDINGS_PATH, index=False)
            st.success("Holdings validated and saved to current_holdings.csv.")
        with st.spinner("Fetching prices, converting to USD, computing..."):
            try:
                st.session_state["analysis"] = analyzer.analyze(
                    holdings, PROJECT_DIR, use_cache=not public)
                st.session_state.pop("user_opt", None)  # stale for new holdings
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return

    res = st.session_state.get("analysis")
    if not res:
        st.info("Edit the holdings above and run the analysis.")
        return

    st.subheader("Currency conversions applied")
    for t, note in res["conversion_notes"].items():
        st.caption(f"{t}: {note}")

    st.subheader("Observations")
    for f in res["flags"]:
        st.markdown(f"- {f}")
    d = res["diversification"]
    st.markdown(f"- Diversification captured: weighted average constituent vol "
                f"{pct(d['weighted_avg_constituent_vol'])} vs portfolio vol "
                f"{pct(d['portfolio_vol'])}, benefit {pct(d['benefit_captured'])}.")

    st.subheader("User portfolio vs engine optimal vs benchmark")
    st.caption(f"USD terms, {res['user']['start']} to {res['user']['end']}, "
               f"risk free {pct(res['rf'], 2)}. Beta and alpha estimated on "
               f"{res.get('beta_method', 'weekly returns')}.")
    comp = analyzer.comparison_table(res)
    disp = comp.copy()
    for col in disp.columns:
        disp[col] = [_fmt_stat(k, v) for k, v in comp[col].items()]
    st.dataframe(disp, width="stretch")

    st.subheader("Position vs the constrained efficient frontier")
    g = res["frontier_gap"]
    c1, c2 = st.columns(2)
    c1.metric("Return given up at the same volatility", pct(g["return_gap"]),
              help="Frontier return at your volatility minus your return")
    c2.metric("Excess volatility at the same return", pct(g["vol_gap"]),
              help="Your volatility minus the frontier volatility at your return")
    for n in g["notes"]:
        st.caption(f"Note: {n}")
    pts = pd.DataFrame([
        {"vol": res["user"]["volatility"], "ret": res["user"]["ann_return"],
         "label": "User portfolio"},
        {"vol": res["optimal"]["volatility"], "ret": res["optimal"]["ann_return"],
         "label": "Engine optimal"},
        {"vol": res["spy"]["volatility"], "ret": res["spy"]["ann_return"],
         "label": res["benchmark"]},
    ])
    fline = alt.Chart(res["frontier"]).mark_line(color="red", strokeWidth=2).encode(
        x=alt.X("vol:Q", axis=alt.Axis(format=".0%"), title="Annualised volatility"),
        y=alt.Y("ret:Q", axis=alt.Axis(format=".0%"), title="Annualised return"))
    fpts = alt.Chart(pts).mark_point(size=300, filled=True, stroke="black").encode(
        x="vol:Q", y="ret:Q", shape=alt.Shape("label:N", title=None),
        color=alt.Color("label:N", title=None),
        tooltip=["label", alt.Tooltip("ret", format=".1%"),
                 alt.Tooltip("vol", format=".1%")])
    st.altair_chart((fline + fpts).resolve_scale(shape="independent")
                    .properties(height=420).interactive(), width="stretch")
    st.caption("The frontier and engine optimal come from the engine's latest run "
               "(its own estimation window); the user portfolio and benchmark "
               "points use USD converted Yahoo data over the analysis window. "
               "Small window differences are noted in the table dates.")

    st.subheader("Historical effect of partial reallocation toward the engine optimal")
    b = res["blends"].reset_index()
    b["reallocation"] = b["reallocation"].map(lambda f: f"{f:.0%}")
    st.dataframe(b.style.format({"ann_return": "{:.2%}", "volatility": "{:.2%}",
                                 "sharpe": "{:.2f}", "max_drawdown": "{:.1%}"}),
                 hide_index=True, width="stretch")
    st.caption("Each row re-runs history with that fraction moved from the user "
               "portfolio to the engine optimal. Backward looking arithmetic, "
               "not a forecast and not a recommendation.")

    _user_opt_section(res, public)
    st.caption(res["disclaimer"])


# ----------------------------------------------------------------------
# Phase 6: optimise the user's own universe (inside the Analyzer tab)
# ----------------------------------------------------------------------

ILLUSTRATION_CAPTION = (
    "Optimised weights are the historical in sample solution under the stated "
    "assumptions. Estimation error means these are illustrations, not "
    "forecasts: small changes in the inputs move the weights, which is exactly "
    "what the sensitivity table below and the Robustness & BL tab demonstrate.")


def _user_opt_section(res: dict, public: bool) -> None:
    st.divider()
    st.subheader("Optimise this universe")
    if res.get("prices_usd") is None:
        st.info("Re-run the analysis above to enable optimisation.")
        return
    holdings = res["holdings"]
    tickers = list(holdings.index)
    n = len(tickers)
    st.caption("Four alternative weightings over exactly the tickers analysed "
               "above, long only and fully invested, solved with the same SLSQP "
               "machinery as the engine. " + ILLUSTRATION_CAPTION)
    if n < 2:
        st.info("Optimisation needs at least 2 valid tickers; add holdings "
                "above and re-run the analysis.")
        return

    min_cap = int(np.ceil(100.0 / n))
    cap = st.slider("Max single position", min_cap, 100, max(25, min_cap),
                    step=1, format="%d%%",
                    help="Long only and fully invested are always enforced") / 100
    if st.button("Run optimisation", type="primary", key="run_user_opt"):
        prices = res["prices_usd"][tickers]
        try:
            with st.spinner("Optimising and simulating 10,000 portfolios..."):
                opt = user_optimiser.optimise(prices, res["rf"], max_pos=cap)
                sens = user_optimiser.sensitivity(opt)
            with st.spinner("Walk forward backtest, roughly 30 to 60 seconds..."):
                bt = user_optimiser.backtest_user(opt, holdings, res["bench_daily"])
            st.session_state["user_opt"] = {"opt": opt, "sens": sens, "bt": bt}
        except ValueError as exc:
            st.error(str(exc))
            return

    uo_state = st.session_state.get("user_opt")
    if not uo_state:
        return
    opt, sens, bt = uo_state["opt"], uo_state["sens"], uo_state["bt"]
    if opt["cap_note"]:
        st.warning(opt["cap_note"])

    labels = {"max_sharpe": "Max Sharpe", "min_vol": "Min volatility",
              "max_div": "Max diversification", "equal_weight": "Equal weight"}
    prices = res["prices_usd"][tickers]
    rets = opt["rets"]

    # ----- side by side stats -------------------------------------------
    def stats_col(w: pd.Series) -> dict:
        s = analyzer.portfolio_stats(prices, w, res["bench_daily"], res["rf"])
        wn = (w / w.sum()).reindex(rets.columns).fillna(0.0).values
        s["avg_pairwise_corr"] = user_optimiser.weighted_avg_correlation(wn, opt["corr"])
        return {k: s[k] for k in ("ann_return", "volatility", "sharpe", "sortino",
                                  "max_drawdown", "beta", "avg_pairwise_corr", "hhi")}

    cols = {"Current": stats_col(holdings)}
    for key, lab in labels.items():
        cols[lab] = stats_col(opt["weights"][key])
    comp = pd.DataFrame(cols)
    disp = comp.copy()
    for c in disp.columns:
        disp[c] = [_fmt_stat(k, v) for k, v in comp[c].items()]
    st.markdown("**Current weights vs optimised alternatives** (USD, historical)")
    st.dataframe(disp, width="stretch")
    st.caption("Average pairwise correlation here is weight weighted, so it "
               "differs per portfolio over the same tickers. HHI 1.00 = single "
               "holding; 1/n = equal weight.")

    # ----- frontier with cloud and the five portfolios --------------------
    st.markdown("**This universe's own efficient frontier**")
    mc = opt["mc"]
    rng = np.random.default_rng(3)
    keep = rng.choice(len(mc["returns"]), size=min(4000, len(mc["returns"])),
                      replace=False)
    cloud = pd.DataFrame({"vol": mc["vols"][keep], "ret": mc["returns"][keep],
                          "sharpe": mc["sharpes"][keep]})
    mu, cov = opt["exp_ret"].values, opt["cov"].values

    def point(w: pd.Series, label: str) -> dict:
        wn = (w / w.sum()).reindex(rets.columns).fillna(0.0).values
        return {"vol": float(np.sqrt(wn @ cov @ wn)), "ret": float(wn @ mu),
                "label": label}

    pts = pd.DataFrame([point(holdings, "Current")] +
                       [point(opt["weights"][k], lab) for k, lab in labels.items()])
    layers = [
        alt.Chart(cloud).mark_circle(size=8, opacity=0.45).encode(
            x=alt.X("vol:Q", axis=alt.Axis(format=".0%"), title="Annualised volatility"),
            y=alt.Y("ret:Q", axis=alt.Axis(format=".0%"), title="Annualised return"),
            color=alt.Color("sharpe:Q", scale=alt.Scale(scheme="viridis"), title="Sharpe"),
            tooltip=[alt.Tooltip("ret", format=".1%"), alt.Tooltip("vol", format=".1%"),
                     alt.Tooltip("sharpe", format=".2f")]),
    ]
    if not opt["frontier"].empty:
        layers.append(alt.Chart(opt["frontier"]).mark_line(color="red", strokeWidth=2)
                      .encode(x="vol:Q", y="ret:Q"))
    layers.append(alt.Chart(pts).mark_point(size=300, filled=True, stroke="black",
                                            strokeWidth=1).encode(
        x="vol:Q", y="ret:Q", shape=alt.Shape("label:N", title=None),
        color=alt.value("red"),
        tooltip=["label", alt.Tooltip("ret", format=".1%"),
                 alt.Tooltip("vol", format=".1%")]))
    st.altair_chart(alt.layer(*layers).resolve_scale(shape="independent")
                    .properties(height=440).interactive(), width="stretch")
    st.caption("The SLSQP portfolios are the exact constrained solutions; the "
               "10,000 simulation cloud is an illustration of the same "
               "opportunity set, identical in spirit to the engine's own "
               "frontier chart.")

    # ----- weight changes --------------------------------------------------
    st.markdown("**Weight changes vs current**")
    pick = st.selectbox("Alternative", list(labels.values()), key="uo_alt")
    key = [k for k, v in labels.items() if v == pick][0]
    cur = (holdings / holdings.sum()).reindex(rets.columns).fillna(0.0)
    prop = opt["weights"][key]
    delta = prop - cur
    wc = pd.DataFrame({
        "current": cur, "proposed": prop, "change": delta,
        "direction": np.where(delta > 0.005, "increase",
                              np.where(delta < -0.005, "decrease", "unchanged")),
    }).sort_values("change", ascending=False)
    st.dataframe(wc.style.format({"current": "{:.1%}", "proposed": "{:.1%}",
                                  "change": "{:+.1%}"}),
                 width="stretch")
    st.caption("Weight deltas are descriptive analytics of the historical "
               "solution, not trade instructions.")

    # ----- correlation heatmap --------------------------------------------
    if n > 1:
        st.markdown("**Correlation between your holdings**")
        t1, t2, cmax = user_optimiser.highest_corr_pair(opt["corr"])
        cdf = opt["corr"].stack().reset_index()
        cdf.columns = ["a", "b", "corr"]
        heat = alt.Chart(cdf).mark_rect().encode(
            x=alt.X("a:N", title=None, sort=tickers),
            y=alt.Y("b:N", title=None, sort=tickers),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redyellowgreen",
                                                      reverse=True, domain=[-1, 1])),
            tooltip=["a", "b", alt.Tooltip("corr", format=".2f")])
        outline = alt.Chart(cdf[((cdf["a"] == t1) & (cdf["b"] == t2)) |
                                ((cdf["a"] == t2) & (cdf["b"] == t1))]).mark_rect(
            fillOpacity=0, stroke="black", strokeWidth=3).encode(
            x=alt.X("a:N", sort=tickers), y=alt.Y("b:N", sort=tickers))
        st.altair_chart((heat + outline).properties(height=max(240, 36 * n)),
                        width="stretch")
        st.caption(f"Highest correlated pair (outlined): {t1} and {t2} at "
                   f"{cmax:.2f}. Highly correlated holdings add little "
                   "diversification to each other.")

    # ----- sensitivity carried over -----------------------------------------
    st.markdown("**Stability of the max Sharpe solution**")
    n_frag = int(sens["fragile"].sum())
    if n_frag:
        st.warning(f"{n_frag} position(s) are fragile: shifting one ticker's "
                   "expected return by 1% moves over 10% of the portfolio. "
                   "The shiny optimum is less solid than it looks.")
    else:
        st.success("The max Sharpe weights survive a 1% shift in any single "
                   "return estimate without moving more than 10% of the book.")
    sshow = sens.reset_index()[["ticker", "weight", "turnover_up",
                                "turnover_down", "fragile"]]
    st.dataframe(sshow.style.format({"weight": "{:.1%}", "turnover_up": "{:.1%}",
                                     "turnover_down": "{:.1%}"}).map(
        lambda v: "color: red; font-weight: bold" if v is True else "",
        subset=["fragile"]), hide_index=True, width="stretch")

    # ----- walk forward backtest ---------------------------------------------
    st.markdown("**Walk forward backtest of this universe**")
    if "skipped" in bt:
        st.info(bt["skipped"])
    else:
        st.caption(f"{bt['start']} to {bt['end']}, {bt['n_rebalances']} monthly "
                   "re-optimisations on trailing data only, 10 bps costs on "
                   "turnover. Out of sample results are the honest measure, and "
                   "they are usually worse than the in sample optimum above. "
                   "That gap is the point of showing them.")
        growth = (1 + bt["daily"].fillna(0)).cumprod().reset_index()
        growth.columns = ["date"] + list(bt["daily"].columns)
        gm = growth.melt(id_vars="date", var_name="series", value_name="growth")
        line = alt.Chart(gm).mark_line().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("growth:Q", title="Growth of 1", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None),
            tooltip=[alt.Tooltip("date:T"), "series",
                     alt.Tooltip("growth", format=".3f")],
        ).properties(height=360).interactive()
        st.altair_chart(line, width="stretch")
        st.dataframe(bt["summary"].style.format({
            "cagr": "{:.2%}", "vol": "{:.2%}", "sharpe": "{:.2f}",
            "max_drawdown": "{:.1%}"}), width="stretch")


# ----------------------------------------------------------------------
# Tab: tear sheets
# ----------------------------------------------------------------------

def tab_reports() -> None:
    reports = sorted((f for f in os.listdir(REPORTS_DIR) if f.endswith(".md")),
                     reverse=True) if os.path.isdir(REPORTS_DIR) else []
    if not reports:
        if is_public():
            st.info("Written tear sheets are not published in the public demo. "
                    "The chart gallery below shows the latest engine run.")
        else:
            st.info("No tear sheets yet. Run the engine from the sidebar.")
    else:
        pick = st.selectbox("Tear sheet", reports)
        with open(os.path.join(REPORTS_DIR, pick), encoding="utf-8") as f:
            st.markdown(f.read())
    with st.expander("Static chart gallery (PNGs from the last run)"):
        pngs = sorted(f for f in os.listdir(CHARTS_DIR) if f.endswith(".png")) \
            if os.path.isdir(CHARTS_DIR) else []
        for png in pngs:
            st.image(os.path.join(CHARTS_DIR, png))


# ----------------------------------------------------------------------
# Phase 7: public landing experience
# ----------------------------------------------------------------------

METRIC_HELP = {
    "sharpe": ("How much reward you got for the risk you took, compared with "
               "leaving the money in a safe short term government investment. "
               "Higher is better; above 1 is generally considered strong."),
    "exp_return": ("The average yearly growth these holdings actually delivered "
                   "over the period analysed. It describes the past; it is not "
                   "a promise about the future."),
    "volatility": ("How much the value bounces around, expressed as a yearly "
                   "figure. Bigger bounces mean a rougher ride, in both "
                   "directions."),
    "beta": ("How much your portfolio tends to move when the US stock market "
             "moves. Around 1 means it moves with the market; below 1 means "
             "gentler moves; above 1 means bigger moves."),
    "sortino": ("Like the Sharpe ratio, but it only treats downward moves as "
                "risk, since upside surprises rarely bother anyone. Higher is "
                "better."),
    "mdd": ("The biggest fall from a high point to a low point over the period. "
            "The worst you would have lived through had you invested right at "
            "the top."),
    "var": ("On 95% of days, your daily loss should be smaller than this. "
            "Roughly 1 day in 20 you would expect to lose more."),
    "cvar": ("When one of those roughly 1 in 20 worse days does happen, this is "
             "the average loss on those days. It shows how bad the bad days "
             "tend to be."),
}

PUBLIC_ILLUSTRATION = (
    "These optimised weightings are what worked best over the past period under "
    "the stated rules. Past patterns shift, so treat them as illustrations of "
    "the trade offs, not predictions. The stability check below shows how "
    "quickly the optimum moves when the inputs change a little.")


def _pf_metric_cards(res: dict) -> None:
    m = res["user"]
    st.caption(f"USD terms, {m['start']} to {m['end']}, vs {res['benchmark']}. "
               "Hover the small question mark on each figure for a plain "
               "English explanation.")
    r1 = st.columns(4)
    r1[0].metric("Sharpe ratio", f"{m['sharpe']:.2f}", help=METRIC_HELP["sharpe"])
    r1[1].metric("Annualised return", pct(m["ann_return"]),
                 help=METRIC_HELP["exp_return"])
    r1[2].metric("Volatility", pct(m["volatility"]), help=METRIC_HELP["volatility"])
    r1[3].metric(f"Beta vs {res['benchmark']}", f"{m['beta']:.2f}",
                 help=METRIC_HELP["beta"])
    r2 = st.columns(4)
    r2[0].metric("Sortino ratio", f"{m['sortino']:.2f}", help=METRIC_HELP["sortino"])
    r2[1].metric("Max drawdown", pct(m["max_drawdown"]), help=METRIC_HELP["mdd"])
    r2[2].metric("Daily VaR 95%", pct(m["var_95"], 2), help=METRIC_HELP["var"])
    r2[3].metric("Daily CVaR 95%", pct(m["cvar_95"], 2), help=METRIC_HELP["cvar"])


def tab_your_portfolio(results: dict | None) -> None:
    st.caption("Build a portfolio below and see what the numbers say about it. "
               "Educational analytics on free market data, not investment advice.")

    # ----- search and add ------------------------------------------------
    c1, c2, c3 = st.columns([3, 4, 1])
    query = c1.text_input("Search a company or fund", key="pf_search",
                          placeholder="e.g. Apple, Vanguard S&P 500, VUSA.L")
    cache = st.session_state.setdefault("pf_search_cache", {})
    matches = []
    q = (query or "").strip()
    if len(q) >= 2:
        if q not in cache:
            try:
                cache[q] = analyzer.search_symbols(q)
            except Exception:
                cache[q] = []
        matches = cache[q]
    sel = None
    if matches:
        sel = c2.selectbox(
            "Matches", matches, key="pf_match",
            format_func=lambda m: f"{m['name']}  ({m['symbol']}, {m['exchange']})")
    elif q:
        c2.caption("No publicly traded matches found." if len(q) >= 2
                   else "Keep typing...")
    c3.write("")  # vertical alignment shim
    if c3.button("Add", key="pf_add", disabled=sel is None) and sel:
        rows = st.session_state.setdefault("pf_rows", [])
        if sel["symbol"] not in [r["ticker"] for r in rows]:
            rows.append({"ticker": sel["symbol"], "weight_pct": None,
                         "amount": None})

    # ----- holdings table -------------------------------------------------
    rows = st.session_state.get("pf_rows", [])
    base = pd.DataFrame(rows, columns=["ticker", "weight_pct", "amount"])
    st.caption("Give each holding a weight %, OR a money amount (any currency, "
               "just use the same one throughout). If both are filled in, the "
               "amounts win.")
    edited = st.data_editor(
        base, num_rows="dynamic", hide_index=True, width="stretch",
        key=f"pf_editor_{len(base)}",
        column_config={
            "ticker": st.column_config.TextColumn(
                "ticker", help="Yahoo symbol; use the search box above"),
            "weight_pct": st.column_config.NumberColumn(
                "weight %", min_value=0.0, max_value=100.0, step=0.1,
                format="%.1f"),
            "amount": st.column_config.NumberColumn(
                "amount", min_value=0.0, step=10.0,
                help="Currency agnostic; keep all amounts in one currency"),
        })
    st.session_state["pf_rows"] = [
        r for r in edited.to_dict("records")
        if str(r.get("ticker") or "").strip() != ""]

    if st.button("Analyze", key="pf_analyze", type="primary"):
        try:
            holdings, input_notes = analyzer.holdings_from_input(edited)
        except ValueError as exc:
            st.error(str(exc))
            return
        with st.spinner("Checking tickers on Yahoo..."):
            checks = analyzer.validate_tickers(list(holdings.index))
        bad = [t for t, c in checks.items() if not c["ok"]]
        if bad:
            for t in bad:
                st.error(f"'{t}' did not resolve on Yahoo Finance.")
            return
        for t, c in checks.items():
            st.caption(f"{t}: {c['name']} ({c['currency']})")
        with st.spinner("Fetching prices, converting to USD, computing..."):
            try:
                st.session_state["analysis"] = analyzer.analyze(
                    holdings, PROJECT_DIR, use_cache=False)
                st.session_state["pf_input_notes"] = input_notes
                st.session_state.pop("user_opt", None)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return

    res = st.session_state.get("analysis")
    if not res:
        return

    # ----- analysis output -------------------------------------------------
    st.divider()
    for n in st.session_state.get("pf_input_notes", []):
        st.info(n)
    _pf_metric_cards(res)
    with st.expander("Currency conversions applied"):
        for t, note in res["conversion_notes"].items():
            st.caption(f"{t}: {note}")

    # ----- improvement section ----------------------------------------------
    st.divider()
    st.subheader("Could these holdings have worked harder?")
    holdings = res["holdings"]
    n = len(holdings)
    if n < 2:
        st.info("Optimisation needs at least 2 holdings; add another above and "
                "re-analyse.")
        return
    st.caption(PUBLIC_ILLUSTRATION)
    min_cap = int(np.ceil(100.0 / n))
    cap = st.slider("Max single position", min_cap, 100, max(25, min_cap),
                    step=1, format="%d%%", key="pf_cap",
                    help="Long only and fully invested are always enforced") / 100
    if st.button("Optimise", key="pf_opt", type="primary"):
        prices = res["prices_usd"][list(holdings.index)]
        try:
            with st.spinner("Optimising and simulating 10,000 portfolios..."):
                opt = user_optimiser.optimise(prices, res["rf"], max_pos=cap)
                sens = user_optimiser.sensitivity(opt)
            with st.spinner("Walk forward backtest, roughly 30 to 60 seconds..."):
                bt = user_optimiser.backtest_user(opt, holdings,
                                                  res["bench_daily"])
            st.session_state["user_opt"] = {"opt": opt, "sens": sens, "bt": bt}
        except ValueError as exc:
            st.error(str(exc))
            return
    if st.session_state.get("user_opt"):
        _pf_optimise_results(res)


def _pf_optimise_results(res: dict) -> None:
    uo_state = st.session_state["user_opt"]
    opt, sens, bt = uo_state["opt"], uo_state["sens"], uo_state["bt"]
    if opt["cap_note"]:
        st.warning(opt["cap_note"])
    holdings = res["holdings"]
    tickers = list(holdings.index)
    rets = opt["rets"]
    labels = {"max_sharpe": "Max Sharpe (historical optimum)",
              "min_vol": "Min volatility", "max_div": "Max diversification",
              "equal_weight": "Equal weight"}
    mu, cov = opt["exp_ret"].values, opt["cov"].values

    def norm(w: pd.Series) -> np.ndarray:
        return (w / w.sum()).reindex(rets.columns).fillna(0.0).values

    # frontier with the user's dot and the optimum starred
    mc = opt["mc"]
    rng = np.random.default_rng(3)
    keep = rng.choice(len(mc["returns"]), size=min(4000, len(mc["returns"])),
                      replace=False)
    cloud = pd.DataFrame({"vol": mc["vols"][keep], "ret": mc["returns"][keep],
                          "sharpe": mc["sharpes"][keep]})

    def point(w: pd.Series, label: str) -> dict:
        wn = norm(w)
        return {"vol": float(np.sqrt(wn @ cov @ wn)), "ret": float(wn @ mu),
                "label": label}

    pts = pd.DataFrame([point(holdings, "Your portfolio")] +
                       [point(opt["weights"][k], lab)
                        for k, lab in labels.items()])
    layers = [alt.Chart(cloud).mark_circle(size=8, opacity=0.4).encode(
        x=alt.X("vol:Q", axis=alt.Axis(format=".0%"),
                title="Annualised volatility"),
        y=alt.Y("ret:Q", axis=alt.Axis(format=".0%"),
                title="Annualised return"),
        color=alt.Color("sharpe:Q", scale=alt.Scale(scheme="viridis"),
                        title="Sharpe"),
        tooltip=[alt.Tooltip("ret", format=".1%"),
                 alt.Tooltip("vol", format=".1%"),
                 alt.Tooltip("sharpe", format=".2f")])]
    if not opt["frontier"].empty:
        layers.append(alt.Chart(opt["frontier"]).mark_line(color="red",
                                                           strokeWidth=2)
                      .encode(x="vol:Q", y="ret:Q"))
    layers.append(alt.Chart(pts).mark_point(size=320, filled=True,
                                            stroke="black", strokeWidth=1).encode(
        x="vol:Q", y="ret:Q", shape=alt.Shape("label:N", title=None),
        color=alt.value("red"),
        tooltip=["label", alt.Tooltip("ret", format=".1%"),
                 alt.Tooltip("vol", format=".1%")]))
    st.altair_chart(alt.layer(*layers).resolve_scale(shape="independent")
                    .properties(height=420).interactive(), width="stretch")
    st.caption("Each grey to green dot is one simulated mix of your holdings; "
               "the red line is the best achievable trade off (the efficient "
               "frontier); the marked points are exact solutions. Your "
               "portfolio sits where its history puts it.")

    st.markdown("**If your goal were a higher Sharpe ratio, the historical "
                "optimum for these holdings looks like this**")
    cur = pd.Series(norm(holdings), index=rets.columns)
    prop = opt["weights"]["max_sharpe"]
    delta = prop - cur
    wc = pd.DataFrame({
        "current": cur, "optimised": prop, "change": delta,
        "direction": np.where(delta > 0.005, "increase",
                              np.where(delta < -0.005, "decrease", "unchanged")),
    }).sort_values("change", ascending=False)
    st.dataframe(wc.style.format({"current": "{:.1%}", "optimised": "{:.1%}",
                                  "change": "{:+.1%}"}), width="stretch")
    st.caption("Weight deltas describe the historical solution; they are not "
               "instructions.")

    with st.expander("Other alternatives: min volatility, max diversification, "
                     "equal weight"):
        def stats_col(w: pd.Series) -> dict:
            s = analyzer.portfolio_stats(res["prices_usd"][tickers], w,
                                         res["bench_daily"], res["rf"])
            s["avg_pairwise_corr"] = user_optimiser.weighted_avg_correlation(
                norm(w), opt["corr"])
            return {k: s[k] for k in ("ann_return", "volatility", "sharpe",
                                      "sortino", "max_drawdown", "beta",
                                      "avg_pairwise_corr", "hhi")}
        cols = {"Your portfolio": stats_col(holdings)}
        for key, lab in labels.items():
            cols[lab] = stats_col(opt["weights"][key])
        comp = pd.DataFrame(cols)
        disp = comp.copy()
        for c in disp.columns:
            disp[c] = [_fmt_stat(k, v) for k, v in comp[c].items()]
        st.dataframe(disp, width="stretch")
        st.caption("Min volatility chases the smoothest ride, max "
                   "diversification the least correlated mix, and equal weight "
                   "is the naive baseline any optimiser should beat.")

    with st.expander("How stable is the optimum?"):
        n_frag = int(sens["fragile"].sum())
        if n_frag:
            st.warning(f"{n_frag} position(s) are fragile: nudging one ticker's "
                       "expected return by 1% moves over 10% of the portfolio. "
                       "The shiny optimum is less solid than it looks.")
        else:
            st.success("The optimised weights survive a 1% nudge to any single "
                       "return estimate without moving more than 10% of the "
                       "portfolio.")
        sshow = sens.reset_index()[["ticker", "weight", "turnover_up",
                                    "turnover_down", "fragile"]]
        st.dataframe(sshow.style.format({
            "weight": "{:.1%}", "turnover_up": "{:.1%}",
            "turnover_down": "{:.1%}"}).map(
            lambda v: "color: red; font-weight: bold" if v is True else "",
            subset=["fragile"]), hide_index=True, width="stretch")

    with st.expander("Correlation between your holdings"):
        t1, t2, cmax = user_optimiser.highest_corr_pair(opt["corr"])
        cdf = opt["corr"].stack().reset_index()
        cdf.columns = ["a", "b", "corr"]
        heat = alt.Chart(cdf).mark_rect().encode(
            x=alt.X("a:N", title=None, sort=tickers),
            y=alt.Y("b:N", title=None, sort=tickers),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redyellowgreen",
                                                      reverse=True,
                                                      domain=[-1, 1])),
            tooltip=["a", "b", alt.Tooltip("corr", format=".2f")])
        outline = alt.Chart(cdf[((cdf["a"] == t1) & (cdf["b"] == t2)) |
                                ((cdf["a"] == t2) & (cdf["b"] == t1))]).mark_rect(
            fillOpacity=0, stroke="black", strokeWidth=3).encode(
            x=alt.X("a:N", sort=tickers), y=alt.Y("b:N", sort=tickers))
        st.altair_chart((heat + outline).properties(
            height=max(220, 36 * len(tickers))), width="stretch")
        st.caption(f"Most correlated pair (outlined): {t1} and {t2} at "
                   f"{cmax:.2f}. Holdings that move together add little "
                   "diversification to each other.")

    with st.expander("Out of sample walk forward backtest"):
        if "skipped" in bt:
            st.info(bt["skipped"])
        else:
            st.caption(f"{bt['start']} to {bt['end']}, {bt['n_rebalances']} "
                       "monthly re-optimisations using only data available at "
                       "the time, 10 bps costs on turnover. Out of sample "
                       "results are the honest measure, and they are usually "
                       "worse than the in sample optimum above. That gap is "
                       "the point of showing them.")
            growth = (1 + bt["daily"].fillna(0)).cumprod().reset_index()
            growth.columns = ["date"] + list(bt["daily"].columns)
            gm = growth.melt(id_vars="date", var_name="series",
                             value_name="growth")
            st.altair_chart(alt.Chart(gm).mark_line().encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("growth:Q", title="Growth of 1",
                        scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", title=None),
                tooltip=[alt.Tooltip("date:T"), "series",
                         alt.Tooltip("growth", format=".3f")],
            ).properties(height=340).interactive(), width="stretch")
            st.dataframe(bt["summary"].style.format({
                "cagr": "{:.2%}", "vol": "{:.2%}", "sharpe": "{:.2f}",
                "max_drawdown": "{:.1%}"}), width="stretch")


def tab_showcase(results: dict | None) -> None:
    st.caption("The author's own engine: an 18 ETF universe optimised monthly "
               "under macro basket constraints. These views are refreshed from "
               "the repository's committed outputs.")
    if results is None:
        st.warning("No engine output committed yet.")
        return
    inner = st.tabs(["Overview", "Efficient frontier", "CAPM / SML", "Risk",
                     "Backtest", "Robustness & BL", "Tear sheets"])
    with inner[0]:
        tab_overview(results)
    with inner[1]:
        tab_frontier(results)
    with inner[2]:
        tab_capm(results)
    with inner[3]:
        tab_risk(results)
    with inner[4]:
        tab_backtest(results)
    with inner[5]:
        tab_robustness(results)
    with inner[6]:
        tab_reports()


# ----------------------------------------------------------------------

def main() -> None:
    public = is_public()
    results = load_json(os.path.join(DASH_DIR, "results.json"))
    sidebar(results, public)
    st.title("Portfolio Engine dashboard")
    if public:
        st.warning("Educational analytics tool. Not investment advice.")

    def no_data() -> None:
        st.warning("No engine output found in outputs/dashboard/."
                   + ("" if public else " Run the optimisation from the sidebar "
                      "(tick 'Offline demo data' if you have no internet)."))

    sections: list[tuple] = []
    if public:
        # Phase 7: portfolio first landing; the engine sits behind one tab
        sections = [
            ("Your Portfolio", lambda: tab_your_portfolio(results)),
            ("Engine Showcase", lambda: tab_showcase(results)),
        ]
    else:
        if results is None:
            sections.append(("Overview", no_data))
        else:
            sections += [
                ("Overview", lambda: tab_overview(results)),
                ("Efficient frontier", lambda: tab_frontier(results)),
                ("CAPM / SML", lambda: tab_capm(results)),
                ("Risk", lambda: tab_risk(results)),
                ("Backtest", lambda: tab_backtest(results)),
                ("Robustness & BL", lambda: tab_robustness(results)),
            ]
        sections.append(("Portfolio Analyzer",
                         lambda: tab_analyzer(results, public)))
        sections.append(("Basket editor", lambda: tab_baskets(results)))
        sections.append(("Holdings & rebalance", tab_holdings))
        sections.append(("Tear sheets", tab_reports))

    tabs = st.tabs([label for label, _ in sections])
    for tab, (_, render) in zip(tabs, sections):
        with tab:
            render()


main()
