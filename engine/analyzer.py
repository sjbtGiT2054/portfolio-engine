"""Phase 5: Portfolio Analyzer.

Analyses an arbitrary user portfolio (any Yahoo tickers, any quote
currency) against the engine's optimal portfolio, the constrained
efficient frontier, and the benchmark. Read only calculations: nothing
here runs the optimisation pipeline or writes engine outputs. The
dashboard is permitted to call these functions directly.

Currency policy, explicit by design:
- Every series is converted to USD before any statistic is computed, so
  user holdings, the engine universe and SPY are always compared in the
  same currency.
- London quotes in pence (Yahoo currency 'GBp' or 'GBX') are divided by
  100 to get GBP, then multiplied by the GBPUSD=X daily close.
- Any other non USD currency CUR is multiplied by the CURUSD=X close.
- Every conversion applied is recorded in the returned notes so the
  output documents exactly what was done.

Framing: all outputs are historical analytics ("what the numbers were"),
never recommendations. Wording in returned labels sticks to that.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_PRICES = os.path.join(DATA_DIR, "analyzer_prices.csv")
CACHE_META = os.path.join(DATA_DIR, "analyzer_meta.json")
PENCE_CODES = {"GBp", "GBX"}


# ----------------------------------------------------------------------
# Currency conversion (pure, unit testable)
# ----------------------------------------------------------------------

def convert_to_usd(close: pd.Series, currency: str,
                   fx_usd: pd.Series | None) -> tuple[pd.Series, str]:
    """Convert a price series to USD. fx_usd is the CURUSD=X close series
    (units of USD per 1 unit of CUR), required for any non USD currency.
    Returns (usd_series, human readable note describing the conversion).
    """
    if currency == "USD":
        return close, "quoted in USD, no conversion"
    if currency in PENCE_CODES:
        if fx_usd is None:
            raise ValueError("GBp quote needs a GBPUSD=X series")
        gbp = close / 100.0
        fx = fx_usd.reindex(gbp.index).ffill()
        return gbp * fx, ("quoted in GBp (pence): divided by 100 to GBP, "
                          "then converted to USD via GBPUSD=X daily close")
    if fx_usd is None:
        raise ValueError(f"{currency} quote needs a {currency}USD=X series")
    fx = fx_usd.reindex(close.index).ffill()
    return close * fx, f"quoted in {currency}: converted to USD via {currency}USD=X daily close"


def _fx_symbol(currency: str) -> str:
    return "GBPUSD=X" if currency in PENCE_CODES else f"{currency}USD=X"


# ----------------------------------------------------------------------
# Price fetch with own cache
# ----------------------------------------------------------------------

def validate_tickers(tickers: list[str]) -> dict:
    """Resolve each ticker on Yahoo. Returns {ticker: {ok, name, currency}}.
    Used by the dashboard before running an analysis."""
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()
            name = info.get("longName") or info.get("shortName")
            cur = info.get("currency")
            out[t] = {"ok": bool(name and cur), "name": name, "currency": cur}
        except Exception as exc:
            out[t] = {"ok": False, "name": None, "currency": None,
                      "error": str(exc)[:120]}
    return out


def fetch_prices_usd(tickers: list[str], lookback_years: int = 5,
                     use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """USD converted adjusted closes for arbitrary tickers, cached in
    data/analyzer_prices.csv (reused within the same day for the same or
    a superset ticker list). Returns (prices, notes per ticker)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if use_cache and os.path.exists(CACHE_PRICES) and os.path.exists(CACHE_META):
        with open(CACHE_META, encoding="utf-8") as f:
            meta = json.load(f)
        if (meta.get("asof") == date.today().isoformat()
                and set(tickers) <= set(meta.get("notes", {}))):
            cached = pd.read_csv(CACHE_PRICES, index_col=0, parse_dates=True)
            return cached[tickers], {t: meta["notes"][t] for t in tickers}

    import yfinance as yf
    start = (datetime.today() - timedelta(days=int(lookback_years * 365.25) + 30))
    series, notes, fx_cache = {}, {}, {}
    for t in tickers:
        tk = yf.Ticker(t)
        hist = tk.history(start=start.strftime("%Y-%m-%d"), auto_adjust=True)
        if hist.empty:
            raise ValueError(f"No price history returned for '{t}'.")
        close = hist["Close"]
        close.index = close.index.tz_localize(None).normalize()
        currency = tk.get_info().get("currency") or "USD"
        fx = None
        if currency != "USD":
            sym = _fx_symbol(currency)
            if sym not in fx_cache:
                fxh = yf.Ticker(sym).history(start=start.strftime("%Y-%m-%d"))
                if fxh.empty:
                    raise ValueError(f"No FX history for {sym} needed by '{t}'.")
                fxs = fxh["Close"]
                fxs.index = fxs.index.tz_localize(None).normalize()
                fx_cache[sym] = fxs
            fx = fx_cache[sym]
        usd, note = convert_to_usd(close, currency, fx)
        series[t] = usd
        notes[t] = note

    prices = pd.DataFrame(series).dropna()
    if use_cache:
        prices.to_csv(CACHE_PRICES)
        with open(CACHE_META, "w", encoding="utf-8") as f:
            json.dump({"asof": date.today().isoformat(), "notes": notes}, f, indent=2)
    return prices, notes


# ----------------------------------------------------------------------
# Statistics (pure, unit testable)
# ----------------------------------------------------------------------

def hhi(weights: pd.Series) -> float:
    """Herfindahl Hirschman concentration index: sum of squared weights.
    1.0 = single holding (maximum possible), 1/n = equal weights."""
    w = weights / weights.sum()
    return float((w ** 2).sum())


def portfolio_stats(prices_usd: pd.DataFrame, weights: pd.Series,
                    bench_daily: pd.Series, rf: float,
                    periods: int = 252) -> dict:
    """Full stat set for a static weight portfolio, USD terms throughout."""
    rets = prices_usd.pct_change().dropna()
    w = (weights / weights.sum()).reindex(rets.columns).fillna(0.0)
    daily = pd.Series(rets.values @ w.values, index=rets.index)

    bench = bench_daily.reindex(daily.index).dropna()
    daily_b = daily.reindex(bench.index)

    # Beta and alpha on WEEKLY returns: cross listed funds (e.g. LSE) close
    # hours before the US benchmark, so same day returns are asynchronous
    # and daily OLS beta is biased toward zero. Weekly compounding restores
    # the overlap.
    wk = (1 + daily_b).resample("W-FRI").prod() - 1
    wk_b = (1 + bench).resample("W-FRI").prod() - 1
    var_m = float(wk_b.var())
    beta = float(wk.cov(wk_b) / var_m) if var_m > 0 else np.nan
    ann_ret = float(daily.mean() * periods)
    bench_ann = float(bench.mean() * periods)
    alpha = ann_ret - (rf + beta * (bench_ann - rf))

    vol = float(daily.std() * np.sqrt(periods))
    mdd, _ = stats.max_drawdown(daily)
    var = stats.value_at_risk(daily)
    n = len(weights)
    if n > 1:
        corr = rets.corr().values
        avg_corr = float(np.nanmean(corr[np.triu_indices(len(corr), k=1)]))
    else:
        avg_corr = None  # a single holding has no pairs

    return {
        "ann_return": ann_ret, "volatility": vol,
        "sharpe": (ann_ret - rf) / vol if vol > 0 else 0.0,
        "sortino": stats.sortino_ratio(daily, rf, periods),
        "max_drawdown": mdd,
        "var_95": var["var_hist"], "cvar_95": var["cvar"],
        "beta": beta, "alpha": alpha,
        "avg_pairwise_corr": avg_corr,
        "hhi": hhi(weights),
        "n_holdings": int((weights > 0).sum()),
        "daily": daily,
        "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
    }


def frontier_gap(user_ret: float, user_vol: float,
                 frontier: pd.DataFrame) -> dict:
    """Distance from the user portfolio to the constrained efficient
    frontier, measured both ways:
    - return_gap: frontier return at the user's volatility minus the
      user's return (return given up at the same risk)
    - vol_gap: the user's volatility minus the frontier volatility at the
      user's return (excess volatility carried for the same return)
    Edges are clamped to the frontier's range and noted.
    """
    f = frontier.sort_values("vol")
    notes = []

    v = float(np.clip(user_vol, f["vol"].min(), f["vol"].max()))
    if v != user_vol:
        notes.append("volatility outside the frontier's range; clamped to its "
                     f"{'minimum' if v > user_vol else 'maximum'}")
    ret_on_frontier = float(np.interp(v, f["vol"], f["ret"]))

    fr = f.sort_values("ret")
    if user_ret <= fr["ret"].min():
        vol_on_frontier = float(fr["vol"].iloc[0])
        notes.append("return below the frontier's minimum; compared against the "
                     "minimum volatility portfolio")
    elif user_ret >= fr["ret"].max():
        vol_on_frontier = float(fr["vol"].iloc[-1])
        notes.append("return above the frontier's maximum; compared against the "
                     "maximum return portfolio")
    else:
        vol_on_frontier = float(np.interp(user_ret, fr["ret"], fr["vol"]))

    return {
        "return_gap": ret_on_frontier - user_ret,
        "frontier_ret_at_user_vol": ret_on_frontier,
        "vol_gap": user_vol - vol_on_frontier,
        "frontier_vol_at_user_ret": vol_on_frontier,
        "notes": notes,
    }


def diversification_foregone(prices_usd: pd.DataFrame, weights: pd.Series,
                             periods: int = 252) -> dict:
    """Weighted average constituent volatility vs portfolio volatility.
    The difference is the diversification benefit actually captured; for
    a single holding it is zero by construction."""
    rets = prices_usd.pct_change().dropna()
    w = (weights / weights.sum()).reindex(rets.columns).fillna(0.0)
    const_vols = rets.std() * np.sqrt(periods)
    weighted_avg = float((const_vols * w).sum())
    port_vol = float(pd.Series(rets.values @ w.values, index=rets.index).std()
                     * np.sqrt(periods))
    return {"weighted_avg_constituent_vol": weighted_avg,
            "portfolio_vol": port_vol,
            "benefit_captured": weighted_avg - port_vol}


def blend_diagnostics(user_daily: pd.Series, optimal_daily: pd.Series,
                      rf: float, periods: int = 252,
                      fractions: tuple = (0.0, 0.10, 0.20, 0.30, 1.0)) -> pd.DataFrame:
    """Historical effect of reallocating a fraction of the portfolio to
    the engine optimal: blended daily returns, then Sharpe, vol and max
    drawdown per blend. 0.0 = the user portfolio as it was, 1.0 = the
    engine optimal. Historical arithmetic, not a forecast."""
    idx = user_daily.index.intersection(optimal_daily.index)
    u, o = user_daily.reindex(idx), optimal_daily.reindex(idx)
    rows = []
    for f in fractions:
        blend = (1 - f) * u + f * o
        ann = float(blend.mean() * periods)
        vol = float(blend.std() * np.sqrt(periods))
        mdd, _ = stats.max_drawdown(blend)
        rows.append({"reallocation": f, "ann_return": ann, "volatility": vol,
                     "sharpe": (ann - rf) / vol if vol > 0 else 0.0,
                     "max_drawdown": mdd})
    return pd.DataFrame(rows).set_index("reallocation")


# ----------------------------------------------------------------------
# Rule based flags (analytics wording only)
# ----------------------------------------------------------------------

def concentration_flags(weights: pd.Series, stats_d: dict) -> list[str]:
    flags = []
    h = stats_d["hhi"]
    n = stats_d["n_holdings"]
    if n == 1:
        flags.append("Single holding concentration: HHI 1.00, the maximum "
                     "possible. All risk is one instrument's risk.")
    elif h > 0.30:
        flags.append(f"High concentration: HHI {h:.2f} across {n} holdings "
                     f"(equal weighting would be {1/n:.2f}).")
    if stats_d["avg_pairwise_corr"] is None:
        flags.append("Diversification benefit is zero by construction: one "
                     "holding has no pairwise correlations to exploit.")
    b = stats_d["beta"]
    if not np.isnan(b):
        if b > 1.1:
            flags.append(f"Beta {b:.2f} vs SPY: amplified market exposure.")
        elif b < 0.9:
            flags.append(f"Beta {b:.2f} vs SPY: dampened market exposure.")
        else:
            flags.append(f"Beta {b:.2f} vs SPY: effectively full market exposure, "
                         "as expected for a broad index holding.")
    return flags


# ----------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------

def analyze(holdings: pd.Series, project_dir: str,
            lookback_years: int = 5, rf: float | None = None,
            periods: int = 252) -> dict:
    """Run the full analysis of a user portfolio against the engine's
    latest outputs. Reads (never writes): outputs/dashboard/results.json,
    outputs/dashboard/frontier.csv, data/prices.csv, optimal weights.
    Raises with a clear message when a needed engine output is missing.
    """
    out_dir = os.path.join(project_dir, "outputs", "dashboard")
    with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
        results = json.load(f)
    frontier = pd.read_csv(os.path.join(out_dir, "frontier.csv"))
    if rf is None:
        rf = float(results["risk_free"]["rate"])

    # user portfolio, SPY and the engine universe, all in USD
    tickers = list(holdings.index)
    bench_ticker = results.get("benchmark", "SPY")
    prices_user, notes = fetch_prices_usd(tickers + [bench_ticker], lookback_years)
    bench_daily = prices_user[bench_ticker].pct_change().dropna()
    user_prices = prices_user[tickers]

    user = portfolio_stats(user_prices, holdings, bench_daily, rf, periods)
    spy = portfolio_stats(prices_user[[bench_ticker]],
                          pd.Series({bench_ticker: 1.0}), bench_daily, rf, periods)

    # engine optimal daily series from the engine's own cached prices
    uni_prices = pd.read_csv(os.path.join(project_dir, "data", "prices.csv"),
                             index_col=0, parse_dates=True)
    w_opt = pd.Series(results["weights"])
    cols = [t for t in w_opt.index if t in uni_prices.columns]
    opt_stats = portfolio_stats(uni_prices[cols], w_opt[cols], bench_daily, rf, periods)

    gap = frontier_gap(user["ann_return"], user["volatility"], frontier)
    blends = blend_diagnostics(user["daily"], opt_stats["daily"], rf, periods)
    divers = diversification_foregone(user_prices, holdings, periods)
    flags = concentration_flags(holdings, user)

    return {
        "rf": rf, "benchmark": bench_ticker,
        "beta_method": ("weekly returns (cross listed funds close before the US "
                        "benchmark; daily betas are biased toward zero)"),
        "conversion_notes": notes,
        "user": user, "optimal": opt_stats, "spy": spy,
        "frontier_gap": gap, "blends": blends,
        "diversification": divers, "flags": flags,
        "frontier": frontier,
        "disclaimer": ("Historical analytics on free Yahoo Finance data, USD "
                       "terms. Educational comparison only, not investment "
                       "advice or a recommendation to trade."),
    }


def comparison_table(result: dict) -> pd.DataFrame:
    """Side by side: user portfolio vs engine optimal vs benchmark."""
    rows = ["ann_return", "volatility", "sharpe", "sortino", "max_drawdown",
            "var_95", "cvar_95", "beta", "alpha", "avg_pairwise_corr",
            "hhi", "n_holdings"]
    df = pd.DataFrame({
        "user_portfolio": {k: result["user"].get(k) for k in rows},
        "engine_optimal": {k: result["optimal"].get(k) for k in rows},
        result["benchmark"]: {k: result["spy"].get(k) for k in rows},
    })
    return df
