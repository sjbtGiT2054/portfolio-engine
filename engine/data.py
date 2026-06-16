"""Data layer: price download (Yahoo Finance via yfinance), caching,
returns computation, and a synthetic offline mode for testing.
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_FILE = os.path.join(DATA_DIR, "prices.csv")

# Populated by the live download: {ticker: human readable conversion note}.
# main.py reports the non USD conversions so every one is documented.
LAST_CONVERSION_NOTES: dict[str, str] = {}


def fetch_prices(tickers: list[str], lookback_years: int, offline: bool = False,
                 use_cache_on_fail: bool = True) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices, one column per ticker.

    Order of attempts: offline synthetic (if requested) -> live Yahoo ->
    cached CSV from a previous successful run.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if offline:
        return _synthetic_prices(tickers, lookback_years)

    try:
        prices = _download_yahoo(tickers, lookback_years)
        prices.to_csv(CACHE_FILE)
        return prices
    except Exception as exc:  # network failure, rate limit, etc.
        if use_cache_on_fail and os.path.exists(CACHE_FILE):
            warnings.warn(f"Live download failed ({exc}). Using cached prices from {CACHE_FILE}.")
            cached = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            cols = [t for t in tickers if t in cached.columns]
            if cols:
                return cached[cols]
        raise RuntimeError(
            f"Could not fetch prices and no usable cache found: {exc}\n"
            "Check your internet connection, or run with --offline for a synthetic demo."
        ) from exc


def _download_yahoo(tickers: list[str], lookback_years: int) -> pd.DataFrame:
    import yfinance as yf

    start = datetime.today() - timedelta(days=int(lookback_years * 365.25) + 30)
    raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    closes = closes.dropna(how="all").ffill()

    # Convert every non USD line to USD so returns and covariance live in one
    # currency, comparable with SPY and the USD risk free rate.
    closes = _to_usd(closes, start.strftime("%Y-%m-%d"))

    # Drop tickers with insufficient history (need at least ~1 year)
    min_obs = 252
    short = [c for c in closes.columns if closes[c].dropna().shape[0] < min_obs]
    if short:
        warnings.warn(f"Dropping tickers with under one year of history: {short}")
        closes = closes.drop(columns=short)
    # Align to common start so the covariance matrix is estimated on a full panel
    common_start = max(closes[c].first_valid_index() for c in closes.columns)
    closes = closes.loc[common_start:].dropna()
    if closes.shape[1] < 2:
        raise RuntimeError("Fewer than two tickers with usable data.")
    return closes


def _detect_currency(ticker: str) -> str:
    """Quote currency for a Yahoo symbol, fast_info first then get_info,
    same source the analyzer uses. Defaults to USD if undetectable."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    try:
        cur = tk.fast_info.get("currency")
        if cur:
            return cur
    except Exception:
        pass
    try:
        return tk.get_info().get("currency") or "USD"
    except Exception:
        return "USD"


def _to_usd(closes: pd.DataFrame, start: str) -> pd.DataFrame:
    """Convert each column to USD, reusing the analyzer currency logic.
    Non USD lines use the matching CURUSD=X (or GBPUSD=X for pence) daily
    close. A column whose FX cannot be fetched is dropped with a warning
    rather than left in a foreign currency. Records every conversion in
    LAST_CONVERSION_NOTES for the run to report.
    """
    import yfinance as yf

    from . import analyzer

    LAST_CONVERSION_NOTES.clear()
    fx_cache: dict[str, pd.Series] = {}
    out: dict[str, pd.Series] = {}
    for t in closes.columns:
        s = closes[t].dropna()
        currency = _detect_currency(t)
        if currency == "USD":
            out[t] = s
            LAST_CONVERSION_NOTES[t] = "USD, no conversion"
            continue
        sym = analyzer._fx_symbol(currency)
        try:
            if sym not in fx_cache:
                fxh = yf.Ticker(sym).history(start=start, auto_adjust=True)["Close"]
                if fxh.empty:
                    raise ValueError(f"no FX history for {sym}")
                fxh.index = fxh.index.tz_localize(None).normalize()
                fx_cache[sym] = fxh
            usd, note = analyzer.convert_to_usd(s, currency, fx_cache[sym])
            out[t] = usd
            LAST_CONVERSION_NOTES[t] = note
        except Exception as exc:
            warnings.warn(f"Dropping {t}: cannot convert {currency} to USD ({exc}).")
            LAST_CONVERSION_NOTES[t] = f"DROPPED, {currency} FX unavailable"
    if not out:
        raise RuntimeError("No tickers left after currency conversion.")
    return pd.DataFrame(out)


def _synthetic_prices(tickers: list[str], lookback_years: int, seed: int = 42) -> pd.DataFrame:
    """Factor model GBM: one market factor plus idiosyncratic noise, so the
    synthetic universe has a realistic positive correlation structure.
    Used only for offline demos and pipeline testing.
    """
    rng = np.random.default_rng(seed)
    n_days = int(lookback_years * 252)
    n = len(tickers)
    betas = rng.uniform(0.6, 1.5, n)
    annual_mkt_mu, annual_mkt_sig = 0.08, 0.16
    mkt = rng.normal(annual_mkt_mu / 252, annual_mkt_sig / np.sqrt(252), n_days)
    idio_sig = rng.uniform(0.10, 0.35, n) / np.sqrt(252)
    alphas = rng.normal(0.01 / 252, 0.02 / 252, n)
    rets = alphas + np.outer(mkt, betas) + rng.normal(0, 1, (n_days, n)) * idio_sig
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range(end=datetime.today(), periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=tickers)


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().dropna()
