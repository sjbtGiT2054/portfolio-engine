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
