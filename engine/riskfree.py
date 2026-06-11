"""Risk free rate: live 3 month US T bill yield with layered fallbacks.

Priority: Yahoo ^IRX -> FRED DTB3 csv -> manual rate from config.
Also fetches the 10 year (^TNX) for context on the Capital Market Line chart.
"""
from __future__ import annotations

import warnings


def get_risk_free_rate(cfg: dict, offline: bool = False) -> dict:
    """Return {'rate': float decimal, 'source': str, 'ten_year': float|None}."""
    rf_cfg = cfg["risk_free"]
    manual = float(rf_cfg.get("manual_rate", 0.045))

    if offline or rf_cfg.get("mode") == "manual":
        return {"rate": manual, "source": "manual (config)", "ten_year": None}

    rate, source = None, None
    try:
        rate = _yahoo_yield("^IRX")
        source = "Yahoo Finance ^IRX (13 week T bill)"
    except Exception as exc:
        warnings.warn(f"^IRX fetch failed: {exc}")
        try:
            rate = _fred_dtb3()
            source = "FRED DTB3 (3 month T bill, secondary market)"
        except Exception as exc2:
            warnings.warn(f"FRED fallback failed: {exc2}. Using manual rate.")
            rate, source = manual, "manual (live sources unavailable)"

    ten_year = None
    try:
        ten_year = _yahoo_yield(cfg["risk_free"].get("cml_long_rate_ticker", "^TNX"))
    except Exception:
        pass

    return {"rate": rate, "source": source, "ten_year": ten_year}


def _yahoo_yield(ticker: str) -> float:
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No data for {ticker}")
    return float(hist["Close"].dropna().iloc[-1]) / 100.0  # quoted in percent


def _fred_dtb3() -> float:
    import io

    import pandas as pd
    import requests

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    series = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna()
    if series.empty:
        raise RuntimeError("FRED DTB3 series empty")
    return float(series.iloc[-1]) / 100.0
