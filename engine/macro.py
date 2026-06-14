"""Macro data layer (Phase 8 macro views).

Free sources only, no API keys:
- FRED via the public fredgraph.csv endpoint
  (https://fred.stlouisfed.org/graph/fredgraph.csv?id=CODE)
- Yahoo Finance via yfinance for market series (^VIX, DX-Y.NYB, CL=F, ...)

Every series is cached to data/macro/<key>.csv with a fetch timestamp and
refreshed at most once a day. A missing or failed series degrades
gracefully: get_series returns None, and the caller greys out that
variable's slider rather than crashing the tab.

Nothing here forecasts. The "baseline path" is a deliberately naive trend
extrapolation, labelled as such, so the user has a neutral line to bend
away from. The whole point downstream is to show the portfolio's implied
response to a user's macro path, never to predict the macro path itself.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

MACRO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "macro")
REFRESH_HOURS = 24
HISTORY_YEARS = 10


# ----------------------------------------------------------------------
# Variable catalogue. transform: level | yoy. unit is for display only.
# active=True ships an interactive slider in v1; the rest are display only
# inside the "More indicators" expander. bounds clamp the user's slider to
# historically plausible levels.
# ----------------------------------------------------------------------

MACRO_VARS: list[dict] = [
    # --- Inflation and prices ---
    {"key": "cpi_yoy", "label": "CPI inflation (YoY)", "group": "Inflation and prices",
     "source": "fred", "code": "CPIAUCSL", "transform": "yoy", "unit": "%",
     "active": True, "bounds": (0.0, 10.0),
     "tooltip": "How fast the prices of everyday goods and services are rising "
                "over a year. High inflation eats into what your money buys and "
                "often pushes interest rates up."},
    {"key": "core_cpi_yoy", "label": "Core CPI (YoY)", "group": "Inflation and prices",
     "source": "fred", "code": "CPILFESL", "transform": "yoy", "unit": "%",
     "active": False, "bounds": (0.0, 8.0),
     "tooltip": "Inflation excluding food and energy, which jump around a lot. "
                "It shows the steadier underlying trend in prices."},
    {"key": "core_pce_yoy", "label": "Core PCE (YoY)", "group": "Inflation and prices",
     "source": "fred", "code": "PCEPILFE", "transform": "yoy", "unit": "%",
     "active": False, "bounds": (0.0, 8.0),
     "tooltip": "The inflation measure the US central bank watches most closely "
                "when setting interest rates."},
    {"key": "ppi_yoy", "label": "Producer prices (YoY)", "group": "Inflation and prices",
     "source": "fred", "code": "PPIACO", "transform": "yoy", "unit": "%",
     "active": False, "bounds": (-10.0, 20.0),
     "tooltip": "How fast prices are rising for businesses producing goods. It "
                "often moves before the prices you see in shops."},
    {"key": "breakeven_5y", "label": "5y breakeven inflation",
     "group": "Inflation and prices", "source": "fred", "code": "T5YIE",
     "transform": "level", "unit": "%", "active": False, "bounds": (0.0, 5.0),
     "tooltip": "The inflation rate markets expect on average over the next five "
                "years, read from bond prices."},
    {"key": "breakeven_10y", "label": "10y breakeven inflation",
     "group": "Inflation and prices", "source": "fred", "code": "T10YIE",
     "transform": "level", "unit": "%", "active": False, "bounds": (0.0, 5.0),
     "tooltip": "The inflation rate markets expect on average over the next ten "
                "years."},
    # --- Rates and policy ---
    {"key": "fed_funds", "label": "Fed funds rate", "group": "Rates and policy",
     "source": "fred", "code": "FEDFUNDS", "transform": "level", "unit": "%",
     "active": True, "bounds": (0.0, 8.0),
     "tooltip": "The interest rate the US central bank sets. It ripples through "
                "to borrowing costs everywhere and strongly affects share prices."},
    {"key": "ust_2y", "label": "2 year Treasury yield", "group": "Rates and policy",
     "source": "fred", "code": "DGS2", "transform": "level", "unit": "%",
     "active": False, "bounds": (0.0, 8.0),
     "tooltip": "The interest rate on a two year US government bond. It tracks "
                "where markets think short term rates are heading."},
    {"key": "ust_10y", "label": "10 year Treasury yield", "group": "Rates and policy",
     "source": "fred", "code": "DGS10", "transform": "level", "unit": "%",
     "active": True, "bounds": (0.0, 8.0),
     "tooltip": "The interest rate on a ten year US government bond, a benchmark "
                "for long term borrowing costs. Rising long rates can weigh on "
                "shares, especially fast growing ones."},
    {"key": "ust_30y", "label": "30 year Treasury yield", "group": "Rates and policy",
     "source": "fred", "code": "DGS30", "transform": "level", "unit": "%",
     "active": False, "bounds": (0.0, 8.0),
     "tooltip": "The interest rate on a thirty year US government bond, the "
                "longest standard maturity."},
    {"key": "curve_2s10s", "label": "2s10s curve slope", "group": "Rates and policy",
     "source": "fred", "code": "T10Y2Y", "transform": "level", "unit": "%",
     "active": True, "bounds": (-2.0, 3.0),
     "tooltip": "The gap between ten year and two year bond yields. When it goes "
                "negative (an inverted curve) it has often preceded recessions."},
    {"key": "real_10y", "label": "10y real (TIPS) yield", "group": "Rates and policy",
     "source": "fred", "code": "DFII10", "transform": "level", "unit": "%",
     "active": False, "bounds": (-2.0, 4.0),
     "tooltip": "The ten year interest rate after stripping out expected "
                "inflation. It is the real reward for lending to the government."},
    # --- Growth and activity ---
    {"key": "gdp_yoy", "label": "Real GDP growth (YoY)", "group": "Growth and activity",
     "source": "fred", "code": "GDPC1", "transform": "yoy", "unit": "%",
     "active": False, "bounds": (-5.0, 8.0),
     "tooltip": "How fast the whole economy is growing after inflation. Strong "
                "growth usually supports company profits."},
    {"key": "unemployment", "label": "Unemployment rate", "group": "Growth and activity",
     "source": "fred", "code": "UNRATE", "transform": "level", "unit": "%",
     "active": True, "bounds": (2.0, 12.0),
     "tooltip": "The share of people who want a job but cannot find one. Low "
                "unemployment means a strong economy; a sharp rise signals trouble."},
    {"key": "payrolls_yoy", "label": "Nonfarm payrolls (YoY)",
     "group": "Growth and activity", "source": "fred", "code": "PAYEMS",
     "transform": "yoy", "unit": "%", "active": False, "bounds": (-8.0, 8.0),
     "tooltip": "How many more (or fewer) people are employed than a year ago. A "
                "broad read on whether the job market is expanding."},
    {"key": "mfg_emp_yoy", "label": "Manufacturing employment (YoY, PMI proxy)",
     "group": "Growth and activity", "source": "fred", "code": "MANEMP",
     "transform": "yoy", "unit": "%", "active": False, "bounds": (-10.0, 10.0),
     "tooltip": "Factory jobs versus a year ago, used here as a free stand in for "
                "manufacturing surveys. A gauge of industrial momentum."},
    {"key": "industrial_production", "label": "Industrial production (YoY)",
     "group": "Growth and activity", "source": "fred", "code": "INDPRO",
     "transform": "yoy", "unit": "%", "active": False, "bounds": (-15.0, 15.0),
     "tooltip": "How much more (or less) the country's factories, mines and "
                "utilities are producing than a year ago."},
    {"key": "retail_sales_yoy", "label": "Retail sales (YoY)",
     "group": "Growth and activity", "source": "fred", "code": "RSAFS",
     "transform": "yoy", "unit": "%", "active": False, "bounds": (-15.0, 20.0),
     "tooltip": "How fast shoppers are spending compared with a year ago, a quick "
                "read on consumer demand."},
    {"key": "housing_starts", "label": "Housing starts", "group": "Growth and activity",
     "source": "fred", "code": "HOUST", "transform": "level", "unit": "k",
     "active": False, "bounds": (0.0, 2500.0),
     "tooltip": "How many new homes builders began this month. Housing is "
                "sensitive to interest rates and leads the economy."},
    # --- Sentiment and risk ---
    {"key": "vix", "label": "VIX (volatility index)", "group": "Sentiment and risk",
     "source": "yahoo", "code": "^VIX", "transform": "level", "unit": "",
     "active": True, "bounds": (10.0, 80.0),
     "tooltip": "Wall Street's fear gauge: how much investors expect the market "
                "to swing. High readings mean nervous, falling markets."},
    {"key": "umich_sentiment", "label": "Consumer sentiment (U Michigan)",
     "group": "Sentiment and risk", "source": "fred", "code": "UMCSENT",
     "transform": "level", "unit": "", "active": False, "bounds": (40.0, 110.0),
     "tooltip": "A survey of how confident households feel about their finances "
                "and the economy. Confident people spend more."},
    {"key": "hy_spread", "label": "High yield credit spread",
     "group": "Sentiment and risk", "source": "fred", "code": "BAMLH0A0HYM2",
     "transform": "level", "unit": "%", "active": False, "bounds": (2.0, 20.0),
     "tooltip": "The extra interest riskier companies must pay to borrow. It "
                "widens fast when investors get scared about defaults."},
    {"key": "ig_spread", "label": "Investment grade credit spread",
     "group": "Sentiment and risk", "source": "fred", "code": "BAMLC0A0CM",
     "transform": "level", "unit": "%", "active": False, "bounds": (0.5, 8.0),
     "tooltip": "The extra interest solid, safer companies pay to borrow over the "
                "government. A calmer cousin of the high yield spread."},
    # --- FX, commodities, liquidity ---
    {"key": "dxy", "label": "US dollar index (DXY)",
     "group": "FX, commodities, liquidity", "source": "yahoo", "code": "DX-Y.NYB",
     "transform": "level", "unit": "", "active": True, "bounds": (80.0, 120.0),
     "tooltip": "How strong the US dollar is against other major currencies. A "
                "strong dollar can hurt US exporters and emerging markets."},
    {"key": "wti", "label": "WTI crude oil", "group": "FX, commodities, liquidity",
     "source": "yahoo", "code": "CL=F", "transform": "level", "unit": "$",
     "active": True, "bounds": (20.0, 150.0),
     "tooltip": "The price of a barrel of US oil. It feeds into petrol, transport "
                "and inflation, and drives energy company profits."},
    {"key": "gold", "label": "Gold", "group": "FX, commodities, liquidity",
     "source": "yahoo", "code": "GC=F", "transform": "level", "unit": "$",
     "active": False, "bounds": (1000.0, 4000.0),
     "tooltip": "The price of gold per ounce. Investors often buy it as a haven "
                "when they are worried about inflation or instability."},
    {"key": "copper", "label": "Copper", "group": "FX, commodities, liquidity",
     "source": "yahoo", "code": "HG=F", "transform": "level", "unit": "$",
     "active": False, "bounds": (1.0, 8.0),
     "tooltip": "The price of copper, nicknamed Dr Copper because demand for it "
                "tracks the health of the global economy."},
    {"key": "m2_yoy", "label": "M2 money supply (YoY)",
     "group": "FX, commodities, liquidity", "source": "fred", "code": "M2SL",
     "transform": "yoy", "unit": "%", "active": False, "bounds": (-5.0, 30.0),
     "tooltip": "How fast the total amount of money in the economy is growing. "
                "Rapid growth can fuel inflation or asset prices."},
]

VARS_BY_KEY = {v["key"]: v for v in MACRO_VARS}
ACTIVE_KEYS = [v["key"] for v in MACRO_VARS if v["active"]]


# ----------------------------------------------------------------------
# Fetching and caching
# ----------------------------------------------------------------------

def _cache_path(key: str) -> str:
    return os.path.join(MACRO_DIR, f"{key}.csv")


def _fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(hours=REFRESH_HOURS)


def _fetch_fred(code: str) -> pd.Series:
    import requests
    from io import StringIO
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
               "Accept": "text/csv,*/*", "Accept-Language": "en-US,en;q=0.9"}
    # Short timeout so a slow or blocked FRED greys out the slider quickly
    # rather than hanging the tab; a healthy FRED responds in 1-2 seconds.
    r = requests.get(url, params={"id": code}, timeout=12, headers=headers)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." = gap
    return df.dropna().set_index("date")["value"]


def _fetch_yahoo(code: str) -> pd.Series:
    import yfinance as yf
    hist = yf.Ticker(code).history(period=f"{HISTORY_YEARS}y", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No history for {code}")
    s = hist["Close"]
    s.index = s.index.tz_localize(None).normalize()
    return s


def _transform(raw: pd.Series, how: str) -> pd.Series:
    if how == "yoy":
        # monthly level series to year on year percent change
        m = raw.resample("ME").last()
        return (m.pct_change(12) * 100).dropna()
    return raw


def get_series(var: dict, use_cache: bool = True) -> pd.Series | None:
    """Return the processed series for a variable, or None on any failure
    (the caller greys out the slider). Cached to data/macro/<key>.csv,
    refreshed at most daily."""
    os.makedirs(MACRO_DIR, exist_ok=True)
    path = _cache_path(var["key"])
    if use_cache and _fresh(path):
        try:
            s = pd.read_csv(path, index_col=0, parse_dates=True)["value"]
            if len(s):
                return s
        except Exception:
            pass
    try:
        raw = (_fetch_fred(var["code"]) if var["source"] == "fred"
               else _fetch_yahoo(var["code"]))
        s = _transform(raw, var["transform"]).dropna()
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=HISTORY_YEARS)
        s = s[s.index >= cutoff]
        if not len(s):
            return None
        s.rename("value").to_frame().to_csv(path)
        return s
    except Exception:
        # last resort: a stale cache beats a dead slider
        if os.path.exists(path):
            try:
                s = pd.read_csv(path, index_col=0, parse_dates=True)["value"]
                if len(s):
                    return s
            except Exception:
                return None
        return None


def current_value(series: pd.Series) -> float:
    return float(series.dropna().iloc[-1])


def baseline_path(series: pd.Series, horizon_years: int,
                  fit_months: int = 24) -> pd.DataFrame:
    """Naive trend baseline: fit a straight line to the last fit_months of
    monthly observations and extend its slope over the horizon. This is a
    neutral line to bend away from, explicitly NOT a forecast. Returns a
    monthly DataFrame with columns date, value."""
    m = series.resample("ME").last().dropna()
    last_date = m.index[-1]
    last_val = float(m.iloc[-1])

    tail = m.tail(fit_months)
    if len(tail) >= 6:
        x = np.arange(len(tail))
        slope = float(np.polyfit(x, tail.values, 1)[0])  # per month
    else:
        slope = 0.0

    months = horizon_years * 12
    future_dates = pd.date_range(last_date, periods=months + 1, freq="ME")[1:]
    values = last_val + slope * np.arange(1, months + 1)
    return pd.DataFrame({"date": future_dates, "value": values})
