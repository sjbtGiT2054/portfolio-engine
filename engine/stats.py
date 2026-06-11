"""Return and risk estimation plus portfolio level metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_returns(returns: pd.DataFrame, periods: int = 252) -> pd.Series:
    """Annualised arithmetic mean daily returns (standard MPT input)."""
    return returns.mean() * periods


def covariance_matrix(returns: pd.DataFrame, periods: int = 252,
                      shrinkage: float = 0.0) -> pd.DataFrame:
    """Annualised sample covariance with optional shrinkage toward the
    diagonal. Shrinkage dampens estimation noise in off diagonal terms,
    which is the main cause of unstable optimised weights.
    """
    sample = returns.cov() * periods
    if shrinkage <= 0:
        return sample
    target = np.diag(np.diag(sample.values))
    shrunk = (1 - shrinkage) * sample.values + shrinkage * target
    return pd.DataFrame(shrunk, index=sample.index, columns=sample.columns)


def portfolio_performance(weights: np.ndarray, exp_ret: np.ndarray,
                          cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    """Return (annual return, annual volatility, Sharpe ratio)."""
    ret = float(weights @ exp_ret)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def portfolio_daily_series(weights: np.ndarray, returns: pd.DataFrame) -> pd.Series:
    """Daily portfolio return series for a static weight vector."""
    return pd.Series(returns.values @ weights, index=returns.index, name="portfolio")


def sortino_ratio(daily: pd.Series, rf: float, periods: int = 252) -> float:
    excess = daily - rf / periods
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf")
    dd = float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods))
    return float(excess.mean() * periods / dd) if dd > 0 else 0.0


def max_drawdown(daily: pd.Series) -> tuple[float, pd.Series]:
    """Return (max drawdown as negative decimal, drawdown series)."""
    wealth = (1 + daily).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return float(dd.min()), dd


def value_at_risk(daily: pd.Series, level: float = 0.95) -> dict:
    """Historical and parametric daily VaR plus CVaR, as positive losses."""
    var_hist = float(-np.percentile(daily, (1 - level) * 100))
    mu, sigma = daily.mean(), daily.std()
    from scipy.stats import norm

    var_param = float(-(mu + sigma * norm.ppf(1 - level)))
    tail = daily[daily <= -var_hist]
    cvar = float(-tail.mean()) if not tail.empty else var_hist
    return {"var_hist": var_hist, "var_param": var_param, "cvar": cvar, "level": level}


def annualise_series(daily: pd.Series, periods: int = 252) -> dict:
    n = len(daily)
    cum = float((1 + daily).prod())
    cagr = cum ** (periods / n) - 1 if n > 0 else 0.0
    vol = float(daily.std() * np.sqrt(periods))
    return {"cagr": cagr, "vol": vol, "cum_return": cum - 1}
