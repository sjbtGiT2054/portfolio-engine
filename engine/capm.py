"""CAPM layer: betas vs the benchmark, Security Market Line data, alphas."""
from __future__ import annotations

import numpy as np
import pandas as pd


def capm_table(returns: pd.DataFrame, benchmark: pd.Series, exp_ret: pd.Series,
               rf: float, periods: int = 252) -> pd.DataFrame:
    """Per ETF: beta, CAPM expected return, realised annualised return, alpha.

    Beta is estimated by OLS of daily ETF returns on daily benchmark returns.
    The SML line is E[R] = rf + beta * (E[Rm] - rf).
    """
    bench_ann = float(benchmark.mean() * periods)
    var_m = float(benchmark.var())
    rows = []
    for t in returns.columns:
        beta = float(returns[t].cov(benchmark) / var_m) if var_m > 0 else np.nan
        capm_ret = rf + beta * (bench_ann - rf)
        realised = float(exp_ret[t])
        rows.append({"ticker": t, "beta": beta, "capm_expected": capm_ret,
                     "realised": realised, "alpha": realised - capm_ret})
    df = pd.DataFrame(rows).set_index("ticker")
    df.attrs["market_return"] = bench_ann
    df.attrs["rf"] = rf
    return df


def portfolio_beta(weights: np.ndarray, capm: pd.DataFrame) -> float:
    return float(weights @ capm["beta"].values)
