"""Risk management layer: limit checks, stress tests, correlation monitor."""
from __future__ import annotations

import numpy as np
import pandas as pd


def limit_checks(weights: pd.Series, max_pos: float, baskets: list[dict]) -> list[dict]:
    """Verify the final portfolio respects every stated limit."""
    checks = []
    worst = weights.max()
    checks.append({"check": f"Single ETF max {max_pos:.0%}",
                   "value": f"{worst:.1%} ({weights.idxmax()})",
                   "pass": bool(worst <= max_pos + 1e-6)})
    for b in baskets:
        tot = float(weights.values @ b["ind"])
        checks.append({"check": f"Basket '{b['name']}' in [{b['min']:.0%}, {b['max']:.0%}]",
                       "value": f"{tot:.1%}",
                       "pass": bool(b["min"] - 1e-6 <= tot <= b["max"] + 1e-6)})
    checks.append({"check": "Fully invested, long only",
                   "value": f"sum {weights.sum():.4f}, min {weights.min():.4f}",
                   "pass": bool(abs(weights.sum() - 1) < 1e-6 and weights.min() >= -1e-9)})
    return checks


def beta_stress_tests(port_beta: float) -> pd.DataFrame:
    """First order CAPM stress: portfolio loss approx beta times market shock."""
    scenarios = [
        ("Correction (market -10%)", -0.10),
        ("Bear market (market -20%)", -0.20),
        ("COVID style crash (market -34%)", -0.34),
        ("GFC style collapse (market -50%)", -0.50),
    ]
    rows = [{"scenario": name, "market_shock": shock,
             "est_portfolio_impact": port_beta * shock} for name, shock in scenarios]
    return pd.DataFrame(rows)


def historical_worst(daily: pd.Series) -> dict:
    monthly = (1 + daily).resample("ME").prod() - 1
    return {"worst_day": float(daily.min()),
            "worst_day_date": daily.idxmin().date(),
            "worst_month": float(monthly.min()),
            "worst_month_date": monthly.idxmin().date()}


def correlation_monitor(returns: pd.DataFrame, window: int = 60) -> dict:
    """Average pairwise correlation, full sample vs recent window. A jump
    in recent correlation means diversification is failing exactly when
    it's needed, which is the classic equity only failure mode.
    """
    def avg_pairwise(corr: pd.DataFrame) -> float:
        vals = corr.values[np.triu_indices_from(corr.values, k=1)]
        return float(np.nanmean(vals))

    full = avg_pairwise(returns.corr())
    recent = avg_pairwise(returns.tail(window).corr())
    return {"avg_corr_full": full, "avg_corr_recent": recent,
            "window_days": window,
            "warning": bool(recent > full + 0.10)}
