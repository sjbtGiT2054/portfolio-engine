"""Unit tests for the Phase 6 user universe optimiser."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import user_optimiser as uo

TICKERS = ["AA", "BB", "CC", "DD", "EE"]


def synthetic_prices(n_days=600, n=5, seed=9):
    """Two factor synthetic market so correlations differ across pairs."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0.0004, 0.010, n_days)
    f2 = rng.normal(0.0002, 0.008, n_days)
    l1 = rng.uniform(0.3, 1.2, n)
    l2 = np.where(np.arange(n) % 2 == 0, rng.uniform(0.5, 1.0, n), 0.0)
    idio = rng.normal(0, 0.006, (n_days, n))
    rets = np.outer(f1, l1) + np.outer(f2, l2) + idio
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=TICKERS[:n])


class TestValidation(unittest.TestCase):
    def test_one_ticker_raises(self):
        p = synthetic_prices(n=2)[["AA"]]
        with self.assertRaises(ValueError) as cm:
            uo.optimise(p, rf=0.03)
        self.assertIn("at least 2", str(cm.exception))

    def test_short_history_raises(self):
        p = synthetic_prices(n_days=100)
        with self.assertRaises(ValueError) as cm:
            uo.optimise(p, rf=0.03)
        self.assertIn("one year", str(cm.exception))

    def test_infeasible_cap_raised_with_note(self):
        p = synthetic_prices()
        out = uo.optimise(p, rf=0.03, max_pos=0.10)   # 5 tickers need >= 20%
        self.assertIsNotNone(out["cap_note"])
        self.assertAlmostEqual(out["max_pos"], 0.20)


class TestSolutions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = synthetic_prices()
        cls.out = uo.optimise(cls.prices, rf=0.03, max_pos=0.40)

    def test_equal_weight_baseline_correct(self):
        eq = self.out["weights"]["equal_weight"]
        np.testing.assert_allclose(eq.values, 0.2)
        self.assertEqual(list(eq.index), TICKERS)

    def test_all_solutions_respect_constraints(self):
        for name, w in self.out["weights"].items():
            self.assertAlmostEqual(float(w.sum()), 1.0, places=6, msg=name)
            self.assertGreaterEqual(float(w.min()), -1e-9, msg=name)
            self.assertLessEqual(float(w.max()), 0.40 + 1e-6, msg=name)

    def test_cap_binds_when_tightened(self):
        loose = uo.optimise(self.prices, rf=0.03, max_pos=0.90)
        w_loose = loose["weights"]["max_sharpe"]
        if float(w_loose.max()) > 0.30:  # the unconstrained favourite exists
            tight = uo.optimise(self.prices, rf=0.03, max_pos=0.30)
            w_tight = tight["weights"]["max_sharpe"]
            self.assertLessEqual(float(w_tight.max()), 0.30 + 1e-6)
            self.assertAlmostEqual(float(w_tight.max()), 0.30, places=3)

    def test_max_div_lowers_weighted_correlation_vs_current(self):
        """The point of the max diversification portfolio: lower weighted
        average pairwise correlation than a concentrated current book."""
        corr = self.out["corr"]
        w_md = self.out["weights"]["max_div"].values
        current = np.array([0.6, 0.4, 0.0, 0.0, 0.0])  # concentrated in the
        # two highest factor loading names is irrelevant; any concentrated mix
        md_corr = uo.weighted_avg_correlation(w_md, corr)
        cur_corr = uo.weighted_avg_correlation(current, corr)
        self.assertLess(md_corr, cur_corr)
        # and it should not be worse than naive equal weight either
        eq_corr = uo.weighted_avg_correlation(np.full(5, 0.2), corr)
        self.assertLessEqual(md_corr, eq_corr + 1e-9)

    def test_min_vol_has_lowest_vol(self):
        cov = self.out["cov"].values
        vols = {name: float(np.sqrt(w.values @ cov @ w.values))
                for name, w in self.out["weights"].items()}
        self.assertEqual(min(vols, key=vols.get), "min_vol")

    def test_mc_and_frontier_present(self):
        self.assertEqual(self.out["mc"]["weights"].shape[0], uo.MC_SIMS)
        self.assertGreater(len(self.out["frontier"]), 5)


class TestTwoTickerEdgeCase(unittest.TestCase):
    def test_two_tickers_work(self):
        p = synthetic_prices(n=2)
        out = uo.optimise(p, rf=0.03, max_pos=0.60)
        for name, w in out["weights"].items():
            self.assertEqual(len(w), 2)
            self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
            self.assertLessEqual(float(w.max()), 0.60 + 1e-6)
        np.testing.assert_allclose(out["weights"]["equal_weight"].values, 0.5)

    def test_two_tickers_sensitivity_runs(self):
        p = synthetic_prices(n=2)
        out = uo.optimise(p, rf=0.03, max_pos=0.60)
        sens = uo.sensitivity(out)
        self.assertEqual(len(sens), 2)
        self.assertIn("fragile", sens.columns)

    def test_highest_corr_pair(self):
        p = synthetic_prices()
        out = uo.optimise(p, rf=0.03, max_pos=0.40)
        t1, t2, c = uo.highest_corr_pair(out["corr"])
        self.assertNotEqual(t1, t2)
        self.assertEqual(c, float(out["corr"].where(
            ~np.eye(5, dtype=bool)).max().max()))


class TestBacktestGates(unittest.TestCase):
    def test_too_many_tickers_skipped(self):
        n = 16
        rng = np.random.default_rng(2)
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (600, n)), axis=0)),
            index=pd.bdate_range("2023-01-02", periods=600),
            columns=[f"T{i}" for i in range(n)])
        out = uo.optimise(prices, rf=0.03, max_pos=0.25)
        bench = prices.iloc[:, 0].pct_change().dropna()
        bt = uo.backtest_user(out, pd.Series(1 / n, index=prices.columns), bench)
        self.assertIn("skipped", bt)
        self.assertIn("15 tickers", bt["skipped"])

    def test_short_history_skipped(self):
        p = synthetic_prices(n_days=400)
        out = uo.optimise(p, rf=0.03, max_pos=0.40)
        bench = p.iloc[:, 0].pct_change().dropna()
        bt = uo.backtest_user(out, pd.Series(0.2, index=p.columns), bench)
        self.assertIn("skipped", bt)
        self.assertIn("two years", bt["skipped"])

    def test_backtest_runs_and_reports_three_rows(self):
        p = synthetic_prices(n_days=900)
        out = uo.optimise(p, rf=0.03, max_pos=0.40)
        bench = p.mean(axis=1).pct_change().dropna()
        bt = uo.backtest_user(out, pd.Series(0.2, index=p.columns), bench)
        self.assertNotIn("skipped", bt)
        self.assertEqual(len(bt["summary"]), 3)
        for col in ("cagr", "vol", "sharpe", "max_drawdown"):
            self.assertIn(col, bt["summary"].columns)
        self.assertEqual(set(bt["daily"].columns), set(bt["summary"].index))


if __name__ == "__main__":
    unittest.main()
