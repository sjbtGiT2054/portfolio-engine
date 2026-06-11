"""Unit tests for the Phase 5 Portfolio Analyzer (pure functions only,
no network)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import analyzer


def idx(n=500):
    return pd.bdate_range("2024-01-01", periods=n)


class TestCurrencyConversion(unittest.TestCase):
    def test_usd_passthrough(self):
        s = pd.Series([100.0, 101.0], index=idx(2))
        out, note = analyzer.convert_to_usd(s, "USD", None)
        pd.testing.assert_series_equal(out, s)
        self.assertIn("no conversion", note)

    def test_pence_divides_by_100_then_fx(self):
        s = pd.Series([5000.0, 5100.0], index=idx(2))     # pence
        fx = pd.Series([1.25, 1.30], index=idx(2))        # GBPUSD
        out, note = analyzer.convert_to_usd(s, "GBp", fx)
        np.testing.assert_allclose(out.values, [50.0 * 1.25, 51.0 * 1.30])
        self.assertIn("pence", note)
        self.assertIn("GBPUSD", note)

    def test_gbx_alias_handled(self):
        s = pd.Series([100.0], index=idx(1))
        fx = pd.Series([1.30], index=idx(1))
        out, _ = analyzer.convert_to_usd(s, "GBX", fx)
        self.assertAlmostEqual(float(out.iloc[0]), 1.30)

    def test_other_currency_multiplies_fx(self):
        s = pd.Series([200.0], index=idx(1))              # EUR
        fx = pd.Series([1.10], index=idx(1))              # EURUSD
        out, note = analyzer.convert_to_usd(s, "EUR", fx)
        self.assertAlmostEqual(float(out.iloc[0]), 220.0)
        self.assertIn("EURUSD", note)

    def test_fx_gap_forward_filled(self):
        s = pd.Series([100.0, 100.0, 100.0], index=idx(3))
        fx = pd.Series([1.20, np.nan, np.nan], index=idx(3)).dropna()
        out, _ = analyzer.convert_to_usd(s, "EUR", fx)
        np.testing.assert_allclose(out.values, [120.0, 120.0, 120.0])

    def test_missing_fx_raises(self):
        s = pd.Series([100.0], index=idx(1))
        with self.assertRaises(ValueError):
            analyzer.convert_to_usd(s, "GBp", None)


class TestHHI(unittest.TestCase):
    def test_single_holding_is_max(self):
        self.assertEqual(analyzer.hhi(pd.Series({"A": 1.0})), 1.0)

    def test_equal_weights(self):
        w = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        self.assertAlmostEqual(analyzer.hhi(w), 0.25)

    def test_normalises_unsummed_weights(self):
        w = pd.Series({"A": 2.0, "B": 2.0})
        self.assertAlmostEqual(analyzer.hhi(w), 0.5)


class TestFrontierGap(unittest.TestCase):
    def setUp(self):
        self.frontier = pd.DataFrame({"vol": [0.10, 0.15, 0.20],
                                      "ret": [0.06, 0.10, 0.12]})

    def test_point_below_frontier(self):
        gap = analyzer.frontier_gap(0.08, 0.15, self.frontier)
        self.assertAlmostEqual(gap["return_gap"], 0.02)          # 10% available at 15% vol
        self.assertAlmostEqual(gap["frontier_vol_at_user_ret"], 0.125)
        self.assertAlmostEqual(gap["vol_gap"], 0.025)
        self.assertEqual(gap["notes"], [])

    def test_point_on_frontier_has_zero_gaps(self):
        gap = analyzer.frontier_gap(0.10, 0.15, self.frontier)
        self.assertAlmostEqual(gap["return_gap"], 0.0)
        self.assertAlmostEqual(gap["vol_gap"], 0.0)

    def test_vol_beyond_frontier_clamps_with_note(self):
        gap = analyzer.frontier_gap(0.08, 0.40, self.frontier)
        self.assertAlmostEqual(gap["frontier_ret_at_user_vol"], 0.12)
        self.assertTrue(any("clamped" in n for n in gap["notes"]))

    def test_return_below_minimum_notes_minvol(self):
        gap = analyzer.frontier_gap(0.01, 0.15, self.frontier)
        self.assertAlmostEqual(gap["frontier_vol_at_user_ret"], 0.10)
        self.assertTrue(any("minimum volatility" in n for n in gap["notes"]))


class TestSingleAssetPortfolio(unittest.TestCase):
    """The owner's actual situation: 100% in one S&P 500 tracker."""

    def setUp(self):
        rng = np.random.default_rng(5)
        n = 750
        spy = rng.normal(0.0005, 0.01, n)
        # tracker = SPY plus small tracking noise
        tracker = spy + rng.normal(0, 0.0005, n)
        self.prices = pd.DataFrame(
            {"TRACKER": 100 * np.exp(np.cumsum(tracker))}, index=idx(n))
        self.bench = pd.Series(spy, index=idx(n))

    def test_full_stat_set(self):
        w = pd.Series({"TRACKER": 1.0})
        s = analyzer.portfolio_stats(self.prices, w, self.bench, rf=0.03)
        self.assertEqual(s["hhi"], 1.0)
        self.assertEqual(s["n_holdings"], 1)
        self.assertIsNone(s["avg_pairwise_corr"])
        self.assertAlmostEqual(s["beta"], 1.0, delta=0.05)   # tracks the index
        for key in ("ann_return", "volatility", "sharpe", "sortino",
                    "max_drawdown", "var_95", "cvar_95", "alpha"):
            self.assertTrue(np.isfinite(s[key]), f"{key} not finite")

    def test_diversification_benefit_zero_for_single_holding(self):
        d = analyzer.diversification_foregone(self.prices, pd.Series({"TRACKER": 1.0}))
        self.assertAlmostEqual(d["benefit_captured"], 0.0, places=10)

    def test_concentration_flags_single_holding(self):
        w = pd.Series({"TRACKER": 1.0})
        s = analyzer.portfolio_stats(self.prices, w, self.bench, rf=0.03)
        flags = analyzer.concentration_flags(w, s)
        self.assertTrue(any("HHI 1.00" in f for f in flags))
        self.assertTrue(all("should" not in f.lower() for f in flags),
                        "flags must stay analytics, not advice")

    def test_blend_zero_fraction_equals_user(self):
        rng = np.random.default_rng(6)
        u = pd.Series(rng.normal(0.0005, 0.01, 500), index=idx(500))
        o = pd.Series(rng.normal(0.0008, 0.009, 500), index=idx(500))
        b = analyzer.blend_diagnostics(u, o, rf=0.03)
        self.assertAlmostEqual(b.loc[0.0, "ann_return"], float(u.mean() * 252))
        self.assertAlmostEqual(b.loc[1.0, "ann_return"], float(o.mean() * 252))
        self.assertEqual(len(b), 5)


if __name__ == "__main__":
    unittest.main()
