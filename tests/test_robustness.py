"""Unit tests for the robustness layer."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import robustness

TICKERS = ["A", "B", "C", "D", "E"]
SETTINGS = {"annualisation": 252, "max_position": 0.5, "shrinkage": 0.1}


def synthetic_returns(n_days=600, seed=3):
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0004, 0.01, n_days)
    betas = rng.uniform(0.7, 1.3, len(TICKERS))
    idio = rng.normal(0, 0.008, (n_days, len(TICKERS)))
    return pd.DataFrame(np.outer(mkt, betas) + idio, columns=TICKERS)


class TestLedoitWolf(unittest.TestCase):
    def test_intensity_in_unit_interval_and_psd(self):
        rets = synthetic_returns()
        cov, intensity = robustness.ledoit_wolf_cov(rets)
        self.assertGreaterEqual(intensity, 0.0)
        self.assertLessEqual(intensity, 1.0)
        eigvals = np.linalg.eigvalsh(cov.values)
        self.assertTrue((eigvals > 0).all(), "LW covariance must be positive definite")

    def test_shrinks_toward_identity(self):
        """Off diagonal magnitude must not exceed the sample's."""
        rets = synthetic_returns()
        cov, intensity = robustness.ledoit_wolf_cov(rets)
        sample = rets.cov() * 252
        off = ~np.eye(len(TICKERS), dtype=bool)
        self.assertLessEqual(np.abs(cov.values[off]).sum(),
                             np.abs(sample.values[off]).sum() + 1e-12)
        self.assertGreater(intensity, 0.0)


class TestResampling(unittest.TestCase):
    def test_resampled_weights_shape_and_sums(self):
        rets = synthetic_returns()
        w_opt = pd.Series(0.2, index=TICKERS)
        out = robustness.resampled_weights(rets, 0.02, SETTINGS, baskets=[],
                                           w_opt=w_opt, n_boot=15)
        self.assertEqual(set(out.index), set(TICKERS))
        self.assertAlmostEqual(float(out["boot_mean"].sum()), 1.0, places=6)
        self.assertTrue((out["boot_std"] >= 0).all())
        self.assertTrue((out["n_samples"] <= 15).all())


class TestSensitivity(unittest.TestCase):
    def test_sensitivity_report_fields(self):
        from engine import optimiser, stats
        rets = synthetic_returns()
        exp_ret = stats.expected_returns(rets)
        cov = stats.covariance_matrix(rets, shrinkage=0.1)
        w = optimiser.max_sharpe(exp_ret, cov, 0.02, SETTINGS["max_position"], [])
        rep = robustness.sensitivity_report(exp_ret, cov, 0.02, SETTINGS, [], w)
        self.assertEqual(set(rep.index), set(TICKERS))
        for col in ("turnover_up", "turnover_down", "own_change_up",
                    "own_change_down", "fragile"):
            self.assertIn(col, rep.columns)
        self.assertTrue((rep["turnover_up"].dropna() >= 0).all())
        self.assertTrue(rep["fragile"].isin([True, False]).all())

    def test_positive_bump_does_not_cut_own_weight(self):
        """Raising an asset's expected return should never reduce its
        optimal weight (monotonicity, up to solver tolerance)."""
        from engine import optimiser, stats
        rets = synthetic_returns()
        exp_ret = stats.expected_returns(rets)
        cov = stats.covariance_matrix(rets, shrinkage=0.1)
        w = optimiser.max_sharpe(exp_ret, cov, 0.02, SETTINGS["max_position"], [])
        rep = robustness.sensitivity_report(exp_ret, cov, 0.02, SETTINGS, [], w)
        self.assertTrue((rep["own_change_up"].dropna() >= -1e-4).all())


if __name__ == "__main__":
    unittest.main()
