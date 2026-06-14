"""Unit tests for the macro data layer and sensitivity / view maths."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import macro, macro_sensitivity as ms


class TestSensitivityEstimation(unittest.TestCase):
    def test_recovers_injected_sign_and_magnitude(self):
        """Inject return_t = 2.5 * delta_var_t + noise and check the OLS
        slope recovers the positive sign and rough magnitude."""
        rng = np.random.default_rng(0)
        months = pd.date_range("2016-01-31", periods=110, freq="ME")
        var_level = pd.Series(np.cumsum(rng.normal(0, 0.3, len(months))) + 3,
                              index=months)
        dvar = macro_sensitivity_change = ms.monthly_var_change(var_level)
        y = pd.Series(2.5 * dvar.values + rng.normal(0, 0.01, len(dvar)),
                      index=dvar.index)
        res = ms.ols_sensitivity(y, dvar)
        self.assertGreater(res["coef"], 1.5)
        self.assertLess(res["coef"], 3.5)
        self.assertGreater(res["tstat"], 1.645)
        self.assertFalse(res["low_confidence"])

    def test_negative_relationship_recovered(self):
        rng = np.random.default_rng(1)
        months = pd.date_range("2016-01-31", periods=90, freq="ME")
        dvar = pd.Series(rng.normal(0, 0.5, len(months)), index=months)
        y = pd.Series(-1.8 * dvar.values + rng.normal(0, 0.005, len(dvar)),
                      index=dvar.index)
        res = ms.ols_sensitivity(y, dvar)
        self.assertLess(res["coef"], 0)

    def test_pure_noise_flagged_low_confidence(self):
        rng = np.random.default_rng(2)
        months = pd.date_range("2016-01-31", periods=80, freq="ME")
        dvar = pd.Series(rng.normal(0, 1, len(months)), index=months)
        y = pd.Series(rng.normal(0, 0.05, len(months)), index=months)
        res = ms.ols_sensitivity(y, dvar)
        self.assertTrue(res["low_confidence"])

    def test_too_few_points_low_confidence(self):
        idx = pd.date_range("2023-01-31", periods=5, freq="ME")
        res = ms.ols_sensitivity(pd.Series(np.arange(5.0), index=idx),
                                 pd.Series(np.arange(5.0), index=idx))
        self.assertTrue(res["low_confidence"])
        self.assertEqual(res["n"], 5)

    def test_estimate_sensitivities_shape(self):
        rng = np.random.default_rng(3)
        days = pd.date_range("2018-01-01", periods=900, freq="B")
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (len(days), 2)), axis=0)),
            index=days, columns=["AAA", "BBB"])
        months = pd.date_range("2018-01-31", periods=40, freq="ME")
        series = {"cpi_yoy": pd.Series(3 + np.cumsum(rng.normal(0, 0.1, 40)),
                                       index=months)}
        out = ms.estimate_sensitivities(prices, series)
        self.assertIn("cpi_yoy", out)
        self.assertEqual(set(out["cpi_yoy"]), {"AAA", "BBB"})
        self.assertIn("coef", out["cpi_yoy"]["AAA"])

    def test_none_series_skipped(self):
        rng = np.random.default_rng(4)
        days = pd.date_range("2018-01-01", periods=600, freq="B")
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(days), 1)), axis=0)),
            index=days, columns=["AAA"])
        out = ms.estimate_sensitivities(prices, {"vix": None})
        self.assertNotIn("vix", out)


class TestViewConstruction(unittest.TestCase):
    def test_tilt_maths(self):
        sens = {"cpi_yoy": {"AAA": {"coef": 0.02}, "BBB": {"coef": -0.01}}}
        # CPI ends 2pp above baseline over a 2 year horizon
        tilts, confs = ms.build_view_tilts(
            sens, {"cpi_yoy": 2.0}, horizon_years=2,
            var_confidence={"cpi_yoy": "high"}, tickers=["AAA", "BBB"])
        # 0.02 * 2 / 2 = 0.02 ; -0.01 * 2 / 2 = -0.01
        self.assertAlmostEqual(tilts["AAA"], 0.02)
        self.assertAlmostEqual(tilts["BBB"], -0.01)
        self.assertAlmostEqual(confs["AAA"], 0.8)  # single variable -> its confidence

    def test_confidence_is_magnitude_weighted_blend(self):
        sens = {"cpi_yoy": {"AAA": {"coef": 0.03}},   # big contributor
                "vix": {"AAA": {"coef": 0.001}}}       # tiny contributor
        _, confs = ms.build_view_tilts(
            sens, {"cpi_yoy": 1.0, "vix": 1.0}, horizon_years=1,
            var_confidence={"cpi_yoy": "low", "vix": "high"}, tickers=["AAA"])
        # weighted toward the low confidence big driver
        self.assertLess(confs["AAA"], 0.5)
        self.assertGreater(confs["AAA"], 0.2)

    def test_zero_deviation_no_views(self):
        sens = {"cpi_yoy": {"AAA": {"coef": 0.02}}}
        tilts, _ = ms.build_view_tilts(sens, {"cpi_yoy": 0.0}, 2,
                                       {"cpi_yoy": "high"}, ["AAA"])
        pi = pd.Series({"AAA": 0.05})
        views = ms.build_bl_views(pi, tilts, {"AAA": 0.8})
        self.assertEqual(views, [])

    def test_bl_view_is_prior_plus_tilt(self):
        pi = pd.Series({"AAA": 0.05, "BBB": 0.04})
        views = ms.build_bl_views(pi, {"AAA": 0.02, "BBB": 0.0},
                                  {"AAA": 0.5, "BBB": 0.5})
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["assets"], ["AAA"])
        self.assertAlmostEqual(views[0]["excess_return"], 0.07)
        self.assertAlmostEqual(views[0]["confidence"], 0.5)

    def test_views_feed_black_litterman_cleanly(self):
        """End to end: built views must be consumable by the BL posterior
        and a nonzero tilt must move the posterior in the tilt direction."""
        from engine import black_litterman as bl
        tickers = ["AAA", "BBB", "CCC"]
        cov = pd.DataFrame(np.diag([0.04, 0.05, 0.06]), index=tickers,
                           columns=tickers)
        pi = pd.Series({"AAA": 0.05, "BBB": 0.05, "CCC": 0.05})
        tilts = {"AAA": 0.03, "BBB": 0.0, "CCC": 0.0}
        views = ms.build_bl_views(pi, tilts, {t: 0.8 for t in tickers})
        post = bl.posterior_returns(pi, cov, views, tau=0.05)
        self.assertGreater(post["AAA"], pi["AAA"])  # upward tilt lifts AAA


class TestBaselinePath(unittest.TestCase):
    def test_flat_series_gives_flat_baseline(self):
        idx = pd.date_range("2020-01-31", periods=60, freq="ME")
        s = pd.Series(np.full(60, 3.0), index=idx)
        path = macro.baseline_path(s, horizon_years=2)
        self.assertEqual(len(path), 24)
        np.testing.assert_allclose(path["value"].values, 3.0, atol=1e-9)

    def test_upward_trend_extrapolates_up(self):
        idx = pd.date_range("2020-01-31", periods=60, freq="ME")
        s = pd.Series(np.linspace(1.0, 3.0, 60), index=idx)  # rising
        path = macro.baseline_path(s, horizon_years=1)
        self.assertEqual(len(path), 12)
        self.assertGreater(path["value"].iloc[-1], 3.0)  # continues upward

    def test_horizon_length(self):
        idx = pd.date_range("2020-01-31", periods=40, freq="ME")
        s = pd.Series(np.arange(40.0), index=idx)
        for h in (1, 3, 5):
            self.assertEqual(len(macro.baseline_path(s, h)), h * 12)


class TestGracefulDegradation(unittest.TestCase):
    def test_get_series_returns_none_on_bad_source(self):
        """A variable whose fetch fails and has no cache returns None so
        the caller can grey out the slider rather than crash."""
        bad = {"key": "_nonexistent_test_var_xyz", "source": "fred",
               "code": "THIS_CODE_DOES_NOT_EXIST_XYZ", "transform": "level"}
        # ensure no stale cache file exists for this key
        p = macro._cache_path(bad["key"])
        if os.path.exists(p):
            os.remove(p)
        self.assertIsNone(macro.get_series(bad, use_cache=False))

    def test_catalogue_integrity(self):
        keys = [v["key"] for v in macro.MACRO_VARS]
        self.assertEqual(len(keys), len(set(keys)))  # unique keys
        self.assertEqual(len(macro.ACTIVE_KEYS), 8)  # eight active sliders in v1
        for v in macro.MACRO_VARS:
            self.assertIn(v["source"], ("fred", "yahoo"))
            self.assertIn(v["transform"], ("level", "yoy"))
            self.assertEqual(len(v["bounds"]), 2)
            self.assertTrue(v["tooltip"])


if __name__ == "__main__":
    unittest.main()
