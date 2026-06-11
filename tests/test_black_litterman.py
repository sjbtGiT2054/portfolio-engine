"""Unit tests for the Black Litterman maths. Run from the project root:

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import black_litterman as bl

TICKERS = ["AAA", "BBB", "CCC"]


def toy_market():
    """Three assets with a known covariance and market weights."""
    cov = pd.DataFrame(
        [[0.040, 0.012, 0.008],
         [0.012, 0.090, 0.015],
         [0.008, 0.015, 0.0625]],
        index=TICKERS, columns=TICKERS)
    w_mkt = pd.Series([0.5, 0.3, 0.2], index=TICKERS)
    delta = 2.5
    pi = bl.equilibrium_returns(cov, w_mkt, delta)
    return cov, w_mkt, delta, pi


class TestEquilibrium(unittest.TestCase):
    def test_zero_views_posterior_equals_equilibrium_exactly(self):
        cov, _, _, pi = toy_market()
        mu = bl.posterior_returns(pi, cov, views=[], tau=0.05)
        np.testing.assert_array_equal(mu.values, pi.values)

    def test_reverse_optimisation_recovers_market_weights(self):
        """Unconstrained tangency weights from pi must be w_mkt: the whole
        point of reverse optimisation is that the equilibrium returns make
        the market portfolio optimal. w* proportional to Sigma^-1 pi."""
        cov, w_mkt, _, pi = toy_market()
        w_star = np.linalg.solve(cov.values, pi.values)
        w_star /= w_star.sum()
        np.testing.assert_allclose(w_star, w_mkt.values, atol=1e-12)

    def test_equilibrium_scales_with_delta(self):
        cov, w_mkt, _, _ = toy_market()
        pi1 = bl.equilibrium_returns(cov, w_mkt, 2.0)
        pi2 = bl.equilibrium_returns(cov, w_mkt, 4.0)
        np.testing.assert_allclose(pi2.values, 2.0 * pi1.values)


class TestViews(unittest.TestCase):
    def test_absolute_view_tilts_the_right_way(self):
        cov, _, _, pi = toy_market()
        view_up = [{"name": "v", "assets": ["AAA"],
                    "excess_return": float(pi["AAA"]) + 0.05, "confidence": 0.5}]
        mu = bl.posterior_returns(pi, cov, view_up, tau=0.05)
        self.assertGreater(mu["AAA"], pi["AAA"])

        view_dn = [{"name": "v", "assets": ["AAA"],
                    "excess_return": float(pi["AAA"]) - 0.05, "confidence": 0.5}]
        mu = bl.posterior_returns(pi, cov, view_dn, tau=0.05)
        self.assertLess(mu["AAA"], pi["AAA"])

    def test_relative_view_widens_the_spread(self):
        cov, _, _, pi = toy_market()
        spread_prior = pi["BBB"] - pi["CCC"]
        views = [{"name": "v", "long": ["BBB"], "short": ["CCC"],
                  "outperformance": float(spread_prior) + 0.04, "confidence": 0.5}]
        mu = bl.posterior_returns(pi, cov, views, tau=0.05)
        self.assertGreater(mu["BBB"] - mu["CCC"], spread_prior)

    def test_higher_confidence_tilts_more(self):
        cov, _, _, pi = toy_market()
        def tilt(conf):
            views = [{"name": "v", "assets": ["AAA"],
                      "excess_return": float(pi["AAA"]) + 0.05, "confidence": conf}]
            mu = bl.posterior_returns(pi, cov, views, tau=0.05)
            return abs(mu["AAA"] - pi["AAA"])
        self.assertGreater(tilt(0.9), tilt(0.5))
        self.assertGreater(tilt(0.5), tilt(0.1))

    def test_view_only_on_view_matrices(self):
        """P rows: absolute sums to +1, relative sums to 0."""
        views = [{"name": "a", "assets": ["AAA", "BBB"],
                  "excess_return": 0.05, "confidence": 0.5},
                 {"name": "r", "long": ["BBB"], "short": ["AAA", "CCC"],
                  "outperformance": 0.02, "confidence": 0.3}]
        P, Q, conf = bl.view_matrices(views, TICKERS)
        np.testing.assert_allclose(P.sum(axis=1), [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(Q, [0.05, 0.02])


class TestViewsFile(unittest.TestCase):
    def test_shipped_views_yaml_has_no_active_views(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "views.yaml")
        self.assertEqual(bl.load_views(path, TICKERS), [])

    def test_unknown_ticker_raises(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                         encoding="utf-8") as f:
            f.write("views:\n  - name: bad\n    assets: [ZZZ]\n"
                    "    excess_return: 0.05\n    confidence: 0.5\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                bl.load_views(path, TICKERS)
        finally:
            os.unlink(path)


class TestMarketWeights(unittest.TestCase):
    def test_inverse_vol_fallback_offline(self):
        rng = np.random.default_rng(0)
        rets = pd.DataFrame(rng.normal(0, [0.01, 0.02, 0.04], (500, 3)),
                            columns=TICKERS)
        w, source = bl.market_weights(TICKERS, rets, offline=True)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertIn("inverse volatility", source)
        # lowest vol asset gets the biggest weight
        self.assertEqual(w.idxmax(), "AAA")

    def test_implied_risk_aversion_sane(self):
        rng = np.random.default_rng(1)
        bench = pd.Series(rng.normal(0.0004, 0.01, 1260))
        delta = bl.implied_risk_aversion(bench, rf=0.03)
        self.assertGreaterEqual(delta, 1.0)
        self.assertLessEqual(delta, 10.0)
        # degenerate market (negative mean) falls back to the default
        bad = pd.Series(rng.normal(-0.001, 0.01, 1260))
        self.assertEqual(bl.implied_risk_aversion(bad, rf=0.03), bl.DEFAULT_DELTA)


if __name__ == "__main__":
    unittest.main()
