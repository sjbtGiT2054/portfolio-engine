"""Unit tests for the Phase 7 holdings input conversion (pure function)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.analyzer import holdings_from_input


def frame(rows):
    return pd.DataFrame(rows, columns=["ticker", "weight_pct", "amount"])


class TestHoldingsInput(unittest.TestCase):
    def test_amounts_only(self):
        w, notes = holdings_from_input(frame([
            ("VOO", None, 6000), ("QQQ", None, 4000)]))
        np.testing.assert_allclose(w.values, [0.6, 0.4])
        self.assertEqual(notes, [])

    def test_weights_only_summing(self):
        w, notes = holdings_from_input(frame([
            ("VOO", 60, None), ("QQQ", 40, None)]))
        np.testing.assert_allclose(w.values, [0.6, 0.4])
        self.assertEqual(notes, [])

    def test_weights_not_summing_normalised_with_note(self):
        w, notes = holdings_from_input(frame([
            ("VOO", 50, None), ("QQQ", 30, None)]))
        np.testing.assert_allclose(w.values, [0.625, 0.375])
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertTrue(any("normalised" in n for n in notes))

    def test_both_given_amounts_win_with_note(self):
        w, notes = holdings_from_input(frame([
            ("VOO", 90, 1000), ("QQQ", 10, 3000)]))
        np.testing.assert_allclose(w.values, [0.25, 0.75])
        self.assertTrue(any("amounts win" in n for n in notes))

    def test_true_mix_rejected(self):
        with self.assertRaises(ValueError) as cm:
            holdings_from_input(frame([
                ("VOO", None, 1000), ("QQQ", 40, None)]))
        self.assertIn("Mixed input", str(cm.exception))

    def test_missing_values_rejected(self):
        with self.assertRaises(ValueError):
            holdings_from_input(frame([("VOO", 60, None), ("QQQ", None, None)]))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            holdings_from_input(frame([]))

    def test_duplicate_tickers_rejected(self):
        with self.assertRaises(ValueError) as cm:
            holdings_from_input(frame([("VOO", 50, None), ("voo", 50, None)]))
        self.assertIn("Duplicate", str(cm.exception))

    def test_ticker_case_and_whitespace_cleaned(self):
        w, _ = holdings_from_input(frame([(" spxp.l ", 100, None)]))
        self.assertEqual(list(w.index), ["SPXP.L"])
        self.assertAlmostEqual(float(w.iloc[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
