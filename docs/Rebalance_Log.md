# Rebalance Log

One entry per monthly run. The discipline of writing the rationale is the point.

---

## 2026-06-11 — first live run

- Tear sheet: `outputs/reports/tear_sheet_2026-06-11.md`
- Sharpe / ret / vol: 1.00 / 20.7% / 17.1% (in sample; walk forward out of sample Sharpe 0.74 vs SPY 0.81)
- Limit checks: ALL PASS. All three theme baskets binding at their maximums, core_market at 35%.
- Config changes since last run: none. Phase 4 dashboard added (`streamlit run dashboard/app.py`), engine untouched.
- Trades taken (if any) vs recommendation, and why I deviated: none. No `current_holdings.csv` yet, so the target weights are the full buy list. Next step is entering actual Trading 212 weights via the dashboard Holdings tab before acting on anything.
- One lesson: the synthetic demo gave beta 0.41; real data gives 0.97. Real equity ETFs are one highly correlated asset class, so the "diversified" optimal portfolio is still essentially full market risk — the CAPM systematic vs idiosyncratic split in practice. Also note the out of sample Sharpe (0.74) trails both the in sample figure (1.00) and SPY (0.81): historical mean returns are noisy forecasts, and the optimiser overfits them. That gap, and explaining it, is the honest headline number, not the 1.00.

---

## [DATE]

- Tear sheet: `outputs/reports/tear_sheet_YYYY-MM-DD.md`
- Sharpe / ret / vol: ... / ... / ...
- Limit checks: ALL PASS / details
- Config changes since last run:
- Trades taken (if any) vs recommendation, and why I deviated:
- One lesson:

---
