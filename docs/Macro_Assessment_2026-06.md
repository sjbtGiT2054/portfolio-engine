# Monthly Macro Assessment — June 2026

Complete before each rebalance. One page maximum.

*First completed copy, drafted from the 2026-06-11 live engine run. The data points below come straight from the run; the judgement calls are drafts — overwrite them with your own view before the next rebalance. The discipline is the point.*

## 1. Regime read

- Growth: [expanding / slowing — CONFIRM] because the 3 month T bill has come down to 3.63% while the 10 year sits at 4.53%, an upward sloping curve consistent with a cutting cycle that markets read as mid cycle easing rather than recession response.
- Inflation and rates: short rates well below the 4.5% the config assumed at setup, so the hurdle every basket must clear to earn its Sharpe has fallen. Risk free source is live ^IRX, no longer the manual fallback.
- Risk sentiment: constructive. Recent 60 day average pairwise correlation is 0.41 vs 0.56 full sample, so the universe is trading on idiosyncratic stories rather than one macro factor. No correlation warning triggered.

## 2. Theme review (one block per basket)

### ai_energy_demand
- Thesis still valid? Strengthening
- Evidence this month: the optimiser pinned the basket at its 25% maximum. Five year alphas vs CAPM: NLR +8.7%, URA +9.8%, XLU +2.0% — nuclear and uranium earned well above what their betas predict.
- Action: range stays 10–25%. Widening the max is the live decision to weigh, since the constraint is binding, but do it on thesis evidence, not on backward looking alpha alone.

### ai_compute
- Thesis still valid? Yes
- Evidence: also pinned at its 25% maximum. SMH alpha +17.4% (realised 39.7% annualised) dominates; XLK +5.4%, QQQ +1.6%. Concentration risk: the basket is one semiconductor cycle.
- Action: range stays 5–25%. SMH already at the 15% single ETF cap, so the cap is the real guardrail here.

### energy_infrastructure
- Thesis still valid? Yes
- Evidence: pinned at its 15% maximum. XLE alpha +11.9% with only 0.64 beta — traditional energy has been the cheapest diversifying return in the universe. XOP +7.8%, PAVE +3.4%.
- Action: range stays 0–15%.

### core_market
- Any reason to change the 30–70% anchor? No. Sits at 35%, near the floor, because every theme basket is maxed out. Inside core, the optimiser dropped IWM (alpha -6.3%) and ICLN (-8.6%) entirely and kept VOO, XLI, XLV. The anchor is doing its job: forcing 35% of the book into diversified beta the themes would otherwise crowd out.

## 3. Config changes made

| Setting | Old | New | Why |
|---|---|---|---|
| none | — | — | First live baseline run; observe one full cycle before touching ranges |

## 4. What would make me wrong this month

- All three theme baskets are at their maximums on the strength of historical alphas. If AI capex guidance rolls over, ai_compute and ai_energy_demand fall together (SMH, NLR, URA all load on the same story) and the 0.97 portfolio beta gives no shelter.
- Energy alpha came partly from the 2021–22 supply shock; that regime may not repeat.
- Rate cuts already delivered are in the price; if inflation re-accelerates and cuts reverse, long duration growth exposure (XLK, SMH) underperforms.
