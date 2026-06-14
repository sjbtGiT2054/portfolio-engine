# Portfolio Engine — context for Claude Code

## What this is

A long only, equity ETF, macro constrained portfolio management adviser built by Samuel Thompson. It is a personal learning vehicle and buy side credibility piece, NOT a trading product and NOT investment advice. Keep that framing in all work.

## Architecture

- `config.yaml` — single source of truth: 18 ETF universe, four macro baskets with min/max ranges, all settings. Config edits should never require code edits.
- `main.py` — orchestrator. Run `python main.py` (live Yahoo data), `--offline` (synthetic demo), `--no-backtest` (faster).
- `engine/data.py` — yfinance download, CSV cache in `data/`, synthetic offline mode.
- `engine/riskfree.py` — live 3 month T bill (^IRX), FRED DTB3 fallback, manual fallback.
- `engine/stats.py` — annualised returns, shrinkage covariance, Sharpe/Sortino/VaR/drawdown.
- `engine/optimiser.py` — 50k Monte Carlo cloud plus exact SLSQP constrained max Sharpe, min vol, efficient frontier. Constraints: sum to 1, long only, 15% single ETF cap, basket ranges.
- `engine/capm.py` — OLS betas vs SPY, SML data, alphas.
- `engine/black_litterman.py` — Phase 2: equilibrium returns reverse optimised from ETF AUM weights (Yahoo totalAssets, inverse vol fallback), views from `views.yaml` with confidences. Zero views = pure equilibrium. Switch: `settings.expected_returns: historical | black_litterman` (default historical).
- `views.yaml` — owner's BL views. Ships with commented examples only; never invent views on the owner's behalf.
- `engine/robustness.py` — Ledoit Wolf covariance (`settings.covariance_estimator: simple | ledoit_wolf`), bootstrap resampled weight stability, plus/minus 1% expected return sensitivity with fragility flags. `robustness:` config section, enabled by default.
- `tests/` — unittest suites for BL maths and robustness. Run: `python -m unittest discover -s tests`.
- `engine/backtest.py` — walk forward, monthly rebalance, trailing window training only (out of sample), 10 bps costs, vs SPY.
- `engine/risk.py` — limit checks, beta stress tests, correlation monitor.
- `engine/charts.py` — six PNGs to `outputs/charts/`.
- `engine/report.py` — dated markdown tear sheet to `outputs/reports/`, plus BUY/SELL recommendations if `current_holdings.csv` (columns ticker,weight) exists.
- `engine/export.py` — machine readable artifacts (results.json, frontier/MC/CAPM/correlation/backtest CSVs) to `outputs/dashboard/` for the dashboard.
- `engine/analyzer.py` — Phase 5 Portfolio Analyzer: arbitrary Yahoo tickers, own cache (`data/analyzer_prices.csv`), explicit USD conversion (GBp pence detected via Yahoo currency metadata, divided by 100, then CURUSD=X), full stat set, frontier gap, blend diagnostics, rule based flags in analytics wording only. Read only: never runs the pipeline, never writes engine outputs.
- `engine/user_optimiser.py` — Phase 6: max Sharpe, min volatility, max diversification (minimises correlation weighted concentration w'Cw) and equal weight over the user's own tickers, reusing optimiser.py SLSQP with empty baskets. 10k Monte Carlo cloud, universe frontier, sensitivity carry over, gated walk forward backtest (skip if >15 tickers or <2y history) reusing backtest.py. Needs 2+ tickers and 1y common history. `optimise(exp_ret=...)` optionally overrides historical means so BL posterior returns can drive re-optimisation.
- `engine/macro.py` — Phase 8 macro data layer. Variable catalogue (exact FRED/Yahoo codes, groups, slider bounds, beginner tooltips), cached fetch to `data/macro/` with daily refresh, graceful degradation to None on any failure (slider greys out, tab never crashes), current value, naive trend baseline path. Free sources only: FRED fredgraph.csv endpoint, Yahoo via yfinance. 8 active v1 sliders, rest display only.
- `engine/macro_sensitivity.py` — Phase 8: OLS of monthly holding returns on monthly macro variable changes (coef + t-stat, low-confidence flag at 10pct), then deviation-from-baseline x sensitivity / horizon = annualised return tilt per holding, blended into absolute Black Litterman views (prior + tilt) fed to black_litterman.posterior_returns. Sensitivities are noisy historical estimates, framed as implied response not forecast.
- `dashboard/app.py` — Streamlit front end (Phase 4, done). Reads `outputs/dashboard/`, `config.yaml`, `current_holdings.csv`; basket sliders rewrite config.yaml via ruamel.yaml (comments preserved); never runs the pipeline itself. Exception (Phases 5/6/8): the dashboard may call `engine/analyzer.py`, `engine/user_optimiser.py`, `engine/macro.py`, `engine/macro_sensitivity.py`, `engine/black_litterman.py` and `engine.report.rebalance_recommendation` directly for read only calculations, and nothing more. Run: `streamlit run dashboard/app.py`. Deps: `dashboard/requirements.txt`.
- `docs/` — Investment Policy Statement, macro assessment template, rebalance log.
- Public deployment: `DEPLOY.md` has the Streamlit Community Cloud steps. `PUBLIC_MODE` secret/env var switches the dashboard to public mode: analyzer is session only (no current_holdings.csv reads or writes), Basket editor and Holdings tabs hidden, engine rerun hidden, standing "not investment advice" banner. Public data source = committed `outputs/dashboard/` + `outputs/charts/`, refreshed by running `python main.py` locally and pushing. `.gitignore` keeps `data/`, `current_holdings.csv` and `outputs/reports/` local.

## Hard rules

- Long only, fully invested, no leverage in phase 1. Do not add shorting without being asked.
- Free public data only (Yahoo Finance, FRED). Never add paid APIs or API keys.
- No brokerage integration. Trading 212 has no public API; holdings are manual CSV.
- Nothing that constitutes regulated investment advice or auto adjusts anyone else's portfolio.
- Windows 11 machine, venv at `.venv`, Python 3.12.

## Agreed roadmap

- Phase 2 (machinery DONE, views pending owner input): Black Litterman as an optional mode alongside basket constraints, plus robustness layer (Ledoit Wolf, bootstrap stability, sensitivity). `views.yaml` deliberately empty until the owner adds his own macro views; do not fill it for him.
- Phase 3: individual equities in the universe with tighter robustness controls.
- Phase 4 (DONE): local Streamlit dashboard in `dashboard/` reading `outputs/dashboard/` and `current_holdings.csv` — interactive frontier, basket sliders that rewrite config.yaml, tear sheet viewer. Never let dashboard work break the engine.
- Phase 5 (DONE): Portfolio Analyzer — arbitrary holdings vs engine optimal, frontier gap, blend diagnostics, dashboard tab.
- Phase 6 (DONE): user universe optimiser inside the Analyzer tab — four alternative weightings, own frontier with MC cloud, weight deltas (increase/decrease wording, never BUY/SELL in public), correlation heatmap, sensitivity flags, gated walk forward backtest. Fully session only in public mode.
- Phase 7 (DONE): public mode is portfolio first — landing tab "Your Portfolio" (Yahoo symbol search via `analyzer.search_symbols`, weight % or amount input via `analyzer.holdings_from_input`, metric cards with plain English help tooltips, optimise results behind expanders), engine views nested under one "Engine Showcase" tab. Local mode tabs unchanged.
- Phase 8 macro views layer (DONE): "Macro Views" tab (both modes, session only in public) is the front end for the Phase 2 BL machinery. User bends naive macro paths (8 active sliders: CPI, fed funds, 10y, 2s10s, unemployment, VIX, DXY, WTI; rest display only in an expander); deviation x estimated historical sensitivity becomes a per holding return view fed through `black_litterman.posterior_returns`, then re-optimised via `user_optimiser.optimise(exp_ret=...)`. Reuses BL and the optimiser, no new maths. Needs an analysed portfolio first. `views.yaml` untouched (that is the personal local BL file; macro tab views are session only). Note: FRED is unreachable from this dev sandbox (egress block), so the 5 FRED sliders grey out here; Yahoo sliders (VIX/DXY/WTI) and the full chain verified live. FRED works on normal hosts e.g. Streamlit Cloud.
- Phase 8 remainder (per README, NOT yet built): equity universe migration to 28 individual names and the basket editor rework. The macro views task did not touch the universe or basket editor.
- Later: market neutral long short toggle.

## Local context

Owner specific working preferences and holding specific notes live in `CLAUDE.local.md`, which is gitignored and exists only on the owner's machine. Read it at the start of every local session. The standing engineering rule that applies everywhere: test with `python main.py --offline` before declaring anything done.
