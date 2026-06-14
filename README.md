# Portfolio Engine

**Author: Samuel Thompson.** Designed, specified and directed by the author; built with AI assisted development.

A long only, equity ETF, macro constrained portfolio management adviser. You own the macro views (defined as ETF baskets with min/max limits in `config.yaml`); the engine owns construction, maximising the Sharpe ratio via Monte Carlo simulation and exact SLSQP optimisation, grounded in Modern Portfolio Theory and CAPM (efficient frontier, Capital Market Line, Security Market Line).

Personal learning and research tool only. Not investment advice — see [DISCLAIMER.md](DISCLAIMER.md). Code released under the [MIT License](LICENSE).

## Why I built this

*(Author's note, to be filled in: what motivated the project, what it demonstrates about portfolio construction discipline, and what it deliberately is not.)*

## Setup (once, in VS Code)

1. Install Python 3.10+ from python.org if not already installed (tick "Add to PATH").
2. Open this `portfolio_engine` folder in VS Code, open a terminal (Ctrl+`) and run:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python main.py                  full run with live Yahoo Finance data
python main.py --no-backtest    faster, skips the walk forward backtest
python main.py --offline        synthetic data demo, no internet needed
```

Outputs land in `outputs/`:

- `reports/tear_sheet_YYYY-MM-DD.md` — the dated written report
- `charts/` — efficient frontier + CML, SML, weights, basket allocation, backtest, correlation matrix
- `optimal_weights.csv` and `backtest_weights_history.csv`

## Monthly workflow

1. Review your macro views and update basket ranges in `config.yaml` (the monthly macro assessment, template in `docs/`).
2. Run `python main.py`.
3. Read the tear sheet, log the decision in `docs/Rebalance_Log.md`.
4. Optional: maintain `current_holdings.csv` (columns `ticker,weight`) in this folder to get explicit BUY/SELL trade recommendations each run.

## How it works

- **Data:** adjusted daily closes from Yahoo Finance (free, no API key), cached in `data/prices.csv` so reruns work offline. Risk free rate fetched live from the 3 month T bill (^IRX), FRED fallback, manual fallback in config.
- **Estimation:** annualised mean returns and a shrinkage adjusted covariance matrix by default. Optional Phase 2 modes in `config.yaml`: `expected_returns: black_litterman` (market implied equilibrium returns from ETF AUM weights, tilted by your views in `views.yaml`; with no views it is pure equilibrium) and `covariance_estimator: ledoit_wolf` (analytically optimal shrinkage).
- **Robustness:** every run bootstraps the returns and re-optimises to show weight stability, and bumps each expected return plus and minus 1% to flag fragile positions. Both appear in the tear sheet and the dashboard's Robustness & BL tab. Disable via `robustness.enabled: false`.
- **Optimisation:** 50,000 Monte Carlo portfolios for intuition and visualisation, then SLSQP solves the exact constrained max Sharpe and traces the constrained efficient frontier. Constraints: fully invested, long only, 15% single ETF cap, every basket inside its min/max range.
- **CAPM:** OLS betas vs SPY, SML chart flags positive/negative alpha ETFs, portfolio beta feeds the stress tests.
- **Backtest:** walk forward, monthly rebalanced, trained only on trailing data (genuinely out of sample), 10 bps transaction costs on turnover, benchmarked against SPY buy and hold.
- **Risk:** limit checks, historical and parametric VaR + CVaR, beta based stress scenarios, max drawdown, and a correlation monitor that warns when recent average pairwise correlation jumps (diversification failing).

## Editing the model

Everything lives in `config.yaml`: universe, baskets and ranges, lookback, simulation count, position cap, costs, risk free mode. No code edits needed for normal use.

## Roadmap (agreed phases)

- Phase 1 (DONE): constrained Sharpe maximiser, charts, backtest, tear sheet
- Phase 2 (machinery DONE): Black Litterman with confidence weighted views plus robustness layer (Ledoit Wolf, bootstrap stability, sensitivity). `views.yaml` ships empty; the owner's macro views are still to be added
- Phase 3: individual equities alongside ETFs, with tighter robustness controls
- Phase 4 (DONE): local Streamlit dashboard in `dashboard/` (interactive frontier, basket sliders, holdings editor, tear sheet viewer)
- Phase 5 (DONE): Portfolio Analyzer dashboard tab — paste any portfolio (any Yahoo tickers, any currency, converted to USD with every conversion documented), compare it with the engine optimal and the frontier, quantify the gap and the historical effect of partial reallocation. Analytics only, no advice
- Phase 6 (DONE): Portfolio Optimiser inside the Analyzer tab — max Sharpe, min volatility, max diversification and equal weight solved over exactly your tickers (long only, fully invested, adjustable position cap), with the universe's own Monte Carlo cloud and frontier, weight change tables, correlation heatmap, sensitivity fragility flags, and a gated walk forward backtest of current weights vs the re-optimised strategy vs SPY. In sample optima are labelled as illustrations; the out of sample backtest is the honest measure
- Phase 7 (DONE): portfolio first public experience — the deployed app lands on "Your Portfolio" (search any listed company or fund by name, enter weights or money amounts, get metric cards with plain English explanations, then optimise), with the author's engine one tab away under "Engine Showcase". Local mode unchanged
- Phase 8 macro views layer (DONE): a "Macro Views" tab turns user adjusted macro paths into Black Litterman views and re-optimises the portfolio. Eight active sliders (CPI, fed funds rate, 10 year yield, 2s10s slope, unemployment, VIX, dollar index, crude oil) over a one to five year horizon, with the rest of an exhaustive macro list shown read only in an expander. The deviation from a naive baseline times each holding's estimated historical sensitivity becomes a return view, fed through `black_litterman.py` and re-optimised, with before versus after weights, stats and the portfolio dot moving on the frontier. Reuses the existing BL and optimiser machinery; output is framed as the portfolio's implied response, never a forecast. Free data only (FRED fredgraph endpoint plus Yahoo), every series cached with daily refresh and graceful degradation. The Phase 8 equity universe migration and basket editor rework remain to be built
- Later: market neutral long short toggle

## Limitations register

- Historical mean returns are noisy forecasts; optimised weights are sensitive to them. Shrinkage, the position cap and basket ranges are the mitigants.
- Equity only universe: high positive correlation, limited diversification in selloffs (by design, as a learning point).
- Free data may have gaps, splits handled by Yahoo adjustment, possible survivorship effects.
- Backtest assumes frictionless monthly execution at close plus a flat cost; no slippage model, no taxes.
- No live execution. Trading 212 has no public API; trades are manual.
- Not FCA authorised activity; for personal use only.
