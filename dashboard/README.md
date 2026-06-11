# Dashboard (Phase 4)

Local Streamlit front end for the portfolio engine. It reads what the engine
writes to `outputs/dashboard/` plus `config.yaml` and `current_holdings.csv`.
It never runs the optimisation pipeline itself, so nothing here can break
the engine.

## Run it

From the project root, with the venv active:

```
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

A browser tab opens at `http://localhost:8501`.

## Tabs

- **Overview** — headline metrics, weights coloured by macro basket, basket allocation vs ranges, risk limit checks.
- **Efficient frontier** — interactive Monte Carlo cloud, constrained frontier, Capital Market Line. Zoom, pan, hover.
- **CAPM / SML** — Security Market Line with per ETF alphas and tooltips.
- **Risk** — beta stress tests, correlation monitor, interactive correlation heatmap.
- **Backtest** — out of sample equity curves vs benchmark, drawdown, weights through time.
- **Basket editor** — sliders for basket min/max ranges and the single ETF cap. Saving rewrites `config.yaml` in place (comments preserved). Infeasible combinations are blocked before saving.
- **Holdings & rebalance** — edit `current_holdings.csv` in the browser and get the BUY/SELL trade list vs the latest optimal weights.
- **Tear sheets** — render any dated markdown report, plus the static PNG chart gallery.

The sidebar can re-run `python main.py` (with offline and skip backtest
toggles) and shows the run log.

Personal learning and research tool only. Not investment advice.
