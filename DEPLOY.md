# Deploying the dashboard to Streamlit Community Cloud

The public app is the same `dashboard/app.py`, switched into public mode
by a `PUBLIC_MODE` secret. Public mode never touches local files: the
Portfolio Analyzer is session only (seeded with a generic 60% VOO / 40%
QQQ example), the Basket editor and Holdings & rebalance tabs are hidden,
the engine rerun button is hidden, and a standing banner reads
"Educational analytics tool. Not investment advice."

## How the public app gets its data

It reads the committed `outputs/dashboard/` artifacts (results.json plus
the frontier, Monte Carlo, CAPM, correlation, backtest, BL and
robustness CSVs) and the `outputs/charts/` PNGs. To refresh the public
numbers:

```
python main.py        # local live run regenerates outputs/
git add outputs/dashboard outputs/charts
git commit -m "Refresh engine outputs"
git push
```

Streamlit Cloud redeploys automatically on push. `data/`,
`current_holdings.csv` and `outputs/reports/` are gitignored and never
leave this machine.

## One time setup

1. Create the GitHub repository (public) and push this folder:

```
git init
git add .
git commit -m "Portfolio engine + dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/portfolio-engine.git
git push -u origin main
```

2. Check the push respected `.gitignore`: on github.com confirm there is
   NO `current_holdings.csv`, NO `data/` folder and NO `outputs/reports/`.

3. Go to https://share.streamlit.io, sign in with GitHub, click
   "Create app".
   - Repository: `<your-username>/portfolio-engine`
   - Branch: `main`
   - Main file path: `dashboard/app.py`

4. Before the first deploy, open Advanced settings (or later: app menu,
   Settings, Secrets) and add exactly:

```toml
PUBLIC_MODE = "true"
```

5. Deploy. The app gets a shareable URL like
   `https://<app-name>.streamlit.app`.

## Testing public mode locally

```
$env:PUBLIC_MODE = "true"; streamlit run dashboard/app.py
```

Unset with `Remove-Item Env:PUBLIC_MODE` to return to the full local app.
Local mode is the default everywhere; nothing about the local workflow
changes.

## Notes

- Python version: Streamlit Cloud defaults to a recent 3.x; the code
  targets 3.12. You can pin it in the app's Advanced settings.
- Dependencies install from the root `requirements.txt`, which includes
  streamlit and ruamel.yaml for this reason.
- The Portfolio Analyzer fetches Yahoo prices at request time on the
  public app; heavy traffic could hit Yahoo rate limits. It is a demo,
  not a service.
