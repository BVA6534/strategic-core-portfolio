# Strategic Core 45/50/5

Interactive backtest for our 2270 Financial Modeling (Nova SBE) group assignment.

**Live app:** https://strategic-core-portfolio.streamlit.app
**Repository:** https://github.com/BVA6534/strategic-core-portfolio
**Group:** `[Moodle group name]`, `[members]`

## Strategy
- 45% US equity (SPY), 50% US aggregate bonds (AGG), 5% gold (GLD)
- Quarterly rebalancing to target weights, 10 bps one-way transaction cost
- Benchmarks: 60/40 SPY/AGG (quarterly, frictionless) and S&P 500 ETF (buy and hold)
- Data: Yahoo Finance via `yfinance`, monthly, Jan 2005 to Dec 2025; risk-free rate: 3M T-bill (`^IRX`)

In the app the user can change the target weights, the rebalancing rule (monthly, quarterly, annual, threshold band, never), the band width, transaction costs and the sample period, and compare the result with the prospectus default and both benchmarks.

## Files
| File | Purpose |
|---|---|
| `portfolio_core.py` | Calculation engine (data, backtest, performance measures). The notebook submitted on Moodle writes an identical copy of this file, so app and notebook give the same numbers. |
| `app.py` | Streamlit user interface |
| `requirements.txt` | Python dependencies |
| `data/monthly_returns_snapshot.csv` | Fallback copy of the monthly return panel (same data as the notebook). Used only if Yahoo Finance blocks the Streamlit server; the app shows a notice when this happens. |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Fork or clone this repository.
2. Go to share.streamlit.io, sign in with GitHub, click **Create app**.
3. Select the repository, branch `main`, main file `app.py`, then **Deploy**.

## Disclaimer
Educational project. Past performance is not a reliable indicator of future results.
