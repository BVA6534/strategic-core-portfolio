"""
portfolio_core.py
-----------------
Shared calculation engine for the "Strategic Core 45/50/5" portfolio
(2270 Financial Modeling, Nova SBE, group assignment).

The SAME file is used by the Jupyter notebook (written to disk with %%writefile)
and by the Streamlit app (imported from the GitHub repository), so every number
in the prospectus, the notebook and the app comes from identical code.

Conventions (as in the course, Lecture 3):
- Monthly data, prices adjusted for splits and distributions (auto_adjust=True).
- Simple returns are used for portfolio aggregation, because a portfolio's
  return is the weighted sum of its assets' simple returns (log returns do not
  add up across assets).
- Annualised mean = 12 x monthly mean; annualised volatility = sqrt(12) x monthly st. dev.
- Sharpe ratio = annualised mean excess return / annualised st. dev. of excess return.
- Alpha and beta: OLS regression of the portfolio's excess return on the market's
  (S&P 500 ETF) excess return. Alpha is annualised (x12); beta is NOT annualised.
- Information ratio = annualised mean active return / annualised tracking error,
  active return = portfolio return - benchmark (60/40) return.
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Fixed assumptions of the prospectus
# --------------------------------------------------------------------------
ASSETS = {
    "SPY": "SPDR S&P 500 ETF (US large-cap equity)",
    "AGG": "iShares Core US Aggregate Bond ETF (US investment-grade bonds)",
    "GLD": "SPDR Gold Shares (physical gold)",
}
RF_TICKER = "^IRX"  # 13-week (3M) US Treasury bill yield, annualised, in percent

STRATEGY_WEIGHTS = {"SPY": 0.45, "AGG": 0.50, "GLD": 0.05}
BENCHMARK_6040_WEIGHTS = {"SPY": 0.60, "AGG": 0.40, "GLD": 0.00}
MARKET_WEIGHTS = {"SPY": 1.00, "AGG": 0.00, "GLD": 0.00}

# First price observation: end of Dec 2004 (GLD listed Nov 2004).
# First return: Jan 2005. Last return: Dec 2025. yfinance's `end` is exclusive.
START_DATE = "2004-12-01"
END_DATE = "2026-01-01"

DEFAULT_REBALANCE = "Q"      # quarterly, at the end of Mar, Jun, Sep, Dec
DEFAULT_THRESHOLD = 0.05     # +/- 5 percentage points (threshold variant)
DEFAULT_TC_BPS = 10.0        # one-way cost in basis points of traded value
BENCHMARK_REBALANCE = "Q"    # 60/40 benchmark: quarterly, frictionless index
PERIODS_PER_YEAR = 12

REBALANCE_LABELS = {
    "M": "Monthly",
    "Q": "Quarterly",
    "A": "Annual",
    "threshold": "Threshold band",
    "none": "Never (buy and hold)",
}


# --------------------------------------------------------------------------
# 2. Data
# --------------------------------------------------------------------------
def _month_end(df):
    """Resample daily data to the last available observation of each month."""
    try:
        return df.resample("ME").last()
    except ValueError:  # pandas < 2.2
        return df.resample("M").last()


def download_close(tickers, start, end, retries=4, wait=15):
    """
    Daily adjusted closes from Yahoo Finance via the yfinance API.
    Yahoo occasionally rate-limits requests; yfinance then returns empty
    columns instead of raising, so the result is checked and the download
    retried a few times before giving up.
    """
    import time
    import yfinance as yf

    tickers = list(tickers)
    problem = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(tickers, start=start, end=end, interval="1d",
                              auto_adjust=True, progress=False, threads=False)
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            close = close.reindex(columns=tickers).copy()
            empty = close.columns[close.isna().all()].tolist()
            if len(close) > 0 and not empty:
                if getattr(close.index, "tz", None) is not None:
                    close.index = close.index.tz_localize(None)
                return close
            problem = f"no data for {empty or tickers}"
        except Exception as exc:  # network error, rate limit
            problem = repr(exc)
        if attempt < retries:
            print(f"Download attempt {attempt} failed ({problem}). "
                  f"Retrying in {wait} seconds...")
            time.sleep(wait)
    raise RuntimeError(f"Yahoo Finance download failed after {retries} attempts "
                       f"({problem}). Please wait a few minutes and run the cell again.")


def download_prices(start=START_DATE, end=END_DATE, retries=4, wait=15):
    """
    Download daily adjusted closes from Yahoo Finance (yfinance API) and convert
    them to month-end observations.

    Returns
    -------
    prices : DataFrame, month-end adjusted prices of SPY, AGG, GLD
    irx    : Series, month-end 3M T-bill yield (annualised, percent)
    """
    close = download_close(list(ASSETS) + [RF_TICKER], start, end, retries, wait)
    monthly = _month_end(close)
    prices = monthly[list(ASSETS)]
    irx = monthly[RF_TICKER].rename("IRX")
    return prices, irx


def build_returns(prices, irx):
    """
    Monthly simple returns of the assets plus the monthly risk-free rate.

    The risk-free rate for month t is the T-bill yield observed at the END of
    month t-1, divided by 12. It is therefore known when month t starts
    (no look-ahead).
    """
    asset_ret = prices.pct_change()
    rf = (irx.shift(1) / 100.0) / PERIODS_PER_YEAR
    data = pd.concat([asset_ret, rf.rename("RF")], axis=1).iloc[1:]
    if data.isna().any().any():
        missing = data.isna().sum()
        raise ValueError(f"Missing values after alignment:\n{missing}")
    return data


def load_data(start=START_DATE, end=END_DATE, retries=4, wait=15):
    """Convenience wrapper: download and build the monthly return panel."""
    prices, irx = download_prices(start, end, retries, wait)
    return build_returns(prices, irx)


# Snapshot of the monthly return panel. Used ONLY by the Streamlit app as a
# fallback when Yahoo Finance blocks the cloud server. The notebook always
# downloads through the API.
SNAPSHOT_PATH = "data/monthly_returns_snapshot.csv"


def save_snapshot(data, path=SNAPSHOT_PATH):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, float_format="%.12g")
    return path


def load_snapshot(path=SNAPSHOT_PATH):
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    data.index.name = None
    return data


# --------------------------------------------------------------------------
# 2b. Pre-sample evidence (used only to MOTIVATE the weights)
# --------------------------------------------------------------------------
# The fixed weights are justified with data that ends before the backtest
# starts (Dec 2004), so the allocation rule is not fitted to the test period.
PRESAMPLE_START = "1994-12-01"   # first price: Dec 1994 -> returns from Jan 1995
PRESAMPLE_END = "2005-01-01"     # last return: Dec 2004 (= backtest base date)
PRESAMPLE_TICKERS = {
    "SPY": "S&P 500 ETF (equity)",
    "VBMFX": "Vanguard Total Bond Market Index Fund (proxy for AGG, which starts in 2003)",
    "GC=F": "COMEX gold futures (proxy for GLD; Yahoo history starts in 2000)",
}


def load_presample(start=PRESAMPLE_START, end=PRESAMPLE_END):
    """Monthly returns before the backtest period (pre-sample)."""
    close = download_close(list(PRESAMPLE_TICKERS) + [RF_TICKER], start, end)
    monthly = _month_end(close)
    rets = monthly[list(PRESAMPLE_TICKERS)].pct_change()
    rets["RF"] = (monthly[RF_TICKER].shift(1) / 100.0) / PERIODS_PER_YEAR
    return rets.iloc[1:]


def equity_bond_frontier(equity, bonds, rf, step=0.05):
    """Risk and return of fixed equity/bond mixes (monthly rebalanced)."""
    rows = []
    for w in np.round(np.arange(0, 1 + 1e-9, step), 2):
        r = w * equity + (1 - w) * bonds
        ex = r - rf
        rows.append({
            "Equity share": w,
            "Ann. average return": PERIODS_PER_YEAR * r.mean(),
            "Ann. volatility": np.sqrt(PERIODS_PER_YEAR) * r.std(),
            "Sharpe ratio": (PERIODS_PER_YEAR * ex.mean())
                            / (np.sqrt(PERIODS_PER_YEAR) * ex.std()),
            "Max drawdown": drawdown(r).min(),
        })
    return pd.DataFrame(rows).set_index("Equity share")


# --------------------------------------------------------------------------
# 3. Backtest engine
# --------------------------------------------------------------------------
def _calendar_rebalance(date, freq):
    if freq == "M":
        return True
    if freq == "Q":
        return date.month in (3, 6, 9, 12)
    if freq == "A":
        return date.month == 12
    return False


def backtest(asset_returns, target_weights, rebalance=DEFAULT_REBALANCE,
             threshold=DEFAULT_THRESHOLD, tc_bps=DEFAULT_TC_BPS):
    """
    Simulate a fixed-target-weight portfolio with drifting weights between
    rebalancing dates.

    Timing (only information available at the time is used):
      1. The portfolio enters month t with post-trade weights w_{t-1}.
      2. Gross return R_t = sum_i w_{i,t-1} * r_{i,t}.
      3. Weights drift: w_drift = w_{t-1} * (1 + r_t) / (1 + R_t).
      4. At the end of month t the rule is checked with data known at t:
           - calendar rules: is t a rebalancing month?
           - threshold rule: is any |w_drift - w_target| > threshold?
      5. If trading: turnover = sum_i |w_target - w_drift| (traded value as a
         share of the portfolio), cost = turnover * tc_bps / 10,000, and the
         net return is (1 + R_t)(1 - cost) - 1. New weights = target.

    The initial purchase is not charged (identical one-off cost for every
    portfolio). ETF expense ratios are already reflected in adjusted prices.

    Returns
    -------
    result  : DataFrame with gross, net, turnover, cost, rebalanced
    weights : DataFrame of pre-trade (drifted) month-end weights
    """
    cols = list(asset_returns.columns)
    w_target = pd.Series(target_weights, dtype=float).reindex(cols).fillna(0.0)
    if abs(w_target.sum() - 1.0) > 1e-8:
        raise ValueError("Target weights must sum to 1.")
    if (w_target < 0).any():
        raise ValueError("Long-only portfolio: weights must be non-negative.")

    w = w_target.copy()
    records, w_hist = [], []
    for date, r in asset_returns.iterrows():
        gross = float((w * r).sum())
        w_drift = w * (1.0 + r) / (1.0 + gross)

        if rebalance == "threshold":
            trade = bool((w_drift - w_target).abs().max() > threshold)
        elif rebalance == "none":
            trade = False
        else:
            trade = _calendar_rebalance(date, rebalance)

        turnover = float((w_target - w_drift).abs().sum()) if trade else 0.0
        cost = turnover * tc_bps / 10_000.0
        net = (1.0 + gross) * (1.0 - cost) - 1.0

        records.append({"date": date, "gross": gross, "net": net,
                        "turnover": turnover, "cost": cost,
                        "rebalanced": trade})
        w_hist.append(w_drift.rename(date))
        w = w_target.copy() if trade else w_drift

    result = pd.DataFrame(records).set_index("date")
    weights = pd.DataFrame(w_hist)
    return result, weights


# --------------------------------------------------------------------------
# 4. Performance measures
# --------------------------------------------------------------------------
def ols(y, x):
    """OLS of y on a constant and x. Returns alpha, beta, t(alpha), residuals."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    if s2 < 1e-14:  # perfect fit (e.g. market on itself): alpha = 0, t undefined
        return 0.0, coef[1], np.nan, resid
    return coef[0], coef[1], coef[0] / se[0], resid


def cumulative(ret):
    """Growth of 1, starting at 1 on the month-end before the first return."""
    wealth = (1.0 + ret).cumprod()
    base_date = ret.index[0] - pd.offsets.MonthEnd(1)
    return pd.concat([pd.Series([1.0], index=[base_date]), wealth])


def drawdown(ret):
    """Percentage drawdown from the high-water mark (negative numbers)."""
    wealth = cumulative(ret)
    return wealth / wealth.cummax() - 1.0


def performance_summary(ret, rf, market, benchmark, bench_label="60/40"):
    """
    Performance measures of one monthly return series.

    ret       : portfolio returns (net of costs)
    rf        : monthly risk-free rate
    market    : market returns (S&P 500 ETF) for alpha/beta
    benchmark : benchmark returns (default 60/40) for tracking error / IR
    """
    k = PERIODS_PER_YEAR
    ex = ret - rf
    mex = market - rf
    alpha_m, beta, t_alpha, _ = ols(ex, mex)
    active = ret - benchmark
    te = np.sqrt(k) * active.std()
    ir = (k * active.mean()) / te if te > 1e-12 else np.nan
    n_years = len(ret) / k
    return {
        "Ann. average return": k * ret.mean(),
        "Ann. volatility": np.sqrt(k) * ret.std(),
        "Sharpe ratio": (k * ex.mean()) / (np.sqrt(k) * ex.std()),
        "Beta (vs S&P 500)": beta,
        "Alpha p.a. (vs S&P 500)": k * alpha_m,
        "Alpha t-stat": t_alpha,
        f"Tracking error p.a. (vs {bench_label})": te,
        f"Information ratio (vs {bench_label})": ir,
        "CAGR": (1.0 + ret).prod() ** (1.0 / n_years) - 1.0,
        "Max drawdown": drawdown(ret).min(),
    }


PCT_PREFIXES = ("Ann. average return", "Ann. volatility", "Alpha p.a.",
                "Tracking error", "CAGR", "Max drawdown", "Avg. annual turnover",
                "Avg. annual cost drag", "Cumulative return")


def is_pct(row):
    """True if a summary row is a percentage (for formatting)."""
    return str(row).startswith(PCT_PREFIXES)


def run_all(data, weights=None, rebalance=DEFAULT_REBALANCE,
            threshold=DEFAULT_THRESHOLD, tc_bps=DEFAULT_TC_BPS):
    """
    Run the strategy and both benchmarks on the same sample and return
    the return series and one summary table.

    Benchmarks:
      - 60/40 SPY/AGG, rebalanced quarterly, frictionless (an index)
      - S&P 500 ETF, buy and hold
    """
    weights = STRATEGY_WEIGHTS if weights is None else weights
    asset_ret = data[list(ASSETS)]
    rf = data["RF"]

    strat, strat_w = backtest(asset_ret, weights, rebalance, threshold, tc_bps)
    b6040, _ = backtest(asset_ret, BENCHMARK_6040_WEIGHTS,
                        BENCHMARK_REBALANCE, tc_bps=0.0)
    mkt, _ = backtest(asset_ret, MARKET_WEIGHTS, "none", tc_bps=0.0)

    returns = pd.DataFrame({
        "Strategy": strat["net"],
        "60/40 benchmark": b6040["net"],
        "S&P 500": mkt["net"],
    })

    table = {}
    for name, series, bt in [("Strategy", returns["Strategy"], strat),
                             ("60/40 benchmark", returns["60/40 benchmark"], b6040),
                             ("S&P 500", returns["S&P 500"], mkt)]:
        stats = performance_summary(series, rf, returns["S&P 500"],
                                    returns["60/40 benchmark"])
        n_years = len(series) / PERIODS_PER_YEAR
        stats["Avg. annual turnover"] = bt["turnover"].sum() / n_years
        stats["Avg. annual cost drag"] = bt["cost"].sum() / n_years
        stats["Rebalancing trades"] = int(bt["rebalanced"].sum())
        table[name] = stats

    summary = pd.DataFrame(table)
    summary.loc["Information ratio (vs 60/40)", "60/40 benchmark"] = np.nan
    summary.loc["Tracking error p.a. (vs 60/40)", "60/40 benchmark"] = np.nan
    return {"returns": returns, "rf": rf, "summary": summary,
            "strategy_detail": strat, "strategy_weights": strat_w}


def format_summary(summary):
    """Human-readable version of the summary table (strings)."""
    out = summary.copy().astype(object)
    for row in summary.index:
        for col in summary.columns:
            v = summary.loc[row, col]
            if pd.isna(v):
                out.loc[row, col] = "n/a"
            elif is_pct(row):
                out.loc[row, col] = f"{100 * v:.2f}%"
            elif row == "Rebalancing trades":
                out.loc[row, col] = f"{int(v)}"
            else:
                out.loc[row, col] = f"{v:.2f}"
    return out


# --------------------------------------------------------------------------
# 5. Additional analyses for the prospectus
# --------------------------------------------------------------------------
NO_GOLD_WEIGHTS = {"SPY": 0.45, "AGG": 0.55, "GLD": 0.00}


def gold_sleeve_comparison(data, rebalance=DEFAULT_REBALANCE,
                           threshold=DEFAULT_THRESHOLD, tc_bps=DEFAULT_TC_BPS):
    """
    Isolate the contribution of the 5% gold sleeve: the strategy against the
    same portfolio without gold (45% SPY / 55% AGG), identical rebalancing
    and costs. Information ratio and tracking error are measured against
    the no-gold portfolio.
    """
    asset_ret = data[list(ASSETS)]
    rf = data["RF"]
    strat, _ = backtest(asset_ret, STRATEGY_WEIGHTS, rebalance, threshold, tc_bps)
    nogold, _ = backtest(asset_ret, NO_GOLD_WEIGHTS, rebalance, threshold, tc_bps)
    mkt, _ = backtest(asset_ret, MARKET_WEIGHTS, "none", tc_bps=0.0)
    returns = pd.DataFrame({"Strategy 45/50/5": strat["net"],
                            "45/55 without gold": nogold["net"]})
    table = pd.DataFrame({
        c: performance_summary(returns[c], rf, mkt["net"],
                               returns["45/55 without gold"],
                               bench_label="45/55 without gold")
        for c in returns.columns})
    table.loc[[r for r in table.index if "45/55" in r], "45/55 without gold"] = np.nan
    return returns, table


def subperiod_table(data, periods, rows=None, **kwargs):
    """Key measures of strategy and benchmarks in several sub-periods."""
    rows = rows or ["Ann. average return", "Ann. volatility", "Sharpe ratio",
                    "Beta (vs S&P 500)", "Alpha p.a. (vs S&P 500)",
                    "Information ratio (vs 60/40)", "Max drawdown"]
    out = {}
    for label, (a, b) in periods.items():
        summ = run_all(data.loc[a:b], **kwargs)["summary"].loc[rows]
        for col in summ.columns:
            out[(label, col)] = summ[col]
    return pd.DataFrame(out)
