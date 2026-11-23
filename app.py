"""
Streamlit app: Strategic Core 45/50/5
2270 Financial Modeling, Nova SBE, group assignment (optional application).

All calculations come from portfolio_core.py, the same engine used in the
notebook. With the default settings the app reproduces Table 1 of the prospectus.
Run locally:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import portfolio_core as pc

st.set_page_config(page_title="Strategic Core 45/50/5", layout="wide")


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading data from Yahoo Finance...")
def get_data():
    return pc.load_data(pc.START_DATE, pc.END_DATE, retries=2, wait=3)


st.title("Strategic Core 45/50/5: interactive backtest")
st.caption("US equity (SPY), US aggregate bonds (AGG) and gold (GLD), monthly data "
           "from Yahoo Finance. Change an assumption in the sidebar and compare the "
           "result with the prospectus default and the benchmarks.")

try:
    data = get_data()
    st.caption("Data source: live download from Yahoo Finance (yfinance API).")
except Exception as e:  # Yahoo often rate-limits shared cloud servers
    try:
        data = pc.load_snapshot()
        st.warning("Yahoo Finance is not reachable from this server right now, so the "
                   "app uses the repository snapshot of the same monthly data "
                   f"(created with portfolio_core.save_snapshot, {data.index[0]:%b %Y} "
                   f"to {data.index[-1]:%b %Y}). Calculations are unchanged.")
    except FileNotFoundError:
        st.error(f"Data download failed ({e}). Please reload the page in a minute.")
        st.stop()

# ---------------- Sidebar: assumptions ----------------
st.sidebar.header("Assumptions")
st.sidebar.subheader("Target weights")
eq = st.sidebar.slider("Equity, S&P 500 (SPY) %", 0, 100, 45, 5)
gold = st.sidebar.slider("Gold (GLD) %", 0, 100 - eq, min(5, 100 - eq), 1)
bonds = 100 - eq - gold
st.sidebar.markdown(f"**Bonds (AGG): {bonds}%** (remainder)")

st.sidebar.subheader("Rebalancing")
reb_options = ["Q", "M", "A", "threshold", "none"]
reb = st.sidebar.selectbox("Rule", reb_options, index=0,
                           format_func=lambda k: pc.REBALANCE_LABELS[k])
threshold = pc.DEFAULT_THRESHOLD
if reb == "threshold":
    threshold = st.sidebar.slider("Band: rebalance when any weight deviates by more than (pp)",
                                  1, 15, 5, 1) / 100
tc = st.sidebar.slider("Transaction cost (bps of traded value, one-way)", 0, 100,
                       int(pc.DEFAULT_TC_BPS), 5)

st.sidebar.subheader("Sample period")
years = sorted(set(data.index.year))
y0, y1 = st.sidebar.select_slider("Years", options=years, value=(years[0], years[-1]))
sample = data.loc[str(y0):str(y1)]

weights = {"SPY": eq / 100, "AGG": bonds / 100, "GLD": gold / 100}

# ---------------- Calculations ----------------
user = pc.run_all(sample, weights, reb, threshold, tc)
default = pc.run_all(sample)  # prospectus settings on the same sample

returns = pd.DataFrame({
    "Your portfolio": user["returns"]["Strategy"],
    "Prospectus default (45/50/5, quarterly)": default["returns"]["Strategy"],
    "60/40 benchmark": user["returns"]["60/40 benchmark"],
    "S&P 500": user["returns"]["S&P 500"],
})
summary = pd.concat([
    user["summary"]["Strategy"].rename("Your portfolio"),
    default["summary"]["Strategy"].rename("Prospectus default (45/50/5, quarterly)"),
    user["summary"]["60/40 benchmark"],
    user["summary"]["S&P 500"],
], axis=1)

# ---------------- Headline ----------------
s, b = summary["Your portfolio"], summary["60/40 benchmark"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ann. return", f"{100*s['Ann. average return']:.2f}%",
          f"{100*(s['Ann. average return']-b['Ann. average return']):+.2f} pp vs 60/40")
c2.metric("Volatility", f"{100*s['Ann. volatility']:.2f}%",
          f"{100*(s['Ann. volatility']-b['Ann. volatility']):+.2f} pp vs 60/40",
          delta_color="inverse")
c3.metric("Sharpe ratio", f"{s['Sharpe ratio']:.2f}",
          f"{s['Sharpe ratio']-b['Sharpe ratio']:+.2f} vs 60/40")
c4.metric("Beta (vs S&P 500)", f"{s['Beta (vs S&P 500)']:.2f}")
c5.metric("Max drawdown", f"{100*s['Max drawdown']:.1f}%",
          f"{100*(s['Max drawdown']-b['Max drawdown']):+.1f} pp vs 60/40")

# ---------------- Charts ----------------
st.subheader(f"Cumulative performance, growth of $1 ({sample.index[0]:%b %Y} to {sample.index[-1]:%b %Y})")
st.line_chart(returns.apply(pc.cumulative), height=380)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Drawdown from previous peak")
    st.area_chart(returns.apply(pc.drawdown) * 100, height=280)
with col_b:
    st.subheader("Your portfolio: weights before rebalancing (%)")
    w = user["strategy_weights"].rename(columns={"SPY": "Equity", "AGG": "Bonds", "GLD": "Gold"})
    st.area_chart(w * 100, height=280)

# ---------------- Table ----------------
st.subheader("Performance measures")
st.dataframe(pc.format_summary(summary))

with st.expander("Definitions and assumptions"):
    st.markdown(f"""
- **Data:** Yahoo Finance (`yfinance`), daily adjusted closes converted to month-end; returns are total returns net of ETF fees. Full sample Jan 2005 to Dec 2025.
- **Risk-free rate:** 3M US T-bill yield (`^IRX`) at the previous month-end, divided by 12.
- **Annualisation:** mean x 12, volatility x sqrt(12). **Beta is not annualised.**
- **Sharpe ratio:** annualised mean excess return / annualised volatility of excess return.
- **Alpha, beta:** OLS of monthly excess returns on the S&P 500 ETF's excess returns; alpha x 12.
- **Information ratio:** annualised mean return difference vs the 60/40 benchmark / annualised tracking error.
- **Rebalancing:** checked at month-end with data known at that time; costs = turnover x bps. Initial purchase not charged.
- **Benchmarks:** 60/40 SPY/AGG rebalanced quarterly without costs; S&P 500 ETF buy and hold.
- Past performance is not a reliable indicator of future results. All figures in USD.
""")
