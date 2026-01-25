import json
from pathlib import Path
import pandas as pd
import streamlit as st
from datetime import datetime
import time

st.set_page_config(page_title="Daily Economic Metrics", layout="wide")

# Put artifacts inside the repo, next to app.py
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dashboard_data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

LATEST = DATA_DIR / "latest.json"
HISTORY = DATA_DIR / "history.csv"

st.title("Daily Economic Metrics")

# ---- Load latest snapshot
if not LATEST.exists():
    st.warning("No data yet. Run the daily script once to create dashboard_data/latest.json")
    st.stop()

latest_records = json.loads(LATEST.read_text())
latest_df = pd.DataFrame(latest_records)

# Make a nice table
latest_df = latest_df[["name", "value", "units", "date", "retrieved_at_utc"]].copy()

st.subheader("Latest snapshot")
st.dataframe(latest_df, use_container_width=True)

# ---- Key metrics in cards
def get_value(name: str):
    row = latest_df[latest_df["name"] == name]
    if row.empty:
        return None
    v = row.iloc[0]["value"]
    u = row.iloc[0]["units"]
    d = row.iloc[0]["date"]
    return v, u, d

cols = st.columns(4)
cards = [
    "US CPI YoY",
    "Unemployment Rate",
    "Effective Fed Funds Rate",
    "S&P 500",
]
for c, metric in zip(cols, cards):
    out = get_value(metric)
    with c:
        if out is None:
            st.metric(metric, "—")
        else:
            v, u, d = out
            if pd.isna(v):
                st.metric(metric, "—")
            else:
                st.metric(metric, f"{v:.2f}{u}".rstrip(), help=f"As of {d}")

# ---- Plots (PNG)
st.subheader("Charts")

macro_png = DATA_DIR / "metrics_macro.png"
markets_png = DATA_DIR / "metrics_markets.png"

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Macro**")
    if macro_png.exists():
        st.image(str(macro_png), use_container_width=True)
    else:
        st.info("Macro plot not found yet.")

with c2:
    st.markdown("**Markets**")
    if markets_png.exists():
        st.image(str(markets_png), use_container_width=True)
    else:
        st.info("Markets plot not found yet.")

# ---- Optional: history chart (if you want a simple line chart)
st.subheader("History (optional)")
if HISTORY.exists():
    hist = pd.read_csv(HISTORY)
    # Example: plot CPI YoY over time
    cpi = hist[hist["name"] == "US CPI YoY"].copy()
    if not cpi.empty:
        cpi["retrieved_at_utc"] = pd.to_datetime(cpi["retrieved_at_utc"])
        cpi = cpi.sort_values("retrieved_at_utc")
        st.line_chart(cpi.set_index("retrieved_at_utc")["value"])
else:
    st.info("No history.csv yet.")

latest_json = DATA_DIR / "latest.json"
if latest_json.exists():
    mtime = datetime.fromtimestamp(latest_json.stat().st_mtime)
    st.caption(f"Last data refresh: {mtime}")
else:
    st.warning("No latest.json found yet (refresh job hasn’t produced output).")

if st.button("Clear cache"):
    st.cache_data.clear()
    st.rerun()

