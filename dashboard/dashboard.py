# dashboard.py (Maximalist Version)

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from wordcloud import WordCloud

# === Setup Paths ===
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
if not os.path.exists(LOG_DIR):
    st.error(f"No log directory found at {LOG_DIR}")
    st.stop()

log_dirs = sorted(os.listdir(LOG_DIR))
if not log_dirs:
    st.error("No logs found in 'logs/' folder.")
    st.stop()

LATEST_LOG = log_dirs[-1]
LOG_PATH = os.path.join(LOG_DIR, LATEST_LOG, "trade_logs.jsonl")
if not os.path.exists(LOG_PATH):
    st.error(f"No trade log file found at: {LOG_PATH}")
    st.stop()

st.set_page_config(layout="wide")
st.title("🧠 Quant Bot Dashboard")
st.caption(f"Live trades, confidence, and indicator analysis from: `{LATEST_LOG}`")

@st.cache_data
def load_logs(log_path):
    data = []
    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(data)
    if not df.empty and "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df

df = load_logs(LOG_PATH)
if df.empty:
    st.warning("Log file is empty or unreadable.")
    st.stop()

# === Sidebar ===
symbols = df["symbol"].unique()
selected_symbol = st.sidebar.selectbox("🪙 Select Symbol", symbols)
signals = df["signal"].unique().tolist()
selected_signals = st.sidebar.multiselect("📶 Filter by Signal", signals, default=signals)
min_date = df["datetime"].min().date()
max_date = df["datetime"].max().date()
date_range = st.sidebar.date_input("🗓 Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# === Filtering ===
filtered = df[
    (df["symbol"] == selected_symbol) &
    (df["signal"].isin(selected_signals)) &
    (df["datetime"].dt.date >= date_range[0]) &
    (df["datetime"].dt.date <= date_range[1])
].copy()

def clean_indicator_str(val):
    if isinstance(val, str): return val
    if isinstance(val, list): return ", ".join(str(i) for i in val)
    return ""
filtered["Top Indicators"] = filtered["top_indicators"].apply(clean_indicator_str)

# === Key Metrics ===
st.subheader("📈 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Score", f"{filtered['score'].mean():.4f}")
col2.metric("Avg Meta Confidence", f"{filtered['meta_confidence'].mean():.4f}")
col3.metric("Latest Signal", filtered['signal'].iloc[-1] if not filtered.empty else "N/A")

# === Score and Confidence ===
st.subheader(f"📊 Final Score / Signal for {selected_symbol}")
fig1, ax1 = plt.subplots(figsize=(7, 3))
ax1.plot(filtered["datetime"], filtered["score"], label="Score", color="blue")
ax1.set_ylabel("Score")
ax1.grid(True)
st.pyplot(fig1)

st.subheader("🔍 Meta Confidence")
fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(filtered["datetime"], filtered["meta_confidence"], label="Confidence", color="green")
ax2.set_ylabel("Confidence")
ax2.grid(True)
st.pyplot(fig2)

# === Word Cloud ===
st.subheader("🌀 Indicator Word Cloud")
indicators = [ind.strip().split("(")[0] for row in filtered["Top Indicators"] for ind in row.split(",")]
word_freq = Counter(indicators)
wordcloud = WordCloud(width=800, height=300, background_color='white').generate_from_frequencies(word_freq)
fig_wc, ax_wc = plt.subplots(figsize=(10, 3))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')
st.pyplot(fig_wc)

# === Indicator Frequency ===
st.subheader("📊 Most Frequent Indicators")
freq_df = pd.DataFrame(word_freq.items(), columns=["Indicator", "Count"]).sort_values("Count", ascending=False)
st.bar_chart(freq_df.set_index("Indicator").head(15))

# === Signal Summary Table ===
st.subheader("📊 Signal Summary Table")
signal_stats = filtered.groupby("signal")["score"].agg(["count", "mean", "std"]).reset_index()
st.dataframe(signal_stats)

# === PnL Simulation ===
st.subheader("💰 Simulated PnL (Rolling)")
def simulate_pnl(scores, initial=1000):
    capital = [initial]
    for score in scores:
        capital.append(capital[-1] * (1 + (score if not pd.isna(score) else 0)))
    return capital[1:]

pnl = simulate_pnl(filtered["score"].fillna(0))
pnl_df = pd.DataFrame({"datetime": filtered["datetime"], "PnL": pnl})
fig3, ax3 = plt.subplots(figsize=(7, 3))
ax3.plot(pnl_df["datetime"], pnl_df["PnL"], color="gold")
ax3.set_ylabel("Portfolio Value")
ax3.grid(True)
st.pyplot(fig3)

# === Rolling Sharpe Ratio ===
st.subheader("🔹 Rolling Sharpe Ratio")
returns = filtered["score"].fillna(0)
rolling_mean = returns.rolling(30).mean()
rolling_std = returns.rolling(30).std()
rolling_sharpe = rolling_mean / rolling_std
fig4, ax4 = plt.subplots(figsize=(7, 2.5))
ax4.plot(filtered["datetime"], rolling_sharpe, color='purple')
ax4.set_title("Sharpe (30 window)")
ax4.grid(True)
st.pyplot(fig4)

# === Drawdown ===
st.subheader("🔹 Drawdown from Peak")
cumulative_returns = (1 + returns).cumprod()
rolling_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - rolling_max) / rolling_max
fig5, ax5 = plt.subplots(figsize=(7, 2.5))
ax5.plot(filtered["datetime"], drawdown.fillna(0).astype(float), color='red')
ax5.fill_between(filtered["datetime"], drawdown.fillna(0).astype(float), 0, alpha=0.3, color='red')
ax5.grid(True)
st.pyplot(fig5)

# === Download Logs ===
csv = filtered.to_csv(index=False)
st.download_button("Download CSV", data=csv, file_name="filtered_trades.csv")

st.caption("Built by Quant Don | Max Accuracy Mode Enabled.")

# === Heatmap: Top Indicator Influence ===
st.subheader("🔥 Top Indicator Influence Heatmap")

# Flatten all top_indicators into a DataFrame
flat_indicators = []
for _, row in filtered.iterrows():
    if isinstance(row["top_indicators"], list):
        for ind in row["top_indicators"]:
            if isinstance(ind, list) or isinstance(ind, tuple):
                flat_indicators.append((row["datetime"], ind[0], ind[1]))

if flat_indicators:
    heat_df = pd.DataFrame(flat_indicators, columns=["datetime", "indicator", "value"])
    pivot = heat_df.pivot_table(index="indicator", columns="datetime", values="value", aggfunc="mean")
    fig_heat, ax_heat = plt.subplots(figsize=(12, max(4, len(pivot) * 0.3)))
    sns.heatmap(pivot.fillna(0), cmap="coolwarm", center=0, ax=ax_heat)
    ax_heat.set_title("Indicator Influence Over Time")
    st.pyplot(fig_heat)
else:
    st.info("No top indicator data available for heatmap.")

# === Category Confidence Breakdown ===
st.subheader("📚 Category Confidence Breakdown")

if "category_subscores" in filtered.columns:
    cat_rows = []
    for _, row in filtered.iterrows():
        if isinstance(row["category_subscores"], dict):
            row_dict = row["category_subscores"]
            row_dict["datetime"] = row["datetime"]
            cat_rows.append(row_dict)

    if cat_rows:
        cat_df = pd.DataFrame(cat_rows).set_index("datetime")
        avg_scores = cat_df.mean().sort_values()
        st.bar_chart(avg_scores)
    else:
        st.info("No category_subscores available.")
else:
    st.info("Category subscores not found in log file.")

# === Debug: Raw JSON Viewer ===
with st.expander("🛠 Show Raw Log Entries"):
    st.json(filtered.tail(1).to_dict(orient="records")[0])
# === 🔁 Signal Streaks (Win/Loss/Neutral) ===
st.subheader("🔁 Signal Streaks")

def calculate_streaks(signals):
    streaks = []
    current = {"signal": None, "length": 0}
    for sig in signals:
        if sig == current["signal"]:
            current["length"] += 1
        else:
            if current["signal"] is not None:
                streaks.append(current.copy())
            current = {"signal": sig, "length": 1}
    streaks.append(current)
    return streaks

streaks = calculate_streaks(filtered["signal"].tolist())
streak_df = pd.DataFrame(streaks)
streak_summary = streak_df.groupby("signal")["length"].agg(["count", "mean", "max"]).reset_index()
st.dataframe(streak_summary.rename(columns={"count": "Streaks", "mean": "Avg Length", "max": "Max Streak"}))

# === 📊 Profitability by Score Range ===
st.subheader("📊 Profitability by Score Range")

bins = [-np.inf, -0.2, -0.1, 0, 0.1, 0.2, np.inf]
labels = ["<-0.2", "-0.2 to -0.1", "-0.1 to 0", "0 to 0.1", "0.1 to 0.2", ">0.2"]
filtered["score_bucket"] = pd.cut(filtered["score"], bins=bins, labels=labels)

bucket_stats = filtered.groupby("score_bucket")["score"].agg(["count", "mean", "std"]).reset_index()
st.dataframe(bucket_stats)
st.bar_chart(bucket_stats.set_index("score_bucket")["mean"])

# === 🧩 Per-Symbol Summary ===
st.subheader("🧩 Per-Symbol Overview")

symbol_stats = df.groupby("symbol")["score"].agg(["count", "mean", "std"]).sort_values("mean", ascending=False)
st.dataframe(symbol_stats.rename(columns={"count": "Logs", "mean": "Avg Score", "std": "Volatility"}))