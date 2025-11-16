# streamlit_app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb  # Needed for plotting booster predictions

from utils import (
    load_clean_pipeline_dataset,
    build_live_dataset,
    make_features_v3,
    load_model,
    predict_proba_model,
)

st.set_page_config(page_title="QQQ 5-Day Predictor", layout="wide")
st.title("📈 QQQ — 5-Day Direction Predictor (Live & Automated)")

# =========================================================
# Absolute paths
# =========================================================
BASE = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_PATH = os.path.join(BASE, "..", "data", "processed", "new_QQQ_data.csv")
MODEL_PATH = os.path.join(BASE, "..", "models", "qqq_xgb_optuna_v3.pkl")


# =========================================================
# Load model + cleaned dataset
# =========================================================
@st.cache_resource
def load_main_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)

model = load_main_model()
clean_df = load_clean_pipeline_dataset(CLEAN_DATA_PATH)

if model is None or clean_df is None:
    st.error("❌ Missing model or pipeline dataset.")
    st.stop()


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "QQQ")
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")
st.sidebar.metric("Test Accuracy", "0.62")
st.sidebar.metric("Test AUC", "0.54")

source = "Live (auto)"   # always live mode
st.sidebar.write("Prediction Source: **Live (auto)**")

st.sidebar.markdown("---")


# =========================================================
# Select dataset
# =========================================================
if source == "Live (auto)":
    df = build_live_dataset(clean_df, ticker)
else:
    df = clean_df.copy()


# =========================================================
# Run feature engineering + prediction
# =========================================================
feats = make_features_v3(df)
latest = feats.iloc[[-1]]

prob = predict_proba_model(model, latest)
p = float(prob[-1])
label = "UP" if p > 0.5 else "DOWN"


# =========================================================
# Confidence Interpretation
# =========================================================
def interpret_confidence(p):
    if p > 0.65: return "High confidence UP 📈"
    if p > 0.55: return "Mild upward bias ↗️"
    if p >= 0.45: return "Neutral ⚪"
    if p >= 0.35: return "Mild downward bias ↘️"
    return "High confidence DOWN 📉"


# =========================================================
# Display Prediction
# =========================================================
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("🔮 Model Prediction")
    st.metric("5-Day UP Probability", f"{p:.3f}")
    st.write("Prediction:", "🟢 UP" if label == "UP" else "🔴 DOWN")
    st.write("Confidence:", interpret_confidence(p))

with col2:
    st.subheader("📊 Latest Feature Values")
    st.dataframe(latest.T)


# =========================================================
# Strategy Behavior Summary (Last 20)
# =========================================================
st.markdown("---")
st.subheader("📌 Strategy Behavior (Last 20 Predictions)")

recent_feats = feats.tail(20).copy()
recent_probs = predict_proba_model(model, recent_feats)
recent_feats["prob"] = recent_probs
recent_feats["signal"] = (recent_feats["prob"] > 0.5).astype(int)

signals = recent_feats["signal"].tolist()
streak = 1
for i in range(len(signals) - 1, 0, -1):
    if signals[i] == signals[i - 1]:
        streak += 1
    else:
        break

current_dir = "UP 🟢" if signals[-1] == 1 else "DOWN 🔴"

c1, c2, c3 = st.columns(3)
c1.metric("Current Signal", current_dir)
c2.metric("Streak Length", streak)
c3.metric("Avg Probability (20)", f"{recent_feats['prob'].mean():.3f}")

# Plot graph
fig, ax = plt.subplots(figsize=(10, 4))

# Numeric x positions
x_positions = list(range(len(recent_feats)))

# Line plot
ax.plot(x_positions, recent_feats["prob"], marker="o", label="Probability")

# Colored markers for UP/DOWN
for x, (_, row) in zip(x_positions, recent_feats.iterrows()):
    color = "green" if row["signal"] == 1 else "red"
    ax.scatter(x, row["prob"], color=color, s=50)

# Replace x-axis ticks with actual dates
ax.set_xticks(x_positions)
ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in recent_feats.index],
                   rotation=45, ha="right")

ax.axhline(0.5, linestyle="--", color="gray", label="Decision Threshold (0.5)")
ax.set_title("Recent Prediction Probabilities (Last 20)")
ax.set_ylabel("Probability of UP")
ax.set_ylim(0, 1)
ax.legend()

st.pyplot(fig)
