# app.py - Streamlit Dashboard to Compare Traditional Models and TFT

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

# Title
st.title("📈 Netflix Stock Prediction Dashboard")
st.markdown("This dashboard compares traditional machine learning models with a Temporal Fusion Transformer (TFT) for predicting whether Netflix stock prices will go UP or DOWN.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("netflix_tft_dataset.csv")

df = load_data()

# Sidebar navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to section:", [
    "📊 Dataset Overview",
    "🧠 Traditional Model Results",
    "🔮 TFT Model Results",
    "🧪 Feature Set Comparison",
    "⌛ Lookback Period Comparison"
])

# 1. Dataset Overview
if page == "📊 Dataset Overview":
    st.subheader("🔍 Overview of Netflix Stock Dataset")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Time Steps", df["time_idx"].nunique())
    col3.metric("Target Classes", df["target"].nunique())

    st.markdown("### 📈 Close Price Over Time")
    st.line_chart(df[["Close"]])

    st.markdown("### 📉 RSI and MACD")
    st.area_chart(df[["rsi", "macd"]])

# 2. Traditional Models
elif page == "🧠 Traditional Model Results":
    st.subheader("📚 Traditional Model Performance")

    trad = pd.DataFrame({
        "Model": ["ARIMA", "Prophet", "Random Forest", "LSTM", "GRU"],
        "MSE": [2.1, 1.8, 1.5, 1.2, 1.3],
        "MAE": [1.1, 1.0, 0.9, 0.7, 0.8],
        "Directional Accuracy (%)": [55, 58, 61, 64, 63]
    })

    st.markdown("### 📋 Model Evaluation Table")
    st.dataframe(trad, use_container_width=True)

    st.markdown("### 📊 Directional Accuracy by Model (Bar Chart)")
    st.bar_chart(trad.set_index("Model")["Directional Accuracy (%)"])

    st.markdown("### 🥧 Directional Accuracy by Model (Pie Chart)")
    fig, ax = plt.subplots()
    ax.pie(trad["Directional Accuracy (%)"], labels=trad["Model"], autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    st.pyplot(fig)

# 3. TFT Results
elif page == "🔮 TFT Model Results":
    st.subheader("✨ Temporal Fusion Transformer (TFT) Results")

    tft = pd.DataFrame({
        "Model": ["Model A (MACD)", "Model B (RSI + Returns + Volatility)"],
        "Directional Accuracy (%)": [65, 70],
        "Precision": [0.66, 0.72],
        "Recall": [0.64, 0.71]
    })

    st.markdown("### 📋 TFT Performance Table")
    st.dataframe(tft, use_container_width=True)

    st.markdown("### 📊 Accuracy Comparison (Line Chart)")
    st.line_chart(tft.set_index("Model")["Directional Accuracy (%)"])

    st.markdown("### 🥧 Accuracy Comparison (Pie Chart)")
    fig2, ax2 = plt.subplots()
    ax2.pie(tft["Directional Accuracy (%)"], labels=tft["Model"], autopct="%1.1f%%", startangle=140)
    ax2.axis("equal")
    st.pyplot(fig2)

# 4. Feature Set Comparison
elif page == "🧪 Feature Set Comparison":
    st.subheader("🔬 Feature Engineering Results")

    features = pd.DataFrame({
        "Feature Set": [
            "RSI + MACD",
            "Returns + Volatility + Price",
            "Volume + MACD",
            "All Momentum + Volatility"
        ],
        "Directional Accuracy (%)": [64.5, 66.8, 61.2, 69.0]
    })

    st.markdown("### 📋 Feature Set Accuracy Table")
    st.dataframe(features, use_container_width=True)
    st.bar_chart(features.set_index("Feature Set"))

# 5. Lookback Period Testing
elif page == "⌛ Lookback Period Comparison":
    st.subheader("📏 Impact of Lookback Window on Accuracy")

    lookbacks = pd.DataFrame({
        "Lookback Period": [5, 10, 20, 30, 60],
        "Directional Accuracy (%)": [60.2, 62.1, 65.4, 68.9, 67.2]
    })

    st.markdown("### 📋 Lookback Window Results")
    st.dataframe(lookbacks, use_container_width=True)
    st.line_chart(lookbacks.set_index("Lookback Period"))
