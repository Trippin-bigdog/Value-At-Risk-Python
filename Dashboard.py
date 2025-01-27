import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the historical data
st.title("Portfolio Risk Dashboard")
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose Analysis:",
    ["Portfolio Data", "VaR Analysis"]
)

# Load saved data
data = pd.read_csv("historical_prices.csv", index_col=0, parse_dates=True)
returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)

# Portfolio configuration
tickers = list(data.columns)
num_assets = len(tickers)
portfolio_weights = np.array([1 / num_assets] * num_assets, dtype=float)  # Ensure weights are floats
portfolio_value = 1000000  # Example portfolio value

# Ensure the returns DataFrame contains numeric values
returns = returns.apply(pd.to_numeric, errors='coerce')

# Calculate portfolio returns
portfolio_returns = returns.dot(portfolio_weights)

# Helper function: calculate VaR
def calculate_var(method, confidence_level=0.95, num_simulations=100000):
    if method == "Historical Simulation":
        var = portfolio_returns.quantile(1 - confidence_level)
    elif method == "Variance-Covariance":
        mean = portfolio_returns.mean()
        std_dev = portfolio_returns.std()
        z_score = norm.ppf(1 - confidence_level)
        var = z_score * std_dev
    elif method == "Monte Carlo":
        simulated_returns = np.random.normal(
            portfolio_returns.mean(), portfolio_returns.std(), num_simulations
        )
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    return var

# Portfolio Data Section
if option == "Portfolio Data":
    st.subheader("Historical Portfolio Data")
    st.write("### Adjusted Closing Prices")
    st.dataframe(data)

    st.write("### Daily Returns")
    st.dataframe(returns)

    # Visualize prices
    st.write("### Portfolio Prices Over Time")
    st.line_chart(data)

# VaR Analysis Section
elif option == "VaR Analysis":
    st.subheader("Value at Risk (VaR) Analysis")

    # Confidence level selection
    confidence = st.slider("Select Confidence Level:", 90, 99, 95) / 100
    method = st.radio("Select VaR Method:", ["Historical Simulation", "Variance-Covariance", "Monte Carlo"])

    # Calculate VaR
    var = calculate_var(method, confidence_level=confidence)
    var_monetary = var * portfolio_value

    st.write(f"**{confidence*100:.0f}% {method} VaR (1-day):**")
    st.write(f"- Percentage Loss: {var:.4%}")
    st.write(f"- Monetary Loss: ${var_monetary:,.2f}")

    # Visualize VaR
    st.write("### Portfolio Return Distribution with VaR")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(portfolio_returns, bins=50, color="skyblue", edgecolor="black", density=True)
    ax.axvline(var, color="red", linestyle="dashed", linewidth=2, label=f"VaR {confidence*100:.0f}%: {var:.4f}")
    ax.set_title("Portfolio Return Distribution")
    ax.set_xlabel("Daily Returns")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

