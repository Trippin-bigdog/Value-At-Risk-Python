import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf

# Title and Sidebar Navigation
st.title("Advanced Portfolio Risk Dashboard")
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose Analysis:",
    ["Portfolio Data", "VaR Analysis", "Fetch Real-Time Data"]
)

# Load historical data
@st.cache_data
def load_data():
    data = pd.read_csv("historical_prices.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)
    return data, returns

data, returns = load_data()

# Ensure returns contains numeric values
returns = returns.apply(pd.to_numeric, errors='coerce')

# Portfolio configuration
tickers = list(data.columns)
portfolio_weights = np.array([1 / len(tickers)] * len(tickers), dtype=float)  # Ensure weights are floats
portfolio_value = 1000000  # Example portfolio value

# Calculate portfolio returns
portfolio_returns = returns.dot(portfolio_weights)

# Helper function: calculate VaR and CVaR
def calculate_var_and_cvar(method, confidence_level=0.95, num_simulations=100000):
    if method == "Historical Simulation":
        var = portfolio_returns.quantile(1 - confidence_level)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
    elif method == "Variance-Covariance":
        mean = portfolio_returns.mean()
        std_dev = portfolio_returns.std()
        z_score = norm.ppf(1 - confidence_level)
        var = z_score * std_dev
        cvar = mean - std_dev * norm.pdf(z_score) / (1 - confidence_level)
    elif method == "Monte Carlo":
        simulated_returns = np.random.normal(
            portfolio_returns.mean(), portfolio_returns.std(), num_simulations
        )
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        cvar = np.mean(simulated_returns[simulated_returns <= var])
    return var, cvar

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
    st.subheader("Value at Risk (VaR) and Conditional VaR (CVaR) Analysis")

    # Confidence level selection
    confidence = st.slider("Select Confidence Level:", 90, 99, 95) / 100
    method = st.radio("Select VaR Method:", ["Historical Simulation", "Variance-Covariance", "Monte Carlo"])

    # Calculate VaR and CVaR
    var, cvar = calculate_var_and_cvar(method, confidence_level=confidence)
    var_monetary = var * portfolio_value
    cvar_monetary = cvar * portfolio_value

    st.write(f"**{confidence*100:.0f}% {method} VaR (1-day):**")
    st.write(f"- Percentage Loss: {var:.4%}")
    st.write(f"- Monetary Loss: ${var_monetary:,.2f}")

    st.write(f"**{confidence*100:.0f}% {method} CVaR (1-day):**")
    st.write(f"- Percentage Loss: {cvar:.4%}")
    st.write(f"- Monetary Loss: ${cvar_monetary:,.2f}")

    # Visualize VaR and CVaR
    st.write("### Portfolio Return Distribution with VaR and CVaR")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(portfolio_returns, bins=50, color="skyblue", edgecolor="black", density=True)
    ax.axvline(var, color="red", linestyle="dashed", linewidth=2, label=f"VaR {confidence*100:.0f}%: {var:.4f}")
    ax.axvline(cvar, color="green", linestyle="dashed", linewidth=2, label=f"CVaR {confidence*100:.0f}%: {cvar:.4f}")
    ax.set_title("Portfolio Return Distribution")
    ax.set_xlabel("Daily Returns")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

# Real-Time Data Section
elif option == "Fetch Real-Time Data":
    st.subheader("Fetch Real-Time Market Data")
    tickers = st.text_input("Enter Stock Tickers (comma-separated):", "AAPL,MSFT,GOOGL,AMZN")
    if st.button("Fetch Data"):
        try:
            real_time_data = yf.download(tickers.split(','), period="5d")['Close']
            if real_time_data.empty:
                st.error("No data found for the entered tickers. Please check the ticker symbols.")
            else:
                st.write("### Real-Time Prices")
                st.dataframe(real_time_data)
                st.line_chart(real_time_data)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
