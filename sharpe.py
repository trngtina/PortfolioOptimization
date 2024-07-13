import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize


tickers = ["SPY", "BND", "GLD", "QQQ", "VTI"]
end_date = datetime.today()
start_date = end_date - pd.to_timedelta(5 * 365, 'D')
print(start_date)

adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start = start_date, end = end_date)
    adj_close_df[ticker] = data["Adj Close"]

print(adj_close_df)

log_returns = np.log(adj_close_df / adj_close_df.shift(1))

log_returns.dropna(inplace=True)
log_returns

# Covariance matrix
cov_matrix = log_returns.cov() * 252
print(cov_matrix)

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

# Define the ticker symbol for the 10-year Treasury bond yield
ticker = "^TNX"

# Fetch the data
ten_year_treasury = yf.Ticker(ticker)

# Get the historical market data (we'll just fetch the latest data point)
#hist = ten_year_treasury.history(start=end_date, end=end_date)
hist = ten_year_treasury.history(period="300d")

# Extract the latest closing price (which represents the yield)
risk_free_rate = hist['Close'].iloc[-1] / 100
risk_free_rate

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.4) for _ in range(len(tickers))]

initial_weights = np.repeat(1/len(tickers), len(tickers))
print(initial_weights)

optimize_results = minimize(neg_sharpe_ratio, initial_weights, args = (log_returns, cov_matrix, risk_free_rate), method = 'SLSQP',
                            constraints=constraints, bounds=bounds)

optimal_weights = optimize_results.x

print("Optimal weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

print()

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print(f"Expected Portfolio Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Expected Sharpe ratio: {optimal_sharpe_ratio:.4f}")