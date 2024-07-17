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


def optimal_portfolio(tickers, hold_days, min_bound, max_bound):

    # Dates
    end_date = datetime.today()
    start_date = end_date - pd.to_timedelta(5 * 365, 'D')

    # Fetch Adjusted Close Prices
    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data["Adj Close"]

    # Calculate Log Returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = pd.DataFrame(log_returns)
    log_returns.dropna(inplace=True)
    log_returns.index = log_returns.index.strftime('%Y-%m-%d')

    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean() * weights) * 252 
        # Yearly return

    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
    
    def split_timeseries(data, chunk_size=hold_days):
        # Reverse the dataframe to start from the bottom
        data_reversed = data.iloc[::-1]
        # Split the dataframe into chunks of `chunk_size`
        chunks = [data_reversed.iloc[i:i + chunk_size] for i in range(0, len(data_reversed), chunk_size)]
        # Reverse each chunk back to the original order
        chunks = [chunk.iloc[::-1] for chunk in chunks]
        # Return the chunks
        return chunks

    # Initial Weights
    initial_weights = np.repeat(1 / len(tickers), len(tickers))

    # Constraints and Bounds
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(min_bound, max_bound) for _ in range(len(tickers))]

    ten_year_treasury = yf.Ticker("^TNX")
    hist = ten_year_treasury.history(period="max")['Close']
    hist.index = hist.index.strftime('%Y-%m-%d')

    optimal_pf_weights = []
    optimal_pf_return = []
    optimal_pf_vol = []
    optimal_pf_sharpe = []
    ts = []

    for log_returns_i in split_timeseries(log_returns):
        # Calculate Covariance Matrix
        cov_matrix = log_returns_i.cov() * 252
        # Fetch Risk-Free Rate     
        risk_free_rate = hist.loc[hist.index.intersection(log_returns_i.index)].mean() / 100
        #print(log_returns_i.index[0], log_returns_i.index[-1], risk_free_rate)

        # Mean Variance Optimization
        # Optimize
        optimize_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns_i, cov_matrix, risk_free_rate), method='SLSQP',
                                    constraints=constraints, bounds=bounds)
        optimal_weights = optimize_results.x
        optimal_return = expected_return(optimal_weights, log_returns_i)
        optimal_volatility = standard_deviation(optimal_weights, cov_matrix)
        optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns_i, cov_matrix, risk_free_rate)
        
        ts.append(log_returns_i.index[-1])
        optimal_pf_weights.append(list(optimal_weights))
        optimal_pf_return.append(optimal_return)
        optimal_pf_vol.append(optimal_volatility)
        optimal_pf_sharpe.append(optimal_sharpe_ratio)
        
    optimal_pf_weights = pd.DataFrame(optimal_pf_weights, columns=tickers, index=ts)

    return optimal_pf_weights