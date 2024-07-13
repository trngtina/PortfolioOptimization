import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

import yfinance as yf

#import optimization

#######################


# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA",
    "SPY", "BND", "GLD", "QQQ", "VTI"
]

# Sidebar for User Inputs
with st.sidebar:
  st.title("📊 Portfolio Optimization")
  st.write("`Created by:`")
  linkedin_url = "https://www.linkedin.com/in/tina-truong-nguyen/"
  st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Tina Truong`</a>', unsafe_allow_html=True)
  
  st.markdown("---")

  options = st.multiselect(
    "Choose your tickers",
    tickers, max_selections = 10)
  

end_date = datetime.today()
start_date = end_date - pd.to_timedelta(5 * 365, 'D')
st.write(end_date)
st.write(start_date)

start_date = end_date - pd.to_timedelta(5 * 365, 'D')

adj_close_df = pd.DataFrame()

for ticker in options:
    data = yf.download(ticker, start = start_date, end = end_date)
    adj_close_df[ticker] = data["Adj Close"]

log_returns = np.log(adj_close_df / adj_close_df.shift(1))
log_returns.dropna(inplace=True)
st.write(log_returns)

cov_matrix = log_returns.cov() * 252
st.write(cov_matrix)

