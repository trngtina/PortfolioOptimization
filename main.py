import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from streamlit_tags import st_tags

import sharpe

#######################
# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# Sidebar for User Inputs
with st.sidebar:
  st.title("📊 Portfolio Optimization")
  st.write("`Created by:`")
  linkedin_url = "https://www.linkedin.com/in/tina-truong-nguyen/"
  st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Tina Truong`</a>', unsafe_allow_html=True)


tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NVDA", "BRK-B", "JPM", "V",
    "PG", "UNH", "HD", "DIS", "MA", "PYPL", "NFLX", "INTC", "CMCSA", "PEP",
    "KO", "T", "PFE", "MRK", "WMT", "XOM", "CVX", "BABA", "ORCL", "NKE",
    "IBM", "BA", "CRM", "GE", "QCOM", "CSCO", "ABBV", "ABT", "MCD", "SBUX"
]

options = st.multiselect(
    "Choose your tickers",
    tickers, max_selections = 10)

st.write("You selected:", options)