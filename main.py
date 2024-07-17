import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import yfinance as yf

import optimization

#######################

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
  # Integer input
  time_steps = st.number_input('# Of Days / period:', min_value=20, max_value=150, value=30)
  minimum = st.slider("Minimum investment in asset", min_value=0.0, max_value=1.0, value = 0.0, step = 0.1)
  maximum = st.slider("Maximum investment in asset", min_value=0.0, max_value=1.0, value =0.4, step = 0.1)

# App layout
st.title("Investment Proportions Over Time")

options = st.multiselect(label = 'Choose your tickers', options = tickers)

results = optimization.optimal_portfolio(options, hold_days=time_steps, min_bound = minimum, max_bound=maximum)[::-1]

# Create a figure
fig = go.Figure()

# Add traces for each timestamp
for i in range(len(results)):
    fig.add_trace(
        go.Bar(
            visible=False,
            name=results.index[i],
            x=results.columns,
            y=results.iloc[i]
        )
    )

# Set the first trace to be visible by default
fig.data[0].visible = True

# Create slider steps
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[
            {"visible": [False] * len(fig.data)},
            {"title": f"Timestamp: {results.index[i]}"}],
    )
    step["args"][0]["visible"][i] = True
    steps.append(step)

# Define the slider
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Timestamp: "},
    pad={"t": 50},
    steps=steps
)]

# Update layout to include the slider
fig.update_layout(
    sliders=sliders,
    title=f"Timestamp: {results.index[0]}",
)

# Display the figure in the Streamlit app
st.plotly_chart(fig)