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
import plotly.colors as pcolors
from plotly.subplots import make_subplots

from optimization import optimal_portfolio

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
  mini = st.slider("Minimum investment in asset", min_value=0.0, max_value=1.0, value = 0.0, step = 0.05)
  maxi = st.slider("Maximum investment in asset", min_value=0.0, max_value=1.0, value =0.4, step = 0.05)

# App layout
st.title("Investment Proportions Over Time")

options = st.multiselect(label = 'Choose your tickers', options = tickers, default = tickers)

if not options:
    st.warning("Please select at least one ticker.")
else:
    retry = True
    while retry:
        try:
            results, performance = optimal_portfolio(options, hold_days=time_steps, min_bound=mini, max_bound=maxi)

            # Generate a list of colors based on the number of columns using Plotly discrete color scale
            num_cols = len(results.columns)
            color_scale = px.colors.qualitative.Plotly
            colors = color_scale * (num_cols // len(color_scale)) + color_scale[:num_cols % len(color_scale)]
            color_dict = {column: colors[i] for i, column in enumerate(results.columns)}

            # Create the figure with subplots
            fig = make_subplots(
                rows=1, cols=2, 
                column_widths=[0.4, 0.6],  # Adjust column widths to add separation
                subplot_titles=("Pie Chart", "Performance Table"),
                specs=[[{"type": "pie"}, {"type": "table"}]]  # Specify subplot types
            )

            # Iterate over each row in the DataFrame to add pie charts and a table trace
            for i in range(len(results)):
                # Filter out columns with 0% values
                filtered_results = results.iloc[i][results.iloc[i] > 0]
                filtered_colors = [color_dict[label] for label in filtered_results.index]

                # Add the filtered pie trace
                fig.add_trace(
                    go.Pie(
                        visible=False,
                        name=results.index[i],
                        labels=filtered_results.index,
                        values=filtered_results,
                        marker=dict(colors=filtered_colors),
                        sort=False,
                        hole=.4,
                        textinfo='label',
                        hoverinfo='label+percent'
                    ),
                    row=1, col=1  # Place pie chart in the first column
                )

                # Add the table trace for performance metrics
                table_trace = go.Table(
                    header=dict(values=list(performance.columns)),
                    cells=dict(values=[performance.iloc[i][col] for col in performance.columns]),
                    visible=False
                )
                fig.add_trace(table_trace, row=1, col=2)  # Place table in the second column

            # Set the last trace (Pie chart) and table trace to be visible by default
            fig.data[-2].visible = True
            fig.data[-1].visible = True

            # Create slider steps
            steps = []
            for i in range(len(results)):
                step = dict(
                    method="update",
                    args=[
                        {"visible": [False] * len(fig.data)},  # Set all traces to invisible initially
                        {"title": f"Timestamp: {results.index[i]}"}  # Update title with timestamp
                    ],
                )
                step["args"][0]["visible"][2 * i] = True  # Set Pie chart trace visible
                step["args"][0]["visible"][2 * i + 1] = True  # Set Table trace visible
                steps.append(step)

            # Define the slider
            sliders = [dict(
                active=len(results) - 1,  # Set the active step to the last timestamp initially
                currentvalue={"prefix": "Timestamp: "},
                pad={"t": 50},
                steps=steps
            )]

            # Update layout to include the slider
            fig.update_layout(
                sliders=sliders,
                title=f"Timestamp: {results.index[-1]}",  # Set initial title to the most recent timestamp
                showlegend=False,  # Hide legend if not needed
                uniformtext_minsize=10,
                uniformtext_mode='hide'
            )

            # Display the figure in the Streamlit app
            st.plotly_chart(fig)
            retry = False  # Exit loop if successful

        except Exception as e:
            # Handle exceptions or errors here
            # st.error(f"An error occurred: {e}")
            retry = st.button("Estimate proportions")