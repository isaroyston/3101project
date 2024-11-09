import json
import requests
from datetime import datetime, timedelta
import plotly.express as px
import pandas as pd
import math

def get_delivery_predictions(supplier_name,n_days):

    payload = {
        "supplier_name": supplier_name,
        "n_days": int(n_days)
    }

    res = requests.post(f"http://127.0.0.1:8000/Subgroup_B_Q3/make_predictions/", json=payload)
    print("TYPE", type(res.json()))
    res_dict = json.loads(res.json()['predictions'])

    x = [(datetime.today() + timedelta(days=i)).strftime(format="%Y-%m-%d") for i in range(1, n_days + 1)]
    y = [i[1] for i in res_dict.items()]

    fig = px.line(x=x, y=y)
    fig.add_hline(y=3, line_color='red')
    fig.update_layout(xaxis_title="Order placed on", yaxis_title="Expected time to deliver (days)", showlegend=False)

    df = pd.DataFrame({"Orders placed on":x, "Orders expected to be late by (days)":y})
    df = df.loc[df["Orders expected to be late by (days)"] > 3]
    df["Orders expected to be late by (days)"] = df["Orders expected to be late by (days)"].apply(math.ceil) - 3

    return fig, df


# streamlit app functions (marketing channels analysis)
"""
ROI functions (q3)
"""
def analyze_avg_roi_by_year(df, campaigns):
    """
    Analyzes and visualizes the average ROI by year for each marketing channel based on specified campaigns.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the campaign data.
    - campaigns (list): A list of campaign names to filter for analysis (e.g., ["Easter Sale", "Mid-Year Sale"]).

    Returns:
    - A Plotly line chart showing the average ROI for each marketing channel across years for the specified campaigns.
    """
    # Filter the DataFrame for the specified campaigns
    df_filtered = df[df['campaign_name'].isin(campaigns)]
    
    # Calculate the average ROI for each channel type by year
    df_avg_roi_by_year = df_filtered.groupby(["year", "channel_type"])["roi"].mean().reset_index()
    
    # Plotting the average ROI for each marketing channel across years
    fig = px.line(
        df_avg_roi_by_year, 
        x="year", 
        y="roi", 
        color="channel_type",
        title=f"Average ROI by Year for Each Marketing Channel (Selected Campaigns: {', '.join(campaigns)})",
        labels={"roi": "Average Return on Investment (%)", "year": "Year", "channel_type": "Channel Type"},
        markers=True
    )
    
    return fig


def plot_rev_cost_roi(df, campaigns):
    """
    Creates a scatter plot with average revenue on the x-axis, average cost on the y-axis, 
    abbreviation as text markers for each point, and color representing the marketing channel,
    along with a year slider.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the campaign data.
    - campaigns (list): A list of campaign names to filter for analysis (e.g., ["Easter Sale", "Mid-Year Sale"]).
    - abbrev (dict): A dictionary mapping campaign names to abbreviations.

    Returns:
    - A Plotly scatter plot showing average revenue vs. average cost, with marketing campaign abbreviations as markers,
      color by channel, and a year slider.
    """
    # Filter the DataFrame for the specified campaigns
    df_filtered = df[df['campaign_name'].isin(campaigns)]
    
    # Calculate the average revenue, cost, and ROI for each channel and year
    df_avg = df_filtered.groupby(['year', 'channel_type']).agg({
        'rev': 'mean',
        'cost': 'mean',
        'roi': 'mean',
    }).reset_index()

    # Scale the ROI for better visualization
    df_avg['scaled_roi'] = df_avg['roi'] ** 0.5
    
    # Plotting average revenue vs. average cost with campaign abbreviations as text and marketing channel as color
    fig = px.scatter(
        df_avg, 
        x="cost", 
        y="rev", 
        color="channel_type",
        text = "channel_type",
        size="roi",  # Use average ROI as the marker size
        animation_frame="year",
        title=f"Average Revenue vs. Cost vs ROI (Campaigns: {', '.join(campaigns)})",
        labels={"rev": "Average Revenue", "cost": "Average Cost", "channel_type": "Marketing Channel", "roi": "Average ROI (%)"},
        hover_data=["year", "roi"]
    )
    
    fig.update_traces(textposition='top center')  # Position the text above the markers
    
    fig.update_layout(
        sliders=[
            dict(
                currentvalue={"prefix": "Year: "},
                pad={"t": 50}
            )
        ]
    )
    
    return fig
