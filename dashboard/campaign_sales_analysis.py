import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from q3 import *

df = pd.read_csv('final_data.csv', low_memory=False)

# Data cleaning
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df['month'] = df['purchase_date'].dt.month  # Extract the month
df['year'] = df['purchase_date'].dt.year  # Extract the year
df['campaign_name'] = df['campaign_key'].str[5:8]  # Extract the campaign code

# Map campaign codes to actual campaign names
campaign_mapping = {
    'EST': 'Easter Sale',
    'MID': 'Mid-Year Sale',
    'HLW': 'Halloween',
    'XMS': 'Christmas',
    'STD': "St. Patrick's Day",
    'CMO': 'Cinco de Mayo',
    'DRK': 'Drinks Bonanza',
    'FBD': 'Food and Beverage Day',
    'SMS': 'Super Mart Sale',
    'MMS': 'Markdown Mega Sale'
}
df['campaign_name'] = df['campaign_name'].map(campaign_mapping)

# Select required columns
df = df[['customer_key', 'quantity_purchased', 'total_price', 'purchase_date', 'description', 'revenue', 'campaign_key', 'mkt_chnl_key','month', 'year', 'campaign_name']]
df['category'] = df['description'].apply(extract_category) # Add 'category' column

# For AOV analysis, filter out rows from 2021 onwards
df2 = df[df['year'] < 2021] 

df_campaign_by_month = aov_analysis(df2)[4]

# Streamlit App
st.title("AOV per Campaign Dashboard (2014 - 2020)")

# Select a year
selected_year = st.selectbox("Select a year", options=range(2014, 2021))

# Plot AOV bar chart for the selected year
st.write(f"### AOV per Campaign in {selected_year}")
fig = plot_aov_for_year(df_campaign_by_month, selected_year)
st.plotly_chart(fig)