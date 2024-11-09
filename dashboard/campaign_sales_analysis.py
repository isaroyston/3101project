import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from functions import *

df = pd.read_csv('final_data.csv', low_memory=False) # edit file name based on what is saved locally
df_mkt = pd.read_csv('mkt_channels_data.csv', low_memory=False) # edit file name based on what is saved locally

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
st.set_page_config(page_title="Ecommerce Dashboard", layout="wide")
st.title("Ecommerce Dashboard (2014 - 2020)")

# Calculate general statistics
total_sales = df['quantity_purchased'].sum()
total_revenue = df['revenue'].sum()

# Display header and metrics in a single row
st.header("General Statistics")
col1, col2 = st.columns(2)
col1.metric("Total Sales to Date", total_sales)
col2.metric("Total Revenue to Date", f"${total_revenue:,.2f}")

# Select a year
selected_year = st.selectbox("Select a year", options=range(2014, 2021))
metrics = campaign_eff_metrics(df, selected_year)


fig1 = sales_growth_dashboard(metrics[0],selected_year)
fig2 = plot_aov_for_year(df_campaign_by_month, selected_year)
fig3 = revenue_plot_dashboard(metrics[1], selected_year)
fig4 = transaction_plot_dashboard(metrics[2], selected_year)
fig5 = analyze_avg_roi_by_year(df_mkt, ["Easter Sale", "Mid-Year Sale", "Christmas"])
fig6 = plot_rev_cost_roi(df_mkt, ["Easter Sale", "Mid-Year Sale", "Christmas"])

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.plotly_chart(fig4, use_container_width=True)

col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    st.plotly_chart(fig5, use_container_width=True)
    st.plotly_chart(fig6, use_container_width=True)


