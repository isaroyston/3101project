import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from functions import *

df = pd.read_csv('final.csv', low_memory=False) # edit file name based on what is saved locally
df_mkt = pd.read_csv('mkt_channels.csv', low_memory=False) # edit file name based on what is saved locally

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
    'FBD': 'Discount Daze',
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

# Select a year
selected_year = st.selectbox("Select a year", options=range(2014, 2021))
metrics = campaign_eff_metrics(df, selected_year)

# Filter the DataFrame for the selected year
df_year = df[df['year'] == selected_year]

# Calculate metrics
total_sales = df_year['total_price'].sum()
total_revenue = df_year['revenue'].sum()
no_trans = df_year['customer_key'].count()
no_customers = df_year['customer_key'].nunique()

# Calculate annual CLV for customers in the selected year
customer_clv_data = annual_individual_clv(df, selected_year)

# Determine low CLV customers for the selected year
low_clv_customers = low_clv(customer_clv_data, selected_year)

# Get at-risk customers for the selected year
at_risk_customers_df = at_risk_customers_for_selected_year(low_clv_customers)

# Calculate mean annual CLV
mean_annual_clv = customer_clv_data['annual_CLV'].mean()

# Display key sales metrics in a single row with icons
st.subheader("Key Sales Metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
col2.metric("📈 Total Revenue", f"${total_revenue:,.2f}")
col3.metric("🛒 No. of Transactions", f"{no_trans:,}")
col4.metric("👥 No. of Customers", f"{no_customers:,}")
col5.metric("💵 Mean Annual CLV", f"${mean_annual_clv:,.2f}")

# # Display at-risk customers
# st.subheader("Annual CLV Analysis")
# st.markdown("**At-Risk Customers by CLV and Purchase Frequency:**")
# st.dataframe(at_risk_customers_df[['customer_key', 'annual_CLV', 'total_purchases']])
# st.write(f"Number of at-risk customers for {selected_year}: {at_risk_customers_df.shape[0]}")

with st.expander("View At-Risk Customers Data"):
    st.dataframe(at_risk_customers_df[['customer_key', 'annual_CLV', 'low_clv_threshold', 'total_purchases']])
    st.write(f"Number of at-risk customers for {selected_year}: {at_risk_customers_df.shape[0]}")

fig1 = sales_growth_dashboard(metrics[0],selected_year)
fig2 = plot_aov_for_year(df_campaign_by_month, selected_year)
fig3 = revenue_plot_dashboard(metrics[1], selected_year)
fig4 = transaction_plot_dashboard(metrics[2], selected_year)
fig5 = analyze_avg_roi_by_year(df_mkt, ["Easter Sale", "Mid-Year Sale", "Christmas"])
fig6 = plot_rev_cost_roi(df_mkt, ["Easter Sale", "Mid-Year Sale", "Christmas"])

# Display plots in rows of two
st.subheader("Campaign Performance")
with st.container():
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    col3.plotly_chart(fig3, use_container_width=True)
    col4.plotly_chart(fig4, use_container_width=True)

st.subheader("ROI and Cost Analysis")
st.plotly_chart(fig5, use_container_width=True)
st.plotly_chart(fig6, use_container_width=True)

# with st.container():
#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(fig1, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig2, use_container_width=True)

#     col3, col4 = st.columns(2)
#     with col3:
#         st.plotly_chart(fig3, use_container_width=True)
#     with col4:
#         st.plotly_chart(fig4, use_container_width=True)

# col_left, col_center, col_right = st.columns([1, 2, 1])
# with col_center:
#     st.plotly_chart(fig5, use_container_width=True)
#     st.plotly_chart(fig6, use_container_width=True)


