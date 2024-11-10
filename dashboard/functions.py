import math
import json
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet.serialize import model_from_json

def get_delivery_predictions(supplier_name,n_days):

    # payload = {
    #     "supplier_name": supplier_name,
    #     "n_days": int(n_days)
    # }
    #
    # res = requests.post(f"http://127.0.0.1:8000/Subgroup_B_Q3/make_predictions/", json=payload)
    # print("TYPE", type(res.json()))
    # res_dict = json.loads(res.json()['predictions'])

    with open(f'../Subgroup_B_Q3/models/serialized_model_{supplier_name}.json', 'r') as f:
        prophet_model = model_from_json(f.read())

    prophet_predicted = prophet_model.predict(
        prophet_model.make_future_dataframe(
            periods=n_days, include_history=False
        )
    )["yhat"].reset_index(drop=True)

    x = [datetime.today() + timedelta(days=i) for i in range(1, n_days + 1)]
    y = [i[1] for i in prophet_predicted.items()]

    fig = px.line(x=x, y=y)
    fig.add_hline(y=3, line_color='red')
    fig.update_layout(xaxis_title="Order placed on", yaxis_title="Expected time to deliver (days)", showlegend=False)

    df = pd.DataFrame({"Customer orders placed on":x, "Delivery expected to be late by (days)":y})
    df = df.loc[df["Delivery expected to be late by (days)"] > 3]

    df["Delivery expected to be late by (days)"] = df["Delivery expected to be late by (days)"].apply(math.ceil) - 3
    df["Stock supplies on"] = df["Customer orders placed on"] - pd.to_timedelta(df["Delivery expected to be late by (days)"]+3, unit='D')

    df["Customer orders placed on"] = df["Customer orders placed on"].dt.strftime('%Y-%m-%d')
    df["Stock supplies on"] = df["Stock supplies on"].dt.strftime('%Y-%m-%d')

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

"""
At-risk customers functions (q2)
"""

def annual_individual_clv(df, selected_year):
    '''
    This function calculates the annual CLV of each individual customer for the selected year.
    We output a dataframe with an annual_CLV column for each customer in the selected year.
    '''
    # Filter the data for the selected year
    year_data = df[df['year'] == selected_year]

    # Calculate total revenue, number of purchases, and customer lifespan for each customer for the selected year
    customer_data = year_data.groupby(['customer_key', 'year']).agg(
        total_revenue=('revenue', 'sum'),              # Total revenue per customer for the selected year
        total_purchases=('revenue', 'count'),          # Total number of purchases per customer for the selected year
        first_purchase=('purchase_date', 'min'),       # Date of first purchase within the year
        last_purchase=('purchase_date', 'max')         # Date of last purchase within the year
    ).reset_index()

    # Calculate the annual customer lifespan (in years) based on the first and last purchase dates within the selected year
    customer_data['annual_lifespan'] = (
        (customer_data['last_purchase'] - customer_data['first_purchase']).dt.days
    ) / 365.25

    # Calculate annual CLV using the formula for the selected year per customer
    customer_data['annual_CLV'] = customer_data['total_revenue'] * (
        customer_data['annual_lifespan'] / customer_data['total_purchases']
    )

    return customer_data

def low_clv(customer_data, selected_year):
    '''
    This function calculates the low CLV threshold for the selected year as the lower quartile of CLV.
    It outputs a DataFrame of low CLV customers for the selected year.
    '''
    # Filter customer data for the selected year
    year_data = customer_data[customer_data['year'] == selected_year]

    # Calculate LQ CLV threshold for the selected year
    low_clv_threshold = year_data['annual_CLV'].quantile(0.25)

    # Mark customers below the threshold as low CLV
    low_clv_customers = year_data[year_data['annual_CLV'] <= low_clv_threshold].copy()
    low_clv_customers['low_clv_threshold'] = low_clv_threshold

    return low_clv_customers

def at_risk_customers_for_selected_year(low_clv_customers):
    '''
    This function identifies the at-risk customers for the selected year.
    At-risk criteria:
    1. Below LQ annual CLV
    2. Below mean purchase frequency of that year
    '''
    # Calculate the mean purchase frequency for the selected year
    mean_purchase_freq = low_clv_customers['total_purchases'].mean()

    # Identify at-risk customers based on the defined criteria
    at_risk_customers = low_clv_customers[
        low_clv_customers['total_purchases'] < mean_purchase_freq
    ].reset_index(drop=True)

    return at_risk_customers
        
"""
AOV functions (q3)
"""
def extract_category(description):
    desc = description.strip()
    # Remove 'a. ' prefix if present
    if desc.startswith('a. '):
        desc = desc[3:].strip()
    # If ' - ' exists, take the part before it
    if ' - ' in desc:
        category = desc.split(' - ')[0].strip()
    else:
        # Special cases for descriptions without ' - '
        if desc.startswith('Coffee'):
            category = 'Coffee Products'
        elif desc == 'Medicine':
            category = 'Medicine'
        elif desc == 'Kitchen Supplies':
            category = 'Kitchen Supplies'
        else:
            category = desc.split()[0].strip()
    return category

def aov_analysis(df):
    '''
    Calculates AOV for campaign and non-campaign periods, specific campaigns per month and year
    '''
    
    # Filter for campaign-related sales
    df_campaign = df[df['mkt_chnl_key'].notnull()]

    # Filter for non-campaign sales
    df_non_campaign = df[df['mkt_chnl_key'].isnull()]

    # Average Order Value (AOV) during Campaign periods
    campaign_aov = df_campaign['revenue'].sum() / len(df_campaign)

    # AOV during Non-Campaign periods
    non_campaign_aov = df_non_campaign['revenue'].sum() / len(df_non_campaign)

    # Group campaign sales by year and campaign
    df_campaign_grouped = df_campaign.groupby(['year', 'campaign_name']).agg({
        'revenue': 'sum',        # Total revenue for campaign purchases
        'customer_key': 'count'  # Counting number of transactions (rows)
    }).reset_index()

    # Group campaign sales by year, month and campaign
    df_campaign_grouped_monthly = df_campaign.groupby(['year', 'month', 'campaign_name']).agg({
        'revenue': 'sum',        # Total revenue for campaign purchases
        'customer_key': 'count'  # Counting number of transactions (rows)
    }).reset_index()

    # Calculate AOV for each campaign
    df_campaign_grouped['aov'] = df_campaign_grouped['revenue'] / df_campaign_grouped['customer_key']

    # Calculate AOV for each campaign by month
    df_campaign_grouped_monthly['aov'] = df_campaign_grouped_monthly['revenue'] / df_campaign_grouped_monthly['customer_key']
    
    # Sort by month within each year for proper ordering in plots
    df_campaign_grouped_monthly = df_campaign_grouped_monthly.sort_values(by=['year', 'month'])

    # Group non-campaign sales by year
    df_non_campaign_grouped = df_non_campaign.groupby('year').agg({
        'revenue': 'sum',        # Total revenue for non-campaign purchases
        'customer_key': 'count'  # Counting number of transactions (rows)
    }).reset_index()

    # Calculate AOV for non-campaign periods
    df_non_campaign_grouped['aov'] = df_non_campaign_grouped['revenue'] / df_non_campaign_grouped['customer_key']

    return df_campaign, df_non_campaign, df_campaign_grouped, df_non_campaign_grouped, df_campaign_grouped_monthly

def campaign_eff_metrics(df, chosen_year):
    '''
    Calculates the metrics for evaluating campaign effectiveness in the chosen year.
    '''

    # Separate campaign and non-campaign data
    # Filter campaign data for the chosen year
    campaign_data = df[df['mkt_chnl_key'].notnull() & (df['purchase_date'].dt.year == chosen_year)]
    non_campaign_data = df[df['mkt_chnl_key'].isnull() & (df['purchase_date'].dt.year == chosen_year)]

    # Prepare lists to store results for revenue, transactions, aos, aov and growth rates
    revenue = []
    transactions = []
    aos = []
    aov = []
    aov_by_category = []
    growth_rates = []

    # Loop through each campaign for the chosen year to calculate before, during, and after campaign stats
    for campaign in campaign_data['campaign_name'].unique():
        # Filter data for the current campaign
        current_campaign_data = df[(df['campaign_name'] == campaign) & 
                                (df['purchase_date'].dt.year == chosen_year)]

        # Get campaign start and end dates
        campaign_start = current_campaign_data['purchase_date'].min()
        campaign_end = current_campaign_data['purchase_date'].max()

        # Calculate the campaign duration
        campaign_duration = campaign_end - campaign_start

        # Define the period "before the campaign" matching the campaign duration
        before_campaign_data = df[(df['purchase_date'] >= campaign_start - campaign_duration) &
                                (df['purchase_date'] < campaign_start) &
                                (df['purchase_date'].dt.year == chosen_year)]
        
        # Catch Christmas data (after-campaign is Jan next year)
        # But for 2020, we don't have full next 1 month data (dataset ends 19/01/21)
        if campaign == "Christmas": 
            after_campaign_data = df[(df['purchase_date'].dt.month == 1) &
                                 (df['purchase_date'].dt.year == chosen_year + 1)]

        else:
            # Define the period "after the campaign" matching the campaign duration
            after_campaign_data = df[(df['purchase_date'] > campaign_end) &
                                (df['purchase_date'] <= campaign_end + campaign_duration) &
                                (df['purchase_date'].dt.year == chosen_year)]

        # Calculate revenue before, during, and after the campaign
        rev_before_campaign = before_campaign_data['revenue'].sum()
        rev_during_campaign = current_campaign_data['revenue'].sum()
        rev_after_campaign = after_campaign_data['revenue'].sum()
        
        # Count transactions before, during, and after the campaign
        trans_before_campaign = before_campaign_data['customer_key'].count()
        trans_during_campaign = current_campaign_data['customer_key'].count()
        trans_after_campaign = after_campaign_data['customer_key'].count()

        # Calculate AOS for before, during, and after the campaign
        aos_before_campaign = before_campaign_data['quantity_purchased'].sum() / before_campaign_data['customer_key'].count()
        aos_during_campaign = current_campaign_data['quantity_purchased'].sum() / current_campaign_data['customer_key'].count()
        aos_after_campaign = after_campaign_data['quantity_purchased'].sum() / after_campaign_data['customer_key'].count()

        # Calculate AOV for before, during, and after the campaign
        aov_before_campaign = before_campaign_data['revenue'].sum() / before_campaign_data['customer_key'].count()
        aov_during_campaign = current_campaign_data['revenue'].sum() / current_campaign_data['customer_key'].count()
        aov_after_campaign = after_campaign_data['revenue'].sum() / after_campaign_data['customer_key'].count()

        # Calculate AOV by category for each phase and append results to aov_by_category list
        def calculate_aov_cat(data, phase):
            if data.empty:
                return pd.DataFrame(columns=['category', 'AOV', 'Phase', 'Campaign'])
            aov_data = data.groupby('category').apply(
                lambda x: x['revenue'].sum() / x['customer_key'].count() if x['customer_key'].count() > 0 else None
            ).reset_index(name='AOV')
            aov_data['Phase'] = phase
            aov_data['Campaign'] = campaign
            return aov_data

        aov_category_before = calculate_aov_cat(before_campaign_data, 'before')
        aov_category_during = calculate_aov_cat(current_campaign_data, 'during')
        aov_category_after = calculate_aov_cat(after_campaign_data, 'after')

        # Calculate sales before, during, and after the campaign
        sales_before_campaign = before_campaign_data['total_price'].sum()
        sales_during_campaign = current_campaign_data['total_price'].sum()

        # Calculate sales growth rate (before and during)
        if sales_before_campaign > 0:
            growth_rate = ((sales_during_campaign - sales_before_campaign) / sales_before_campaign) * 100
        else:
            growth_rate = None  # Avoid division by zero if there were no sales before the campaign

        # Append revenue data for this campaign
        revenue.append({
            'Campaign': campaign,
            'Revenue before Campaign': rev_before_campaign,
            'Revenue during Campaign': rev_during_campaign,
            'Revenue after Campaign': rev_after_campaign
        })

        # Append transactions data for this campaign
        transactions.append({
            'Campaign': campaign,
            'No. of Transactions before Campaign': trans_before_campaign,
            'No. of Transactions during Campaign': trans_during_campaign,
            'No. of Transactions after Campaign': trans_after_campaign
        })

        # Append AOS data
        aos.append({
            'Campaign': campaign,
            'AOS before Campaign': aos_before_campaign,
            'AOS during Campaign': aos_during_campaign,
            'AOS after Campaign': aos_after_campaign
        })

        # Append AOV data
        aov.append({
            'Campaign': campaign,
            'AOV before Campaign': aov_before_campaign,
            'AOV during Campaign': aov_during_campaign,
            'AOV after Campaign': aov_after_campaign
        })

        # Append AOV by category data
        aov_by_category.extend([aov_category_before, aov_category_during, aov_category_after])

        # Append sales growth rate data
        growth_rates.append({
            'Campaign': campaign,
            'Sales before Campaign': sales_before_campaign,
            'Sales during Campaign': sales_during_campaign,
            'Growth Rate (%)': growth_rate
        })

    # Create DataFrames for revenue, transactions, aos, aov, aov by category, and growth rates
    rev_df = pd.DataFrame(revenue)
    trans_df = pd.DataFrame(transactions)
    aos_df = pd.DataFrame(aos)
    aov_df = pd.DataFrame(aov)
    aov_by_category_df = pd.concat(aov_by_category, ignore_index=True)
    growth_df = pd.DataFrame(growth_rates)

    return growth_df, rev_df, trans_df, aos_df, aov_df, aov_by_category_df

def sales_growth_dashboard(growth_df, chosen_year):
    '''
    Outputs sales growth rate bar plot for each campaign for the chosen year.
    '''

    # Sales Growth Rate plot
    fig_growth = go.Figure(data=[
        go.Bar(name='Growth Rate (%)', x=growth_df['Campaign'], y=growth_df['Growth Rate (%)'], marker_color='pink')
    ])

    # Update layout for growth rate plot
    fig_growth.update_layout(
        title=f'Sales Growth Rate During Campaigns in {chosen_year}',
        xaxis_title='Campaign',
        yaxis_title='Growth Rate (%)',
        height=500,
        width=900,
        showlegend=False
    )

    return fig_growth # Return instead of show for dashboard

def plot_aov_for_year(df_campaign_grouped_monthly, selected_year):
    '''
    Plots bar graph of AOV per campaign for the selected year to be used in dashboard.
    '''
    # Filter data for the selected year
    year_data = df_campaign_grouped_monthly[df_campaign_grouped_monthly['year'] == selected_year]
    
    # Create bar chart
    fig = go.Figure(
        data=go.Bar(
            x=year_data['campaign_name'],
            y=year_data['aov'],
            marker_color='lightgoldenrodyellow',
            hovertemplate="AOV: %{y:.2f}<extra></extra>"
        )
    )

    # Update layout for readability
    fig.update_layout(
        title=f"Average Order Value (AOV) per Campaign in {selected_year}",
        xaxis_title="Campaign",
        yaxis_title="Average Order Value (AOV)",
        height=500,
        width=900
    )

    return fig

def revenue_plot_dashboard(rev_df, chosen_year):
    '''
    Outputs a bar plot for Revenue before, during, and after each campaign for the chosen year.
    '''
    # Create a figure for Revenue plot
    fig_revenue = go.Figure()

    # Add Revenue bars
    fig_revenue.add_trace(go.Bar(name='Revenue before Campaign', x=rev_df['Campaign'], y=rev_df['Revenue before Campaign'], 
                                 marker_color='lightblue'))
    fig_revenue.add_trace(go.Bar(name='Revenue during Campaign', x=rev_df['Campaign'], y=rev_df['Revenue during Campaign'], 
                                 marker_color='blue'))
    fig_revenue.add_trace(go.Bar(name='Revenue after Campaign', x=rev_df['Campaign'], y=rev_df['Revenue after Campaign'], 
                                 marker_color='darkblue'))

    # Update layout for the Revenue figure
    fig_revenue.update_layout(
        title=f'Revenue Comparison for {chosen_year}',
        xaxis_title="Campaign",
        yaxis_title="Revenue",
        height=500,
        width=900,
        barmode='group',
        showlegend=True
    )

    return fig_revenue


def transaction_plot_dashboard(trans_df, chosen_year):
    '''
    Outputs a bar plot for Transactions before, during, and after each campaign for the chosen year.
    '''
    # Create a figure for Transaction plot
    fig_transactions = go.Figure()

    # Add Transaction bars
    fig_transactions.add_trace(go.Bar(name='Transactions before Campaign', x=trans_df['Campaign'], y=trans_df['No. of Transactions before Campaign'], 
                                      marker_color='lightgreen'))
    fig_transactions.add_trace(go.Bar(name='Transactions during Campaign', x=trans_df['Campaign'], y=trans_df['No. of Transactions during Campaign'], 
                                      marker_color='green'))
    fig_transactions.add_trace(go.Bar(name='Transactions after Campaign', x=trans_df['Campaign'], y=trans_df['No. of Transactions after Campaign'], 
                                      marker_color='darkgreen'))

    # Update layout for the Transaction figure
    fig_transactions.update_layout(
        title=f'Transaction Comparison for {chosen_year}',
        xaxis_title="Campaign",
        yaxis_title="Number of Transactions",
        height=500,
        width=900,
        barmode='group',
        showlegend=True
    )

    return fig_transactions

def plot_average_monthly_sales(data):
    data['purchase_date'] = pd.to_datetime(data['purchase_date'], errors='coerce')
    data['year'] = data['purchase_date'].dt.year
    data['month'] = data['purchase_date'].dt.month
    average_monthly_sales = data.groupby('month')['total_price'].sum() / data['year'].nunique()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=average_monthly_sales.index,
        y=average_monthly_sales.values,
        mode='lines+markers',
        marker=dict(symbol='circle')
    ))
    fig.update_layout(
        title='Average Monthly Sales Throughout the Year',
        xaxis_title='Month',
        yaxis_title='Average Monthly Sales',
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13))),
        yaxis=dict(tickformat=','),
        template='plotly_white'
    )
    return fig

def plot_average_monthly_revenue(data):
    data['purchase_date'] = pd.to_datetime(data['purchase_date'], errors='coerce')
    data['year'] = data['purchase_date'].dt.year
    data['month'] = data['purchase_date'].dt.month
    monthly_revenue = data.groupby(['year', 'month'])['revenue'].sum().reset_index()
    average_monthly_revenue = monthly_revenue.groupby('month')['revenue'].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=average_monthly_revenue.index,
        y=average_monthly_revenue.values,
        mode='lines+markers',
        marker=dict(symbol='circle')
    ))
    fig.update_layout(
        title='Average Monthly Revenue Throughout the Year',
        xaxis_title='Month',
        yaxis_title='Average Monthly Revenue',
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13))),
        yaxis=dict(tickformat=','),
        template='plotly_white'
    )
    return fig

def plot_market_share_by_category(data):
    data['purchase_date'] = pd.to_datetime(data['purchase_date'], errors='coerce')
    data['year'] = data['purchase_date'].dt.year
    data['month'] = data['purchase_date'].dt.month
    def extract_category(description):
        desc = description.strip()
        if desc.startswith('a. '):
            desc = desc[3:].strip()
        if ' - ' in desc:
            category = desc.split(' - ')[0].strip()
        else:
            if desc.startswith('Coffee'):
                category = 'Coffee products'
            elif desc == 'Medicine':
                category = 'Medicine'
            elif desc == 'Kitchen Supplies':
                category = 'Kitchen Supplies'
            else:
                category = desc.split()[0].strip()
        return category

    data['Category'] = data['description'].apply(extract_category)
    category_yearly_revenue = data.groupby(['Category', 'year'])['revenue'].sum().reset_index()
    category_total_revenue = data.groupby('Category')['revenue'].sum().reset_index()
    category_total_revenue = category_total_revenue.sort_values(by='revenue', ascending=False)
    total_revenue = category_total_revenue['revenue'].sum()
    category_total_revenue['Market_Share'] = (category_total_revenue['revenue'] / total_revenue) * 100
    labels = category_total_revenue['Category']
    sizes = category_total_revenue['Market_Share']
    explode = [0.05 if (size >= 15) else 0 for size in sizes]
    colors = sns.color_palette('tab10', n_colors=len(labels)).as_hex()
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels,
        values=sizes,
        hole=0.3,
        marker=dict(colors=colors, line=dict(color='white', width=1)),
        textinfo='label+percent',
        textposition='inside',
        insidetextorientation='radial',
        pull=explode
    ))
    fig.update_layout(
        title='Market Share by Category',
        showlegend=False
    )
    return fig

def plot_monthly_sales_by_year(data):
    data['purchase_date'] = pd.to_datetime(data['purchase_date'], errors='coerce')
    data['year'] = data['purchase_date'].dt.year
    data['month'] = data['purchase_date'].dt.month
    data = data[data['year'] != 2021]
    monthly_sales = data.groupby(['year', 'month'])['total_price'].sum().reset_index()
    monthly_sales_pivot = monthly_sales.pivot(index='month', columns='year', values='total_price')

    fig = go.Figure()
    for year in monthly_sales_pivot.columns:
        fig.add_trace(go.Scatter(
            x=monthly_sales_pivot.index,
            y=monthly_sales_pivot[year],
            mode='lines+markers',
            name=str(year)
        ))
    fig.update_layout(
        title='Monthly Sales by Year',
        xaxis_title='Month',
        yaxis_title='Total Monthly Sales',
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13))),
        yaxis=dict(tickformat=','),
        template='plotly_white',
        legend_title='Year'
    )
    return fig

def plot_monthly_sales_trends(data, top_n=5):
    data['purchase_date'] = pd.to_datetime(data['purchase_date'], errors='coerce')
    data['year'] = data['purchase_date'].dt.year
    data['month'] = data['purchase_date'].dt.month
    # Identify top products
    top_products = data.groupby('item_name')['total_price'].sum().nlargest(top_n).index

    # Create a modified 'item_name' column with 'Others' for non-top products
    data['item_name_mod'] = data['item_name'].where(data['item_name'].isin(top_products), 'Others')

    # Prepare monthly sales data for top products and 'Others'
    monthly_sales_pivot = (
        data.groupby(['month', 'item_name_mod'])['total_price']
        .sum()
        .unstack()
        .fillna(0)
    )

    # Create the Plotly figure
    fig = go.Figure()

    for column in monthly_sales_pivot.columns:
        fig.add_trace(go.Scatter(
            x=monthly_sales_pivot.index,
            y=monthly_sales_pivot[column],
            mode='lines+markers',
            name=column
        ))

    fig.update_layout(
        title='Monthly Sales Trends for Top Products and Others',
        xaxis_title='Month',
        yaxis_title='Total Sales',
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13))),
        yaxis=dict(tickformat=','),
        template='plotly_white',
        legend_title='Product'
    )
    return fig

def plot_top_consistent_sellers(data, n):
    top_monthly_products = (
        data.groupby(['year', 'month', 'item_name'])['total_price']
        .sum()
        .reset_index()
        .sort_values(['year', 'month', 'total_price'], ascending=[True, True, False])
        .groupby(['year', 'month'])
        .head(n)
    )
    consistent_top_sellers = (
        top_monthly_products.groupby('item_name')
        .size()
        .reset_index(name=f'frequency_in_top_{n}')
        .sort_values(by=f'frequency_in_top_{n}', ascending=False)
        .head(10)
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=consistent_top_sellers['item_name'],
        y=consistent_top_sellers[f'frequency_in_top_{n}'],
        marker_color='skyblue'
    ))

    fig.update_layout(
        title=f"Top 10 Products by Frequency in Monthly Top {n}",
        xaxis_title="Product Name",
        yaxis_title=f"Months in Top {n}",
        xaxis=dict(tickangle=45),
        template='plotly_white'
    )
    return fig

def plot_monthly_sales_and_decomposition(data):
    data['purchase_date'] = pd.to_datetime(data['purchase_date'], errors='coerce')
    data['year'] = data['purchase_date'].dt.year
    data['month'] = data['purchase_date'].dt.month
    # If 'purchase_date' is missing, identify the correct column name
    if 'purchase_date' not in data.columns:
        # Let's assume the correct column name is 'PurchaseDate'
        if 'PurchaseDate' in data.columns:
            data.rename(columns={'PurchaseDate': 'purchase_date'}, inplace=True)
        else:
            print("Error: 'purchase_date' column not found in the DataFrame.")
            print("Available columns:", data.columns.tolist())
            # Exit or raise an error to prevent further issues
            raise KeyError("'purchase_date' column not found in the DataFrame.")

    # Now proceed with the data preprocessing
    data['purchase_date'] = pd.to_datetime(data['purchase_date'], errors='coerce')
    data = data[data['purchase_date'] <= '2020-12-31']
    data = data.dropna(subset=['total_price', 'purchase_date'])
    data.set_index('purchase_date', inplace=True)
    data['total_price'] = pd.to_numeric(data['total_price'], errors='coerce')

    # Proceed with the seasonal decomposition
    monthly_sales = data['total_price'].resample('M').sum()

    # Plot monthly sales
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_sales.index,
        y=monthly_sales.values,
        mode='lines+markers',
        marker=dict(symbol='circle'),
        name='Total Sales'
    ))

    fig.update_layout(
        title='Monthly Total Sales Over Time (Up to 2020)',
        xaxis_title='Date',
        yaxis_title='Total Sales',
        template='plotly_white'
    )

    return fig

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(monthly_sales, model='additive')

    # Plot the decomposition
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decomposition.trend.index,
        y=decomposition.trend,
        mode='lines',
        name='Trend'
    ))
    fig.add_trace(go.Scatter(
        x=decomposition.seasonal.index,
        y=decomposition.seasonal,
        mode='lines',
        name='Seasonal'
    ))
    fig.add_trace(go.Scatter(
        x=decomposition.resid.index,
        y=decomposition.resid,
        mode='lines',
        name='Residual'
    ))
    fig.update_layout(
        title='Seasonal Decomposition of Monthly Sales',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white'
    )
    return fig

def plot_revenue_treemap(data):
    data['purchase_date'] = pd.to_datetime(data['purchase_date'], errors='coerce')
    data['year'] = data['purchase_date'].dt.year
    data['month'] = data['purchase_date'].dt.month
    # Step 1: Data Preparation
    data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')
    data_clean = data.dropna(subset=['store_region', 'store_district', 'store_sub_district', 'revenue'])

    # Step 2: Build Hierarchical Data

    # Sub-Districts
    sub_districts = data_clean.groupby(['store_region', 'store_district', 'store_sub_district'], as_index=False)['revenue'].sum()
    sub_districts['id'] = sub_districts['store_region'] + '/' + sub_districts['store_district'] + '/' + sub_districts['store_sub_district']
    sub_districts['parent'] = sub_districts['store_region'] + '/' + sub_districts['store_district']
    sub_districts['label'] = sub_districts['store_sub_district']
    sub_districts['level'] = 2

    # Districts
    districts = data_clean.groupby(['store_region', 'store_district'], as_index=False)['revenue'].sum()
    districts['id'] = districts['store_region'] + '/' + districts['store_district']
    districts['parent'] = districts['store_region']
    districts['label'] = districts['store_district']
    districts['level'] = 1

    # Regions
    regions = data_clean.groupby(['store_region'], as_index=False)['revenue'].sum()
    regions['id'] = regions['store_region']
    regions['parent'] = 'All Regions'
    regions['label'] = regions['store_region']
    regions['level'] = 0

    # Root Node
    root = pd.DataFrame({
        'id': ['All Regions'],
        'parent': [''],
        'label': ['All Regions'],
        'revenue': [regions['revenue'].sum()],
        'level': -1  # Root level
    })

    # Combine DataFrames
    all_data = pd.concat(
        [root[['id', 'parent', 'label', 'revenue', 'level']],
         regions[['id', 'parent', 'label', 'revenue', 'level']],
         districts[['id', 'parent', 'label', 'revenue', 'level']],
         sub_districts[['id', 'parent', 'label', 'revenue', 'level']]],
        ignore_index=True
    )

    # Step 3: Create the Treemap
    fig = go.Figure(go.Treemap(
        labels=all_data['label'],
        parents=all_data['parent'],
        ids=all_data['id'],
        values=all_data['revenue'],
        branchvalues='total',
        maxdepth=2,  # Initial view shows up to districts
        textinfo='label+value',
        hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<extra></extra>',
        marker=dict(
            colors=all_data['revenue'],
            colorscale='ylgn',
            colorbar=dict(title='Revenue'),
        )))
    # Step 4: Update Layout and Traces
    fig.update_layout(
        title='Revenue by Store Region, District, and Sub-District',
        margin=dict(t=50, l=25, r=25, b=25)
    )
    # Adjust font sizes and colors
    fig.update_traces(
        insidetextfont=dict(size=14, color='black'),
        selector=dict(type='treemap')
    )
    # Step 5: Display the Treemap
    return fig