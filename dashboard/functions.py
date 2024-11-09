import json
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
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
            category = 'Coffee products'
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
        title=f'Sales Growth Rate During Campaigns for {chosen_year}',
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
        title=f"AOV per Campaign in {selected_year}",
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
