import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

"""
ROI analysis functions
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
    
    fig.show()

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
    
    fig.show()

"""
(Part 2.1) AOV Analysis functions 
"""
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

def aov_yearly(df_campaign, df_non_campaign):
    df_campaign_yearly = df_campaign.groupby('year').agg({
        'revenue': 'sum',      # Total revenue for campaigns
        'customer_key': 'count' # Number of transactions (rows)
    }).reset_index()

    # Calculate AOV for campaign periods
    df_campaign_yearly['campaign_aov'] = df_campaign_yearly['revenue'] / df_campaign_yearly['customer_key']

    # Non-Campaign Yearly Data: Group by year
    df_non_campaign_yearly = df_non_campaign.groupby('year').agg({
        'revenue': 'sum',      # Total revenue for non-campaign purchases
        'customer_key': 'count' # Number of transactions (rows)
    }).reset_index()

    # Calculate AOV for non-campaign periods
    df_non_campaign_yearly['non_campaign_aov'] = df_non_campaign_yearly['revenue'] / df_non_campaign_yearly['customer_key']

    # Merge campaign and non-campaign AOV data by year
    df_comparison = pd.merge(df_campaign_yearly[['year', 'campaign_aov']], 
                            df_non_campaign_yearly[['year', 'non_campaign_aov']], 
                            on='year', how='inner')
    
    return df_comparison

def aov_yearly_plot(df_comparison):
    '''
    Plots Campaign VS Non-Campaign AOV overall trend.
    '''
    
    # Create a line plot for Campaign AOV vs Non-Campaign AOV over years
    fig = go.Figure()

    # Add Campaign AOV line
    fig.add_trace(go.Scatter(
        x=df_comparison['year'], 
        y=df_comparison['campaign_aov'],
        mode='lines+markers',
        name='Campaign',
        marker=dict(symbol='circle', color='blue'),
        line=dict(color='blue')
    ))

    # Add Non-Campaign AOV line
    fig.add_trace(go.Scatter(
        x=df_comparison['year'], 
        y=df_comparison['non_campaign_aov'],
        mode='lines+markers',
        name='Non-Campaign',
        marker=dict(symbol='square', color='green'),
        line=dict(color='green')
    ))

    # Update layout for the plot
    fig.update_layout(
        title='Campaign vs Non-Campaign AOV Over Time',
        xaxis_title='Year',
        yaxis_title='Average Order Value (AOV)',
        template='plotly_white',
        width=800,
        height=500,
    )

    # Show grid lines
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    fig.show()

def aov_campaign_plot(df_campaign_grouped, df_non_campaign_grouped):
    '''
    Plots Specific Campaign AOV vs Non-Campaign AOV trend.
    '''
    
    # Initialize a figure
    fig = go.Figure()

    # Plot Non-Campaign AOV (Baseline)
    fig.add_trace(go.Scatter(
        x=df_non_campaign_grouped['year'], 
        y=df_non_campaign_grouped['aov'],
        mode='lines+markers',
        name='Non-Campaign (Baseline)',
        line=dict(color='black', dash='dash'),
        marker=dict(symbol='square', size=8),
        hovertemplate='Year: %{x}<br>AOV: %{y}<extra></extra>'
    ))

    # Plot AOV for each campaign over time
    for campaign in df_campaign_grouped['campaign_name'].unique():
        campaign_data = df_campaign_grouped[df_campaign_grouped['campaign_name'] == campaign]
        fig.add_trace(go.Scatter(
            x=campaign_data['year'],
            y=campaign_data['aov'],
            mode='lines+markers',
            name=f'{campaign}',
            marker=dict(size=6),
            hovertemplate='Year: %{x}<br>AOV: %{y}<br>Campaign: ' + campaign + '<extra></extra>'
        ))

    # Update layout for better presentation
    fig.update_layout(
        title='Specific Campaign vs Non-Campaign AOV Over Time',
        xaxis_title='Year',
        yaxis_title='Average Order Value (AOV)',
        legend_title='Campaigns',
        hovermode='x',  # Hover effect for all campaigns when hovering over the same year
        template='plotly_white',
        width=900,
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    # Show the plot
    fig.show()

"""
(Part 2.2) Sales Growth Rate, Revenue, Transactions, AOS, AOV Before/During/After Campaign 
"""
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

def sales_growth_rate_plot(growth_df, chosen_year):
    '''
    Outputs sales growth rate bar plot for each campaign for the chosen year.
    '''

    # Sales Growth Rate plot
    fig_growth = go.Figure(data=[
        go.Bar(name='Growth Rate (%)', x=growth_df['Campaign'], y=growth_df['Growth Rate (%)'], marker_color='orange')
    ])

    # Update layout for growth rate plot
    fig_growth.update_layout(
        title=f'Sales Growth Rate During Campaigns for {chosen_year}',
        xaxis_title='Campaign',
        yaxis_title='Growth Rate (%)',
        height=600,
        width=900,
        xaxis=dict(tickangle=45), 
        showlegend=False
    )

    fig_growth.show()

def revenue_transaction_plot(rev_df, trans_df, chosen_year):
    '''
    Outputs a figure with two separate grouped bar plots (Revenue and Transactions)
    for each campaign for the chosen year, displayed top and bottom.
    '''
    
    # Create subplot structure with two rows and one column
    fig = make_subplots(rows=2, cols=1, subplot_titles=(f'Revenue Comparison for {chosen_year}', f'Transaction Comparison for {chosen_year}'))

    # Add Revenue bars (top plot)
    fig.add_trace(go.Bar(name='Revenue before Campaign', x=rev_df['Campaign'], y=rev_df['Revenue before Campaign'], 
                         marker_color='lightblue'), row=1, col=1)
    fig.add_trace(go.Bar(name='Revenue during Campaign', x=rev_df['Campaign'], y=rev_df['Revenue during Campaign'], 
                         marker_color='blue'), row=1, col=1)
    fig.add_trace(go.Bar(name='Revenue after Campaign', x=rev_df['Campaign'], y=rev_df['Revenue after Campaign'], 
                         marker_color='darkblue'), row=1, col=1)
    
    # Add Transaction bars (bottom plot)
    fig.add_trace(go.Bar(name='Transactions before Campaign', x=trans_df['Campaign'], y=trans_df['No. of Transactions before Campaign'], 
                         marker_color='lightgreen'), row=2, col=1)
    fig.add_trace(go.Bar(name='Transactions during Campaign', x=trans_df['Campaign'], y=trans_df['No. of Transactions during Campaign'], 
                         marker_color='green'), row=2, col=1)
    fig.add_trace(go.Bar(name='Transactions after Campaign', x=trans_df['Campaign'], y=trans_df['No. of Transactions after Campaign'], 
                         marker_color='darkgreen'), row=2, col=1)

    # Update layout for the combined figure
    fig.update_layout(
        height=750, width=850,
        barmode='group',
        showlegend=True
    )
    
    # Set axis titles
    fig.update_xaxes(title_text="Campaign", row=1, col=1)
    fig.update_yaxes(title_text="Revenue", row=1, col=1)
    fig.update_xaxes(title_text="Campaign", row=2, col=1)
    fig.update_yaxes(title_text="Number of Transactions", row=2, col=1)

    # Show the plot
    fig.show()

def aos_aov_plot(aos_df, aov_df, chosen_year):
    '''
    Outputs a figure with two separate grouped bar plots (AOS and AOV)
    for each campaign for the chosen year, displayed top and bottom.
    '''
    
    # Create subplot structure with two rows and one column
    fig = make_subplots(rows=2, cols=1, subplot_titles=(f'AOS Comparison for {chosen_year}', f'AOV Comparison for {chosen_year}'))

    # Add AOS bars (top plot)
    fig.add_trace(go.Bar(name='AOS before Campaign', x=aos_df['Campaign'], y=aos_df['AOS before Campaign'], 
                         marker_color='lightcoral'), row=1, col=1)
    fig.add_trace(go.Bar(name='AOS during Campaign', x=aos_df['Campaign'], y=aos_df['AOS during Campaign'], 
                         marker_color='coral'), row=1, col=1)
    fig.add_trace(go.Bar(name='AOS after Campaign', x=aos_df['Campaign'], y=aos_df['AOS after Campaign'], 
                         marker_color='darkred'), row=1, col=1)
    
    # Add AOV bars (bottom plot)
    fig.add_trace(go.Bar(name='AOV before Campaign', x=aov_df['Campaign'], y=aov_df['AOV before Campaign'], 
                         marker_color='thistle'), row=2, col=1)
    fig.add_trace(go.Bar(name='AOV during Campaign', x=aov_df['Campaign'], y=aov_df['AOV during Campaign'], 
                         marker_color='violet'), row=2, col=1)
    fig.add_trace(go.Bar(name='AOV after Campaign', x=aov_df['Campaign'], y=aov_df['AOV after Campaign'], 
                         marker_color='purple'), row=2, col=1)

    # Update layout for the combined figure
    fig.update_layout(
        height=750, width=850,
        barmode='group',
        showlegend=True
    )
    
    # Set axis titles
    fig.update_xaxes(title_text="Campaign", row=1, col=1)
    fig.update_yaxes(title_text="Average Order Size (AOS)", row=1, col=1)
    fig.update_xaxes(title_text="Campaign", row=2, col=1)
    fig.update_yaxes(title_text="Average Order Value (AOV)", row=2, col=1)

    # Show the plot
    fig.show()

def plot_aov_by_category(aov_by_category_df, chosen_year):
    '''
    Plots AOV by category for each campaign in the chosen year.
    '''
    # Create a grouped bar chart
    fig = px.bar(
        aov_by_category_df,
        x='category',
        y='AOV',
        color='Phase',  # Use different colors for each phase
        barmode='group',  # Group bars by category
        facet_col='Campaign',  # Create separate plots for each campaign
        title=f'AOV by Category for Each Campaign in {chosen_year}',
        labels={'AOV': 'Average Order Value', 'category': 'Category'},
        color_discrete_map={'before': 'darkgrey', 'during': 'blue', 'after': 'darkgrey'}
    )

    # Update layout to improve readability
    fig.update_layout(
        xaxis_title='Category',
        yaxis_title='Average Order Value (AOV)',
        legend_title_text='Campaign Phase',
        height=600,
        width=1200,
        title_x=0.5
    )
    
    # Update facet labels to show just the campaign name
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split('=')[1]))

    fig.show()


"""
The code below is for dashboarding with Streamlit app
"""

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
