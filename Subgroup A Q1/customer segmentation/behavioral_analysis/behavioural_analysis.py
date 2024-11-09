import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime

"""
Data Cleaning and Preprocessing
"""
# Define a function to extract the category
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

# Define the function to categorize time of day
def extract_time_of_day(time_str):
    try:
        # Convert the time string to a datetime.time object
        time_obj = datetime.strptime(time_str, '%H:%M:%S').time()  # Adjust format if needed
        hour = time_obj.hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "midnight"
    except ValueError:
        return None  # Handle cases where time format is incorrect
    
def extract_channel_type(description):
    """
    Extracts the channel type from a description string formatted as 'MKT-CHANNEL-CAMPAIGN-YEAR'.
    
    Parameters:
    - description (str): The description string containing channel information.
    
    Returns:
    - str: The extracted channel type, or None if the format is incorrect.
    """
    if isinstance(description, str):
        parts = description.split('-')
        if len(parts) > 1:
            return parts[1]  
    return 'None'  



def add_marketing_indicator(rfm_label_df, original_df, customer_id_col='customer_id', marketing_key_col='mkt_key'):
    """
    Adds a marketing indicator to rfm_label_df based on participation in the original_df.

    Parameters:
    - rfm_label_df (DataFrame): The RFM label DataFrame with unique customers.
    - original_df (DataFrame): The original DataFrame containing marketing campaign data.
    - customer_id_col (str): The column name for the customer identifier.
    - marketing_key_col (str): The column name in the original DataFrame indicating marketing participation.

    Returns:
    - DataFrame: Updated rfm_label_df with a 'marketing_indicator' column.
    """

    # Step 1: Identify customers who have participated in any marketing campaign
    participated_customers = original_df[original_df[marketing_key_col].notnull()][customer_id_col].unique()

    # Step 2: Create a DataFrame to map customer IDs to marketing participation
    participation_df = pd.DataFrame({customer_id_col: participated_customers, 'marketing_indicator': 1})

    # Step 3: Merge with rfm_label_df to add the marketing_indicator
    rfm_label_df = rfm_label_df.merge(participation_df, on=customer_id_col, how='left')

    # Step 4: For non-participants, 'marketing_indicator' will remain NaN
    return rfm_label_df


def plot_campaign_participation_percentage_by_segment(rfm_label_df, segment_col='segment', marketing_indicator_col='marketing_indicator'):
    """
    Creates a stacked percentage bar chart showing the proportion of people who participated in a campaign
    versus those who did not, grouped by customer segment.

    Parameters:
    - rfm_label_df (DataFrame): The DataFrame containing segment and marketing indicator data.
    - segment_col (str): The column name for customer segments.
    - marketing_indicator_col (str): The column name for the marketing indicator.
    """
    
    # Fill NaN values in 'marketing_indicator' with 0 for non-participants
    rfm_label_df[marketing_indicator_col] = rfm_label_df[marketing_indicator_col].fillna(0)
    
    # Group by segment and marketing indicator to get counts
    participation_counts = rfm_label_df.groupby([segment_col, marketing_indicator_col]).size().reset_index(name='count')
    
    # Calculate total count per segment
    total_counts = participation_counts.groupby(segment_col)['count'].transform('sum')
    
    # Calculate percentage of each group within each segment
    participation_counts['percentage'] = participation_counts['count'] / total_counts * 100
    
    # Map marketing_indicator values to 'Participated' and 'Not Participated'
    participation_counts[marketing_indicator_col] = participation_counts[marketing_indicator_col].map({1: 'Participated', 0: 'Not Participated'})
    
    # Plot using Plotly
    fig = px.bar(participation_counts, 
                 x=segment_col, 
                 y='percentage', 
                 color=marketing_indicator_col,
                 title='Campaign Participation Percentage by Customer Segment',
                 labels={'percentage': 'Percentage (%)', segment_col: 'Customer Segment'},
                 barmode='stack')
    
    fig.show()


def analyze_popular_items_by_average_revenue(df, segment_col='Segment', category_col='category', revenue_col='revenue'):
    """
    Analyzes the average revenue earned per item description within each customer segment
    and visualizes the top descriptions by average revenue for each segment.

    Parameters:
    - df (DataFrame): The DataFrame containing segment, item description, and revenue data.
    - segment_col (str): The column name for customer segments.
    - category_col (str): The column name for item category.
    - revenue_col (str): The column name for revenue.

    Returns:
    - DataFrame: A DataFrame with average revenue per description within each segment.
    """
    
    # Group by segment and description, then calculate the average revenue
    avg_revenue_df = df.groupby([segment_col, category_col]).agg(
        avg_revenue=(revenue_col, 'mean')
    ).reset_index()
    
    # Sort by segment and average revenue within each segment
    avg_revenue_df = avg_revenue_df.sort_values(by=[segment_col, 'avg_revenue'], ascending=[True, False])
    
    # Visualization: Top descriptions by average revenue per segment
    fig = px.bar(avg_revenue_df, 
                 x='avg_revenue', 
                 y=category_col, 
                 color=segment_col, 
                 facet_row=segment_col, 
                 orientation='h',
                 title='Average Revenue by Item Description for Each Segment',
                 labels={'avg_revenue': 'Average Revenue', category_col: 'Category'},
                 height=600)

    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, 
                      showlegend=False)
    fig.show()
    
    return avg_revenue_df

import plotly.express as px

def plot_time_of_day_purchases_by_segment(df):
    """
    Plots a stacked bar chart showing the number of purchases for each segment
    at different times of the day (morning, afternoon, evening, midnight).
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing the purchase data. 
                         Must have 'Segment' and 'time_of_day' columns.
    
    Returns:
    - fig: A Plotly figure object of the stacked bar chart.
    """

    # Define custom colors for each time of day
    color_map = {
        'morning': '#fbb4ae',
        'afternoon': '#b3cde3',
        'evening': '#fed9a6',
        'midnight': '#decbe4'
    }

    # Group by Segment and time_of_day, then count the number of purchases
    time_of_day_counts = df.groupby(['Segment', 'time_of_day']).size().reset_index(name='count')

    # Create the stacked bar chart
    fig = px.bar(
        time_of_day_counts, 
        x='Segment', 
        y='count', 
        color='time_of_day', 
        title='Most Popular Time-of-Day Purchases by Segment',
        labels={'count': 'Number of Purchases', 'time_of_day': 'Time of Day'},
        color_discrete_map=color_map, 
        category_orders={'time_of_day': ['morning', 'afternoon', 'evening', 'midnight']}  # Define order of time segments
    )
    
    fig.update_layout(barmode='stack', xaxis_title='Segment', yaxis_title='Number of Purchases')
    return fig


import plotly.express as px

def plot_campaign_participation_percentage_by_segment(df, segment_col='Segment', campaign_col='campaign_name'):
    """
    Plots the percentage participation in each campaign for each segment using a stacked bar chart.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - segment_col (str): The column name representing the segment.
    - campaign_col (str): The column name representing the campaign.
    
    Returns:
    - fig: A Plotly figure object of the stacked bar chart.
    """
    # Calculate counts of each campaign within each segment
    segment_campaign_counts = df.groupby([segment_col, campaign_col]).size().reset_index(name='count')

    # Calculate total counts for each segment to get percentages
    total_counts_per_segment = segment_campaign_counts.groupby(segment_col)['count'].transform('sum')
    segment_campaign_counts['percentage'] = (segment_campaign_counts['count'] / total_counts_per_segment) * 100

    # Create the stacked bar chart
    fig = px.bar(
        segment_campaign_counts,
        x=segment_col,
        y='percentage',
        color=campaign_col,
        title='Campaign Participation Percentage by Segment',
        labels={'percentage': 'Participation Percentage (%)', segment_col: 'Segment', campaign_col: 'Campaign'},
    )
    
    return fig

import plotly.express as px

def plot_channel_participation_distribution(df, channels, year_col='year', segment_col='Segment', channel_col='channel_type'):
    """
    Plots the percentage distribution of people in each segment that participated in selected channels for each year
    using a percentage stacked bar chart.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - channels (list): A list of channel types to include in the analysis.
    - year_col (str): The column name representing the year.
    - segment_col (str): The column name representing the segment.
    - channel_col (str): The column name representing the channel type.
    
    Returns:
    - fig: A Plotly figure object of the percentage stacked bar chart.
    """
    # Filter the DataFrame for the selected channels
    df_filtered = df[df[channel_col].isin(channels)]

    # Calculate the count of participants for each combination of year, segment, and channel
    participation_counts = df_filtered.groupby([year_col, segment_col, channel_col]).size().reset_index(name='count')

    # Calculate total participants per year and segment to get percentages
    total_counts_per_segment = participation_counts.groupby([year_col, segment_col])['count'].transform('sum')
    participation_counts['percentage'] = (participation_counts['count'] / total_counts_per_segment) * 100

    # Create the percentage stacked bar chart
    fig = px.bar(
        participation_counts,
        x=year_col,
        y='percentage',
        color=channel_col,
        facet_col=segment_col,
        title='Percentage Distribution of Segment Participation in Selected Channels by Year',
        labels={'percentage': 'Participation Percentage (%)', year_col: 'Year', segment_col: 'Segment'},
    )
    
    return fig

import plotly.express as px

def plot_campaign_participation_distribution(df, campaigns, year_col='year', segment_col='Segment', campaign_col='campaign_name'):
    """
    Plots the distribution of people in each segment that participated in selected campaigns for each year
    using a stacked bar chart.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - campaigns (list): A list of campaign names to include in the analysis.
    - year_col (str): The column name representing the year.
    - segment_col (str): The column name representing the segment.
    - campaign_col (str): The column name representing the campaign.
    
    Returns:
    - fig: A Plotly figure object of the stacked bar chart.
    """
    # Filter the DataFrame for the selected campaigns
    df_filtered = df[df[campaign_col].isin(campaigns)]

    # Calculate the count of participants for each combination of year, segment, and campaign
    participation_counts = df_filtered.groupby([year_col, segment_col, campaign_col]).size().reset_index(name='count')

    # Create the stacked bar chart
    fig = px.bar(
        participation_counts,
        x=year_col,
        y='count',
        color=campaign_col,
        facet_col=segment_col,
        title='Distribution of Segment Participation in Selected Campaigns by Year',
        labels={'count': 'Number of Participants', year_col: 'Year', segment_col: 'Segment'},
    )
    
    # Format layout
    fig.update_layout(barmode='stack', xaxis_title='Year', yaxis_title='Number of Participants')

    return fig

# Example usage