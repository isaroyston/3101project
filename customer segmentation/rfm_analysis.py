# rfm_analysis.py

"""
rfm_analysis.py

This module provides functions for analyzing and segmenting customer data using RFM metrics.
Each function is documented to explain its purpose, parameters, and usage.

Author: Richelle
Last Update: 4/11/2024
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go


#############################################################################################################################

# Kmeans plots and functions

# 1. Plot RFM Distributions

def plot_rfm_distributions(rfm_df, recency_col='Recency', frequency_col='Frequency', monetary_col='Monetary'):
    """
    Plots the distribution of Recency, Frequency, and Monetary in an RFM DataFrame.
    
    Parameters:
    rfm_df (pd.DataFrame): DataFrame containing the RFM data 
    recency_col (str): Column name for Recency values
    frequency_col (str): Column name for Frequency values
    monetary_col (str): Column name for Monetary values
    
    Returns:
    None
    """
    # Adjust the figure size
    plt.figure(figsize=(12, 10))

    # Plot recency density
    plt.subplot(3, 1, 1)
    sns.kdeplot(rfm_df[recency_col], fill=True)
    plt.title("Recency Density")

    # Plot frequency density
    plt.subplot(3, 1, 2)
    sns.kdeplot(rfm_df[frequency_col], fill=True)
    plt.title("Frequency Density")

    # Plot monetary value density
    plt.subplot(3, 1, 3)
    sns.kdeplot(rfm_df[monetary_col], fill=True)
    plt.title("Monetary Density")

    # Show the plot
    plt.tight_layout()
    plt.show()

# 2. Analyze Skewness of RFM Data 

def analyze_skewness(rfm, x):
    """
    Analyzes skewness of a given column with different transformations, identifies the best method, and prints formatted results.

    Parameters:
    rfm (pd.DataFrame): DataFrame containing the RFM scores
    x (str): Column name to analyze skewness
    
    Returns:
    tuple: A dictionary containing the skewness values for each transformation, and the best method as a string.
    """
    # Print formatted header
    print(f'Skewness analysis for "{x}":')
    
    # Original skewness
    original_skew = rfm[x].skew().round(2)
    
    # Log transformation skewness
    log_data = np.log1p(rfm[x])  # log1p to handle zeros safely
    log_skew = log_data.skew().round(2)
    
    # Square root transformation skewness
    sqrt_data = np.sqrt(rfm[x])
    sqrt_skew = sqrt_data.skew().round(2)
    
    # Box-Cox transformation skewness (only on positive values)
    boxcox_data, _ = stats.boxcox(rfm[x][rfm[x] > 0])  # Ensure positive values for Box-Cox
    boxcox_skew = pd.Series(boxcox_data).skew().round(2)

    # Store skewness results in a dictionary
    skewness_results = {
        'Original Skewness': original_skew,
        'Log Transform Skewness': log_skew,
        'Square Root Transform Skewness': sqrt_skew,
        'Box-Cox Transform Skewness': boxcox_skew
    }

    # Identify the best transformation (closest to zero skewness)
    abs_skewness = {k: abs(v) for k, v in skewness_results.items()}
    best_method = min(abs_skewness, key=abs_skewness.get)

    # Print skewness results and best method
    for transform, skew in skewness_results.items():
        print(f"{transform}: {skew}")
    print(f"\nBest transformation method: {best_method} with skewness of {skewness_results[best_method]}")
    
    # Add a separator line for readability between analyses
    print('\n' + '-' * 40 + '\n')
    
    # Return skewness results and best method as separate values
    return skewness_results, best_method

# 3. Standardize and Transform the RFM Data

def transform_and_scale_rfm_data(rfm_df, best_method_r, best_method_f, best_method_m, customer_key='customer_key'):
    """
    Applies the specified transformations for each RFM metric based on the best method identified 
    and scales the transformed data.
    
    Parameters:
    rfm_df (pd.DataFrame): DataFrame containing the RFM scores
    best_method_r (str): Best transformation method for 'Recency'
    best_method_f (str): Best transformation method for 'Frequency'
    best_method_m (str): Best transformation method for 'Monetary'
    customer_key (str): Column name for the customer identifier
    
    Returns:
    pd.DataFrame: A new DataFrame with the customer key and scaled, transformed RFM values
    """
    # Initialize the transformed DataFrame with the customer key
    transformed_rfm_df = pd.DataFrame()
    transformed_rfm_df[customer_key] = rfm_df[customer_key]

    # Transformation mapping
    best_methods = {
        'Recency': best_method_r,
        'Frequency': best_method_f,
        'Monetary': best_method_m
    }

    # Apply transformations based on best methods
    for column, best_method in best_methods.items():
        if best_method == 'Log Transform Skewness':
            transformed_rfm_df[column] = np.log1p(rfm_df[column])
        elif best_method == 'Square Root Transform Skewness':
            transformed_rfm_df[column] = np.sqrt(rfm_df[column])
        elif best_method == 'Box-Cox Transform Skewness':
            # Box-Cox transformation requires positive values
            transformed_rfm_df[column] = stats.boxcox(rfm_df[column][rfm_df[column] > 0])[0]
        else:
            # If no transformation is recommended, use original values
            transformed_rfm_df[column] = rfm_df[column]

    # Initialize the scaler
    scaler = StandardScaler()  # Replace with MinMaxScaler() if normalization is preferred

    # Scale the RFM columns (excluding the customer key)
    rfm_columns = ['Recency', 'Frequency', 'Monetary']
    transformed_rfm_df[rfm_columns] = scaler.fit_transform(transformed_rfm_df[rfm_columns])

    return transformed_rfm_df

# 4. Perform K-Means Clustering

def find_optimal_clusters(data, max_k=10):
    """
    Finds the optimal number of clusters for KMeans using the elbow method.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data to cluster (RFM scores transformed and scaled).
    max_k (int): The maximum number of clusters to test.

    Returns:
    list: The inertia values based on the elbow method.
    """
    # List to store the inertia for each k
    inertia_values = []

    # Test a range of cluster numbers
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)

    # Plot the inertia values to visualize the elbow
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertia_values, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()

    return inertia_values

# 5. Assign Cluster Labels

def train_and_label_clusters(original_rfm_df, transformed_rfm_df, customer_key, n_clusters):
    """
    Trains a KMeans model on transformed data, assigns clusters, and merges cluster labels
    with original RFM values for easy interpretation.
    
    Parameters:
    original_rfm_df (pd.DataFrame): Original DataFrame with untransformed RFM values and customer key
    transformed_rfm_df (pd.DataFrame): DataFrame with transformed and scaled RFM data
    customer_key (str): Column name for the customer identifier
    n_clusters (int): Optimal number of clusters for KMeans

    Returns:
    pd.DataFrame: DataFrame containing original RFM values, customer key, and assigned cluster labels
    """
    # Initialize and train the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(transformed_rfm_df[['Recency', 'Frequency', 'Monetary']])

    # Predict cluster labels
    transformed_rfm_df['cluster_id'] = kmeans.predict(transformed_rfm_df[['Recency', 'Frequency', 'Monetary']])

    # Add the customer_key to the transformed data
    transformed_rfm_df[customer_key] = original_rfm_df[customer_key].values

    # Merge original RFM values with cluster labels using customer_key
    labeled_rfm_df = original_rfm_df.merge(
        transformed_rfm_df[[customer_key, 'cluster_id']], on=customer_key, how='left'
    )

    return labeled_rfm_df

# 6. Cluster Analysis
def analyze_clusters(segmented_df, rfm_columns=['Recency', 'Frequency', 'Monetary'], cluster_id_col='cluster_id'):
    """
    Analyzes clusters by providing summary statistics for each cluster.
    
    Parameters:
    segmented_df (pd.DataFrame): DataFrame with RFM metrics and cluster labels
    rfm_columns (list): List of RFM metric columns to analyze
    cluster_id_col (str): Column name for cluster labels
    
    Returns:
    pd.DataFrame: DataFrame containing the summary statistics for each cluster
    """
    # Calculate summary statistics for each cluster
    cluster_summary = segmented_df.groupby(cluster_id_col)[rfm_columns].agg(['mean', 'median', 'count'])
    
    # Flatten the MultiIndex columns for easier interpretation
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
    
    # Reset index to make cluster_id a column
    cluster_summary.reset_index(inplace=True)
    
    # Display and return the summary statistics for clusters
    print("Cluster Analysis Summary:")
    print(cluster_summary)
    
    return cluster_summary

# 7. Visualizing RFM Clusters

# using snake plots to visualize RFM values across clusters
def plot_snake_plot(df, rfm_columns=['Recency', 'Frequency', 'Monetary'], cluster_col='cluster_id'):
    """
    Plots a snake plot using Plotly to visualize RFM values across clusters.

    Parameters:
    df (pd.DataFrame): DataFrame containing RFM metrics and cluster labels
    rfm_columns (list): List of RFM metric columns to plot
    cluster_col (str): Column name for cluster labels

    Returns:
    None: Displays the snake plot
    """
    # Calculate mean RFM values for each cluster
    cluster_means = df.groupby(cluster_col)[rfm_columns].mean()

    # Normalize the values for better visualization
    cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    cluster_means_normalized.reset_index(inplace=True)

    # Melt the DataFrame to have RFM metrics on the x-axis and cluster means as values
    plot_df = cluster_means_normalized.melt(id_vars=cluster_col, var_name='RFM Metric', value_name='Normalized Value')

    # Plot using Plotly
    fig = px.line(
        plot_df,
        x='RFM Metric',
        y='Normalized Value',
        color=cluster_col,
        markers=True,
        title="Snake Plot of Clusters Across RFM Metrics"
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="RFM Metric",
        yaxis_title="Normalized Value",
        legend_title_text="Cluster",
        title_x=0.5
    )

    fig.show()

def plot_cluster_scatter(df, recency_col='Recency', frequency_col='Frequency', monetary_col='MonetaryValue', cluster_col='cluster_id'):
    """
    Plots a 3D scatter plot to visualize clusters based on RFM metrics.

    Parameters:
    df (pd.DataFrame): DataFrame containing RFM metrics and cluster labels
    recency_col (str): Column name for Recency values
    frequency_col (str): Column name for Frequency values
    monetary_col (str): Column name for Monetary values
    cluster_col (str): Column name for cluster labels

    Returns:
    None: Displays the 3D scatter plot
    """

    # Ensure the cluster column is treated as a categorical variable
    df[cluster_col] = df[cluster_col].astype(str)

        # Create 3D scatter plot
    fig = px.scatter_3d(
        df,
        x=recency_col,
        y=frequency_col,
        z=monetary_col,
        color=cluster_col,
        title="3D Scatter Plot of Customer Clusters Based on RFM Metrics",
        labels={recency_col: 'Recency', frequency_col: 'Frequency', monetary_col: 'Monetary Value'},
        opacity=0.7
    )

    # Update marker size and layout for readability
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        scene=dict(
            xaxis_title='Recency',
            yaxis_title='Frequency',
            zaxis_title='Monetary Value'
        ),
        legend_title_text="Cluster",
        title_x=0.5
    )

    fig.show()

# pair plot 

def plot_pairplot(df, rfm_columns=['Recency', 'Frequency', 'Monetary'], cluster_col='cluster_id'):
    """
    Plots a pair plot of RFM metrics to visualize clusters.

    Parameters:
    df (pd.DataFrame): DataFrame containing RFM metrics and cluster labels
    rfm_columns (list): List of RFM metric columns to include in the plot
    cluster_col (str): Column name for cluster labels

    Returns:
    None: Displays the pair plot
    """
    # Ensure clusters are treated as categorical data
    df[cluster_col] = df[cluster_col].astype(str)
    
    # Pair plot
    sns.pairplot(df, vars=rfm_columns, hue=cluster_col, palette='viridis', diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle("Pair Plot of RFM Metrics by Cluster", y=1.02)
    plt.show()


#############################################################################################################################

# manual rfm analysis functions

# Perform RFM analysis on the dataset
def rfm_analysis(df, ref_date, start_date, end_date):
    # Convert dates to datetime format
    ref_date = pd.to_datetime(ref_date)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter the data for the specified time period
    df_filtered = df[(df['purchase_date'] >= start_date) & (df['purchase_date'] <= end_date)]
    
    # Recency: Days since the last purchase within the time period
    recency_df = df_filtered.groupby('customer_key').purchase_date.max().reset_index()
    recency_df['Recency'] = (ref_date - recency_df['purchase_date']).dt.days
    
    # Frequency: Number of purchases within the time period
    frequency_df = df_filtered.groupby('customer_key').purchase_date.count().reset_index()
    frequency_df.columns = ['customer_key', 'Frequency']
    
    # Monetary: Total revenue within the time period
    monetary_df = df_filtered.groupby('customer_key').revenue.sum().reset_index()
    monetary_df.columns = ['customer_key', 'Monetary']
    
    # Merge the dataframes
    rfm_df = recency_df.merge(frequency_df, on='customer_key').merge(monetary_df, on='customer_key')
    
    # Assign scores based on quantiles
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'], 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    
    # Combine RFM scores into a single RFM segment
    rfm_df['RFM_Segment'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
    rfm_df['RFM_Score'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    
    return rfm_df

# Create a tree map visualization of RFM segments
def plot_treemap(df, segment_col='Segment'):
    """
    Plots an interactive treemap for customer segments using Plotly.

    Parameters:
    df (pd.DataFrame): DataFrame containing customer segments
    segment_col (str): Column name for customer segment labels

    Returns:
    None: Displays the treemap plot
    """
    # Calculate the count of each segment
    segment_counts = df[segment_col].value_counts().reset_index()
    segment_counts.columns = [segment_col, 'Count']

    # Create the treemap
    fig = px.treemap(
        segment_counts,
        path=[segment_col],
        values='Count',
        color='Count',
        color_continuous_scale='Viridis',  # You can adjust the color scale if desired
        title='Customer Segmentation Treemap'
    )

    # Customize the layout for better readability
    fig.update_layout(
        title_x=0.5,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    fig.update_traces(textinfo='label+value', hovertemplate='<b>%{label}</b><br>Count: %{value}')

    # Display the treemap
    fig.show()

# Define the segment labels based on RFM scores
def label_customer_segments(df, rfm_col='RFM_Segment'):
    """
    Labels RFM segments based on predefined RFM score combinations.

    Parameters:
    df (pd.DataFrame): DataFrame containing RFM scores
    rfm_col (str): Name of the column containing combined RFM scores

    Returns:
    pd.DataFrame: New DataFrame with an additional 'Segment' column
    """

    # Create a copy of the original DataFrame
    new_df = df.copy()

    # Define the segments and their corresponding RFM score lists
    segments = {
        'Champions': {'555', '554', '544', '545', '454', '455', '445'},
        'Loyal': {'543', '444', '435', '355', '354', '345', '344', '335'},
        'Potential Loyalist': {'553', '551', '552', '541', '542', '533', '532', '531', '452', '451', 
                               '442', '441', '431', '453', '433', '432', '423', '353', '352', '351', 
                               '342', '341', '333', '323'},
        'New Customers': {'512', '511', '422', '421', '412', '411', '311'},
        'Promising': {'525', '524', '523', '522', '521', '515', '514', '513', '425', '424', 
                      '413', '414', '415', '315', '314', '313'},
        'Need Attention': {'535', '534', '443', '434', '343', '334', '325', '324'},
        'About To Sleep': {'331', '321', '312', '221', '213', '231', '241', '251'},
        'At Risk': {'255', '254', '245', '244', '253', '252', '243', '242', '235', '234', 
                    '225', '224', '153', '152', '145', '143', '142', '135', '134', '133', 
                    '125', '124'},
        'Cannot Lose Them': {'155', '144', '214', '215', '115', '114', '113'},
        'Hibernating': {'332', '322', '231', '241', '251', '233', '232', '223', '222', '132', 
                        '123', '122', '212', '211'},
        'Lost': {'111', '112', '121', '131', '141', '151'}
    }

    # Define a function to map each RFM score to the corresponding segment
    def get_segment(rfm_score):
        for segment, scores in segments.items():
            if rfm_score in scores:
                return segment
        return 'Other'  # Default label if no match found

    # Apply the segmentation on new_df
    new_df['Segment'] = new_df[rfm_col].apply(get_segment)
    
    return new_df

def label_customer_segments_simple(df, recency_col='R_Score', frequency_col='F_Score', monetary_col='M_Score', segment_col='Segment'):
    """
    Segments customers in a DataFrame based on their RFM scores.

    Parameters:
    df (pd.DataFrame): DataFrame containing RFM scores
    recency_col (str): Column name for Recency scores
    frequency_col (str): Column name for Frequency scores
    monetary_col (str): Column name for Monetary scores
    segment_col (str): Column name to store the customer segment label

    Returns:
    pd.DataFrame: New DataFrame with an additional 'Segment' column
    """

    # Create a copy of the original DataFrame
    new_df = df.copy()

    def get_segment(r, f, m):
        # Champions: high scores in Recency, Frequency, and Monetary
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        
        # Loyal Customers: high Frequency and Monetary, moderate Recency
        elif f >= 3 and m >= 3:
            return 'Loyal Customers'
        
        # At Risk: low Recency, moderate to high Frequency and Monetary
        elif r <= 2 and (f >= 2 or m >= 2):
            return 'At Risk'
        
        # Low Value: low scores in Recency, Frequency, and Monetary
        else:
            return 'Low Value'
        
    # Create new rfm_df
    new_df = df.copy()
    
    # Apply the segmentation function to each row in the DataFrame
    new_df[segment_col] = df.apply(lambda row: get_segment(row[recency_col], row[frequency_col], row[monetary_col]), axis=1)
    
    return new_df


#############################################################################################################################

# Extra plots

# get customer_key of at risk customers

def get_at_risk_customers(df, cluster_col='cluster_id', at_risk_cluster=1):
    """
    Extracts 'at-risk' customers from the DataFrame based on the specified cluster.

    Parameters:
    df (pd.DataFrame): DataFrame containing customer data with cluster labels
    cluster_col (str): Column name containing the cluster IDs
    at_risk_cluster (int): The cluster ID associated with 'at-risk' customers

    Returns:
    pd.DataFrame: DataFrame containing only 'at-risk' customers
    """
    # Filter the DataFrame for the specified 'at-risk' cluster
    at_risk_df = df[df[cluster_col] == at_risk_cluster].copy()
    
    return at_risk_df

# rename clusters
def rename_clusters(df, cluster_col='cluster_id', mapping=None, new_col='Segment'):
    """
    Renames cluster IDs to descriptive labels based on a given mapping.

    Parameters:
    df (pd.DataFrame): DataFrame containing cluster labels.
    cluster_col (str): Column name containing the cluster IDs.
    mapping (dict): Dictionary where keys are cluster IDs and values are descriptive labels.
                    Example: {2: "Champions", 3: "Loyal", 0: "Retention Potential", 1: "At-Risk"}
    new_col (str): Name of the new column to store descriptive labels.

    Returns:
    pd.DataFrame: DataFrame with an additional column containing the descriptive cluster labels.
    """
    # Check if mapping is provided
    if mapping is None:
        raise ValueError("A mapping dictionary must be provided with cluster IDs as keys and labels as values.")
    
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Map cluster IDs to labels
    df_copy[new_col] = df_copy[cluster_col].map(mapping)
    
    # Check if any cluster IDs did not match the mapping
    if df_copy[new_col].isnull().any():
        missing_clusters = df_copy[df_copy[new_col].isnull()][cluster_col].unique()
        print(f"Warning: The following cluster IDs were not found in the mapping: {missing_clusters}")
    
    return df_copy

#############################################################################################################################

# Analysis Plots

# segment changes
def plot_segment_changes(base_df, comparison_df, segment_col='Segment', id_col='CustomerID', base_year_label='2014', comparison_year_label='2020'):
    """
    Plots a horizontal bar chart showing changes in segment counts between two DataFrames for different years.

    Parameters:
    base_df (pd.DataFrame): DataFrame for the base year, containing segment labels and customer IDs.
    comparison_df (pd.DataFrame): DataFrame for the comparison year, containing segment labels and customer IDs.
    segment_col (str): Column name for the segment labels.
    id_col (str): Column name for customer IDs or unique identifiers.
    base_year_label (str): Label for the base year (e.g., '2014').
    comparison_year_label (str): Label for the comparison year (e.g., '2020').

    Returns:
    None: Displays the horizontal bar chart.
    """
    # Calculate segment counts for the base and comparison DataFrames
    base_counts = base_df[[segment_col, id_col]].groupby(segment_col).count().rename(columns={id_col: f'count_{base_year_label}'})
    comparison_counts = comparison_df[[segment_col, id_col]].groupby(segment_col).count().rename(columns={id_col: f'count_{comparison_year_label}'})
    
    # Merge the counts and calculate the difference
    segment_changes = base_counts.join(comparison_counts, how='outer').fillna(0)
    segment_changes['Change'] = segment_changes[f'count_{comparison_year_label}'] - segment_changes[f'count_{base_year_label}']
    
    # Separate positive and negative changes for coloring
    colors = ['green' if change > 0 else 'red' for change in segment_changes['Change']]
    
    # Plot using Plotly
    fig = go.Figure(go.Bar(
        x=segment_changes['Change'],
        y=segment_changes.index,
        orientation='h',
        marker=dict(color=colors),
        text=segment_changes['Change'],
        textposition='outside'
    ))

    # Update layout for better readability
    fig.update_layout(
        title=f"Change in Segment Counts ({comparison_year_label} - {base_year_label})",
        xaxis_title="Change in Count",
        yaxis_title="Segments",
        title_x=0.5,
        xaxis=dict(showgrid=True, zeroline=True),
    )

    fig.show()


def plot_segment_counts_over_years(*rfm_dfs, year_start=2014):
    """
    Plots the count of people in each segment across multiple years.

    Parameters:
    - *rfm_dfs: Variable number of RFM DataFrames (one for each year).
    - year_start (int): The starting year, which will increment by 1 for each RFM DataFrame provided.
    
    Returns:
    - None: Displays a Plotly line chart.
    """
    
    # Add year column to each RFM DataFrame and concatenate them
    rfm_dfs_with_years = []
    for i, rfm_df in enumerate(rfm_dfs):
        rfm_df = rfm_df.copy()
        rfm_df['year'] = year_start + i
        rfm_dfs_with_years.append(rfm_df)
    
    combined_df = pd.concat(rfm_dfs_with_years, ignore_index=True)
    
    # Group by year and segment to count the number of people in each segment per year
    segment_counts = combined_df.groupby(['year', 'Segment']).size().reset_index(name='count')
    
    # Plot the line chart using Plotly
    fig = px.line(segment_counts, 
                  x='year', 
                  y='count', 
                  color='Segment', 
                  title='Segment Counts Over Years',
                  labels={'year': 'Year', 'count': 'Number of People', 'Segment': 'Customer Segment'},
                  markers=True)
    
    fig.show()

