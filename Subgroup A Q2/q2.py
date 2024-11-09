import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

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
            category = 'Coffee Products'
        elif desc == 'Medicine':
            category = 'Medicine'
        elif desc == 'Kitchen Supplies':
            category = 'Kitchen Supplies'
        else:
            category = desc.split()[0].strip()
    return category

def yearly_churn(df, start_date='2014-01-01', end_date='2021-01-01'):
    '''
    This function calculates the yearly churn rate for each year from 2014 to 2021.
    '''
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Sort data by purchase_date
    df = df.sort_values(by='purchase_date')
    
    # Generate a list of yearly periods from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='YS').shift(19, freq='D') # Shift by 19 days to align with dataset start date
    
    # Initialize a list to store churn rates
    churn_rates = []
    
    # Loop through each date in the date range to calculate yearly churn
    for date in date_range:
        # Define current and previous year date ranges
        current_year_end = date
        previous_year_start = current_year_end - pd.DateOffset(years=2)
        previous_year_end = current_year_end - pd.DateOffset(years=1)
        
        # Filter data for purchases before the current year end
        prev_year_data = df[(df['purchase_date'] >= previous_year_start) & (df['purchase_date'] < previous_year_end)]
        current_year_data = df[(df['purchase_date'] >= previous_year_end) & (df['purchase_date'] < current_year_end)]
        
        # Get unique customers who purchased in the previous year
        prev_customers = set(prev_year_data['customer_key'].unique())
        
        # Get unique customers who purchased in the current year
        current_customers = set(current_year_data['customer_key'].unique())
        
        # Calculate stayed and churned customers
        stayed_customers = prev_customers & current_customers
        churned_customers = prev_customers - stayed_customers
        
        # Calculate total customers
        total_customers = prev_year_data['customer_key'].nunique()
        
        # Calculate churn rate
        if total_customers > 0:
            churn_rate = (len(churned_customers) / total_customers) * 100
        else:
            churn_rate = 0
            
        # Append churn rate for the current year to the list
        churn_rates.append({
            'year': current_year_end.strftime('%Y'),
            'churn_rate': churn_rate
        })
    
    # Convert churn rates list to a DataFrame and return it
    churn_df = pd.DataFrame(churn_rates)
    return churn_df

def plot_yearly_graph(df):
    '''
    This function plots the yearly churn rate from 2015 to 2021.
    '''
    # Create an interactive line plot with Plotly
    fig = px.line(
        df,
        x='year',
        y='churn_rate',
        title='Yearly Churn Rate',
        labels={'year': 'Year', 'churn_rate': 'Churn Rate (%)'}
    )

    # Customize the layout for better readability
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Churn Rate (%)',
        hovermode="x unified"  # Shows hover information for all traces at the same x-coordinate
    )
    # Show the plot
    fig.show()

def six_monthly_churn(data, start_date='2014-01-01', end_date='2021-01-01'):
    '''
    This function calculates the 6-monthly churn rate for each 6-month period from 2014 to 2021.
    '''
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Sort data by purchase_date
    data = data.sort_values(by='purchase_date')
    
    # Generate a list of 6-month periods from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='6MS').shift(19, freq='D') # Shift by 19 days to align with dataset start date
    
    # Initialize a list to store churn rates
    churn_rates = []
    
    # Loop through each date in the date range to calculate 6-monthly churn
    for date in date_range:
        # Define current and previous 6-month period date ranges
        current_period_end = date
        previous_period_start = current_period_end - pd.DateOffset(months=12)
        previous_period_end = current_period_end - pd.DateOffset(months=6)
        
        # Filter data for purchases within the previous and current 6-month periods
        prev_period_data = data[(data['purchase_date'] >= previous_period_start) & (data['purchase_date'] < previous_period_end)]
        current_period_data = data[(data['purchase_date'] >= previous_period_end) & (data['purchase_date'] < current_period_end)]
        
        # Get unique customers who purchased in the previous 6-month period
        prev_customers = set(prev_period_data['customer_key'].unique())
        
        # Get unique customers who purchased in the current 6-month period
        current_customers = set(current_period_data['customer_key'].unique())
        
        # Calculate stayed and churned customers
        stayed_customers = prev_customers & current_customers
        churned_customers = prev_customers - stayed_customers
        
        # Calculate total customers
        total_customers = prev_period_data['customer_key'].nunique()
        
        # Calculate churn rate
        if total_customers > 0:
            churn_rate = (len(churned_customers) / total_customers) * 100
        else:
            churn_rate = 0
            
        # Append churn rate for the current 6-month period to the list
        churn_rates.append({
            'period': current_period_end.strftime('%Y-%m'),
            'churn_rate': churn_rate
        })
    
    # Convert churn rates list to a DataFrame and return it
    churn_df = pd.DataFrame(churn_rates)
    return churn_df

def plot_six_monthly_graph(df):
    '''
    This function plots the 6-monthly churn rate from 2014 to 2021.
    '''
    # Create an interactive line plot with Plotly
    fig = px.line(
        df,
        x='period',
        y='churn_rate',
        title='6-Monthly Churn Rate',
        labels={'period': 'Period', 'churn_rate': 'Churn Rate (%)'}
    )

    # Customize the layout for better readability
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Churn Rate (%)',
        hovermode="x unified"  # Shows hover information for all traces at the same x-coordinate
    )
    # Show the plot
    fig.show()

def quarterly_churn(data, start_date='2014-01-01', end_date='2021-01-01'):
    '''
    This function calculates the quarterly churn rate for each quarter from 2014 to 2021.
    '''
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Sort data by purchase_date
    data = data.sort_values(by='purchase_date')
    
    # Generate a list of quarterly periods from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='QS').shift(19, freq='D') # Shift by 19 days to align with dataset start date

    # Initialize a list to store churn rates
    churn_rates = []
    
    # Loop through each date in the date range to calculate quarterly churn
    for date in date_range:
        # Define current and previous quarter date ranges
        current_quarter_end = date
        previous_quarter_start = current_quarter_end - pd.DateOffset(months=6)
        previous_quarter_end = current_quarter_end - pd.DateOffset(months=3)
        
        # Filter data for purchases within the previous and current quarters
        prev_quarter_data = data[(data['purchase_date'] >= previous_quarter_start) & (data['purchase_date'] < previous_quarter_end)]
        current_quarter_data = data[(data['purchase_date'] >= previous_quarter_end) & (data['purchase_date'] < current_quarter_end)]
        
        # Get unique customers who purchased in the previous quarter
        prev_customers = set(prev_quarter_data['customer_key'].unique())
        
        # Get unique customers who purchased in the current quarter
        current_customers = set(current_quarter_data['customer_key'].unique())
        
        # Calculate stayed and churned customers
        stayed_customers = prev_customers & current_customers
        churned_customers = prev_customers - stayed_customers
        
        # Calculate total customers
        total_customers = prev_quarter_data['customer_key'].nunique()
        
        # Calculate churn rate
        if total_customers > 0:
            churn_rate = (len(churned_customers) / total_customers) * 100
        else:
            churn_rate = 0
            
        # Append churn rate for the current quarter to the list
        churn_rates.append({
            'quarter': current_quarter_end.strftime('%Y-%m'),
            'churn_rate': churn_rate
        })
    
    # Convert churn rates list to a DataFrame and return it
    churn_df = pd.DataFrame(churn_rates)
    return churn_df

def plot_quarterly_graph(df):
    '''
    This function plots the quarterly churn rate from 2014 to 2021.
    '''
    # Create an interactive line plot with Plotly
    fig = px.line(
        df,
        x='quarter',
        y='churn_rate',
        title='Quarterly Churn Rate',
        labels={'period': 'Quarter', 'churn_rate': 'Churn Rate (%)'}
    )

    # Customize the layout for better readability
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Churn Rate (%)',
        hovermode="x unified"  # Shows hover information for all traces at the same x-coordinate
    )
    # Show the plot
    fig.show()

def monthly_churn(data, start_date='2014-01-20', end_date='2021-01-20'):
    '''
    This function calculates the monthly churn rate for each month from 2014 to 2021.
    '''

    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Sort data by purchase_date
    data = data.sort_values(by='purchase_date')
    
    # Generate a list of monthly periods from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS').shift(19, freq='D')
    
    # Initialize a list to store churn rates
    churn_rates = []
    
    # Loop through each date in the date range to calculate monthly churn
    for date in date_range:
        # Define current and previous month date ranges
        current_month_end = date
        previous_month_start = current_month_end - pd.DateOffset(months=2)
        previous_month_end = current_month_end - pd.DateOffset(months=1)
        
        # Filter data for purchases within the previous and current months
        prev_month_data = data[(data['purchase_date'] >= previous_month_start) & (data['purchase_date'] < previous_month_end)]
        current_month_data = data[(data['purchase_date'] >= previous_month_end) & (data['purchase_date'] < current_month_end)]
        
        # Get unique customers who purchased in the previous month
        prev_customers = set(prev_month_data['customer_key'].unique())
        
        # Get unique customers who purchased in the current month
        current_customers = set(current_month_data['customer_key'].unique())
        
        # Calculate stayed and churned customers
        stayed_customers = prev_customers & current_customers
        churned_customers = prev_customers - stayed_customers
        
        # Calculate total customers
        total_customers = prev_month_data['customer_key'].nunique()
        
        # Calculate churn rate
        if total_customers > 0:
            churn_rate = (len(churned_customers) / total_customers) * 100
        else:
            churn_rate = 0
            
        # Append churn rate for the current month to the list
        churn_rates.append({
            'month': current_month_end.strftime('%Y-%m'),
            'churn_rate': churn_rate
        })
    
    # Convert churn rates list to a DataFrame and return it
    churn_df = pd.DataFrame(churn_rates)
    return churn_df

def plot_monthly_graph(df):
    '''
    This function plots the monthly churn rate from 2014 to 2021.
    '''
    # Create an interactive line plot with Plotly
    fig = px.line(
        df,
        x='month',
        y='churn_rate',
        title='Monthly Churn Rate',
        labels={'month': 'Month', 'churn_rate': 'Churn Rate (%)'}
    )

    # Customize the layout for better readability
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Churn Rate (%)',
        hovermode="x unified"  # Shows hover information for all traces at the same x-coordinate
    )
    # Show the plot
    fig.show()

def plot_rolling_average(df):
    '''
    This function plots the monthly churn rate with a 3-month rolling average from 2014 to 2021. 
    3-month rolling average is calculated by taking the average of the current month's churn rate and the two previous months' churn rates. 
    This is to smooth out the fluctuations in the churn rate and highlight the underlying trend.
    '''
    # Add a column for the 3-month rolling average churn rate
    df['rolling_churn_rate'] = df['churn_rate'].rolling(window=3).mean()

    # Melt the DataFrame to have a single column for churn rates
    df_melted = df.melt(id_vars='month', value_vars=['churn_rate', 'rolling_churn_rate'],
                        var_name='Metric', value_name='Rate')

    # Create an interactive line plot with Plotly
    fig = px.line(
        df_melted,
        x='month',
        y='Rate',
        color='Metric',
        labels={'Rate': 'Churn Rate (%)', 'month': 'Year'},
        title='Monthly Churn Rate with 3-Month Rolling Average'
    )

    # Customize the layout to show months on the x-axis
    fig.update_layout(
        xaxis=dict(
            tickformat="%Y-%m",  # Display YYYY-MM on x-axis
            title="Year"
        ),
        yaxis_title="Churn Rate (%)",
        hovermode="x unified"  # Shows hover information for all lines at the same x-coordinate
    )
    fig.show()

def plot_bar_graph(df):
    '''
    This function plots the average monthly churn rate from 2014 to 2021 in a bar chart. 
    '''
    # Convert the 'month' column to datetime format
    df['month'] = pd.to_datetime(df['month'], errors='coerce')

    # Extract the month name for grouping purposes
    df['month_name'] = df['month'].dt.strftime('%B')

    # Group by month name and calculate the average churn rate for each month
    monthly_churn = df.groupby('month_name')['churn_rate'].mean().reset_index()

    # To ensure the months are in calendar order, reindex based on month order
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    monthly_churn['month_name'] = pd.Categorical(monthly_churn['month_name'], categories=month_order, ordered=True)
    monthly_churn = monthly_churn.sort_values('month_name')

    # Create an interactive bar plot with Plotly
    fig = px.bar(
        monthly_churn,
        x='month_name',
        y='churn_rate',
        title='Average Monthly Churn Rate (2014-2021)',
        labels={'month_name': 'Month', 'churn_rate': 'Average Churn Rate (%)'},
        color_discrete_sequence=['skyblue']
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Average Churn Rate (%)',
        xaxis_tickangle=-45  # Rotate x-axis labels for better readability
    )

    fig.show()

def individual_clv(df):
    '''
    This function calculates the CLV of each individual customer based on transactions in the WHOLE dataset (2014-2020), excluding 2021 incomplete data.
    Therefore, this CLV value is an measure of the lifetime value of each customer at the end of 2020. 
    We output the dataframe with a new CLV column.
    '''

    # Calculate total revenue, number of purchases, and customer lifespan for each customer
    customer_data = df.groupby('customer_key').agg(
        total_revenue=('revenue', 'sum'),  # Total revenue per customer
        total_purchases=('customer_key', 'count'),  # Total number of purchases per customer
        first_purchase=('purchase_date', 'min'),  # Date of first purchase
        last_purchase=('purchase_date', 'max')  # Date of last purchase
    ).reset_index()

    # Calculate customer lifespan (in years) based on the first and last purchase dates
    customer_data['customer_lifespan'] = ((customer_data['last_purchase'] - customer_data['first_purchase']).dt.days) / 365.25

    # Calculate individual CLV using the formula
    customer_data['CLV'] = customer_data['total_revenue'] * (customer_data['customer_lifespan'] / customer_data['total_purchases'])

    return customer_data

def clv_plot(customer_data):
    '''
    Generates a histogram and a boxplot of CLV distribution
    and scatter plots of CLV vs. Total Revenue and CLV vs. Lifespan to illustrate the relationship between the variables.
    '''

    plt.figure(figsize=(12, 10))
    
    # 1. Histogram of CLV
    plt.subplot(2, 2, 1)
    sns.histplot(customer_data['CLV'], bins=20, kde=True)
    plt.title('Distribution of CLV')
    plt.xlabel('CLV')
    plt.ylabel('Frequency')

    # 2. Box Plot of CLV
    plt.subplot(2, 2, 2)
    sns.boxplot(y=customer_data['CLV'])
    plt.title('Box Plot of CLV')
    plt.ylabel('CLV')

    # 3. Scatter Plot: CLV vs. Total Revenue
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=customer_data, x='total_revenue', y='CLV')
    plt.title('CLV vs. Total Revenue')
    plt.xlabel('Total Revenue')
    plt.ylabel('CLV')

    # 4. Scatter Plot: CLV by Customer Lifespan
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=customer_data, x='customer_lifespan', y='CLV')
    plt.title('CLV vs. Customer Lifespan')
    plt.xlabel('Customer Lifespan (Years)')
    plt.ylabel('CLV')

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

def annual_individual_clv(df):
    '''
    This function calculates the annual CLV of each individual customer according to transactions in the particular year.
    Therefore, the lifespan is a fraction of a year.
    We output a dataframe with an annual_CLV column for each customer and each year.
    '''

    # Calculate total revenue, number of purchases, and customer lifespan for each customer for each year
    customer_data = df.groupby(['customer_key', 'year']).agg(
        total_revenue=('revenue', 'sum'),              # Total revenue per customer per year
        total_purchases=('revenue', 'count'),          # Total number of purchases per customer per year
        first_purchase=('purchase_date', 'min'),       # Date of first purchase within the year
        last_purchase=('purchase_date', 'max')         # Date of last purchase within the year
    ).reset_index()

    # Calculate the annual customer lifespan (in years) based on the first and last purchase dates within each year
    customer_data['annual_lifespan'] = (
        (customer_data['last_purchase'] - customer_data['first_purchase']).dt.days
    ) / 365.25

    # Calculate annual CLV using the formula for each year per customer
    customer_data['annual_CLV'] = customer_data['total_revenue'] * (
        customer_data['annual_lifespan'] / customer_data['total_purchases']
    )

    return customer_data

def annual_CLV_plot(customer_data):
    '''
    Generates a histogram showing the annual CLV distribution.
    '''

    ### Histogram distribution
    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Plot the histogram of annual CLV for each year
    sns.histplot(data=customer_data, x='annual_CLV', hue='year', kde=True, multiple='stack', palette='Set2')

    # Customize the plot for better readability
    plt.title('Distribution of Annual CLV for Each Year')
    plt.xlabel('Annual CLV')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Show the plot
    plt.show()

def low_clv(customer_data):
    '''
    Here, we take the low CLV threshold for each year to be the lower quartile CLV each year.
    This function outputs a dataframe showing the threshold for each year.
    '''

    # For each year's data, calculate LQ CLV 
    low_clv_threshold_yearly = customer_data.groupby('year')['annual_CLV'].quantile(0.25).reset_index()

    # Rename calculated column to low_clv_threshold
    low_clv_threshold_yearly.rename(columns={'annual_CLV': 'low_clv_threshold'}, inplace=True)

    # Merge customer_data with low_clv_threshold_yearly to get the threshold for each year
    customer_data = customer_data.merge(
        low_clv_threshold_yearly, 
        on='year', 
        how='left'
    )

    # To store low CLV customers for each year
    low_clv_customers_by_year = {}

    for year in range(2014, 2021):
        # Filter customers in that year
        year_customers = customer_data[customer_data['year'] == year]

        # Filter customers whose annual_CLV is below the low_clv_threshold for that year
        low_clv_customers = year_customers[
            (year_customers['annual_CLV'] <= year_customers['low_clv_threshold'])
        ]

        # Store the low-CLV customers for that year
        low_clv_customers_by_year[year] = low_clv_customers
        
    # Output the dataframes
    return low_clv_threshold_yearly, low_clv_customers_by_year

def at_risk_customers_yearly(low_clv_customers_by_year):
    '''
    This function identifies the at-risk customers of every year. 
    We define our at-risk criteria to be:
    1. Below LQ annual CLV
    2. Below mean purchase frequency of that year
    '''

    # Store at-risk customers yearly in a dictionary
    at_risk_customers_by_year = {}

    for year in range(2014, 2021):
        # Filter customers in that year
        df = low_clv_customers_by_year[year]

        # Calculate the mean purchase frequency of that year
        mean_purchase_freq = df['total_purchases'].mean()

        # Identify at-risk customers based on the defined criteria
        at_risk_customers = df[
            (df['total_purchases'] < mean_purchase_freq)
        ].reset_index()

        # Store the at-risk customers for that year
        at_risk_customers_by_year[year] = at_risk_customers
        
        # Print the at-risk customers for the year
        print(f"\nAt-Risk Customers for Year {year}:")
        display(at_risk_customers[['customer_key', 'annual_CLV', 'low_clv_threshold']].head())
        print(f"Number of at-risk customers for year {year}: {at_risk_customers.shape[0]}")

def high_clv(customer_data):
    '''
    This function outputs high CLV customers.
    '''

    # Define the threshold for high CLV customers
    high_clv_threshold = customer_data['CLV'].quantile(0.75)  # 75th percentile

    # Filter high CLV customers
    high_clv_customers = customer_data[customer_data['CLV'] >= high_clv_threshold]

    return high_clv_customers


def top_categories(df, high_clv_customers):
    '''
    This function shows the top categories by AOV of high CLV customers with a bar chart.
    '''
    
    # Filter original dataframe for high CLV customers
    high_clv_df = df[df['customer_key'].isin(high_clv_customers['customer_key'])]
    
    # Analyze top products by category for high CLV customers
    top_categories_by_aov = high_clv_df.groupby('category').agg(
        total_revenue=('revenue', 'sum'),
        total_purchases=('customer_key', 'count')
    ).reset_index()
    
    top_categories_by_aov['aov'] = top_categories_by_aov['total_revenue'] / top_categories_by_aov['total_purchases']

    # Sort by AOV to find top categories
    top_categories_by_aov = top_categories_by_aov.sort_values(by='aov', ascending=False)

    # Display the results
    display(top_categories_by_aov)

    # plt.figure(figsize=(10, 6))

    # # Create a bar chart for AOV
    # plt.bar(top_categories_by_aov['category'], top_categories_by_aov['aov'], color='skyblue')

    # # Labels and title
    # plt.xlabel('Category')
    # plt.ylabel('Average Order Value (AOV)')
    # plt.title('Average Order Value (AOV) by Category of High CLV Customers')
    # plt.xticks(rotation=45, ha='right') 

    # plt.show()
