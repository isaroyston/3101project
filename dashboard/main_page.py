import streamlit as st
from functions import *

st.set_page_config(page_title="8!", page_icon=":8ball:", layout="wide", initial_sidebar_state="expanded")

st.title("Group 8!")
st.write("Arnav, Covan, Richelle, Royston, Samuel, Shirlyn, Yi Faye, Zhi Chao")
st.divider()

#Change for your own path
data = pd.read_csv("final_data.csv")
##################################################

with st.container():

    supplier_list = ['MAESA SAS', 'NINGBO SEDUNO IMP & EXP CO.LTD', 'BIGSO AB', 'Bolsius Boxmeer', 'Indo Count Industries Ltd', 'CHROMADURLIN S.A.S', 'DENIMACH LTD', 'Friedola 1888 GmbH', 'CHERRY GROUP CO.,LTD', 'HARDFORD AB']

    st.subheader("Delivery time forecast (Subgroup B Q3)")

    st.write("""
    This graph shows the forecast of expected time for item delivery, for orders placed on a particular date.
    The range of forecast can be adjusted, and expected late deliveries (more than 3 days) will be highlighted.
    A date to order from the supplier will be recommended.
    
    **Assumptions**:
    1. Seller does not have excess stock, and places orders from supplier based on customer demand. This is to lower the risk of customers receiving poor quality items, as food items are perishable.
    2. Deliveries completed within 3 days are considered on time, any longer and the delivery will be considered late.
    3. Constant expected time taken (3 days) for items to reach customers from the seller. Late deliveries are entirely due to supplier bottlenecks.
    """)

    supplier_choice = st.selectbox(label="**Select supplier**", options=supplier_list)

    n_days = st.slider(label="**Select number of days to forecast**", value=1, step=1, min_value=1, max_value=365)
    if supplier_choice and n_days:
        fig, df = get_delivery_predictions(supplier_choice, n_days)
        st.plotly_chart(fig, use_container_width=True)
        st.write("Recommended supply schedule")
        st.dataframe(df, use_container_width=True)

with st.container():
    st.subheader("Average Monthly Revenue")
    st.markdown("""
        This plot shows the average monthly revenue over the years. 
        It helps in understanding the revenue trends and identifying any seasonal patterns or anomalies.
                
        **Note:** The data starts from January 20th, which is why January 2014 has a lower value.
        """)
    fig = plot_monthly_sales_by_year(data)
    st.plotly_chart(fig, use_container_width=True)

with st.container():
        st.subheader("Market Share by Category")
        st.markdown("""
        This plot illustrates the market share distribution across different categories. 
        It provides insights into which categories dominate the market and their relative contributions to the overall revenue.
        """)
        fig = plot_market_share_by_category(data)
        st.plotly_chart(fig, use_container_width=True)   

with st.container():
        st.subheader("Top Consistent Sellers")
        st.markdown("""
        This plot highlights the top consistent sellers by showing how many months each product has been in the top N products. 
        It helps in identifying products that consistently perform well over time.

        Use the slider below to select the number of top products (N) to consider for each month. 
        Adjusting the slider allows you to see how different numbers of top products affect the consistency of the top sellers.
        """)
        n = st.slider(label="**Select number of months for period**", value=1, step=1, min_value=1, max_value=12)
        fig = plot_top_consistent_sellers(data, n)
        st.plotly_chart(fig, use_container_width=True)   

with st.container():
        st.subheader("Monthly sales and decomposition")
        st.markdown("""
        This plot shows the monthly sales data along with its decomposition into trend, seasonality, and residual components. 
        It helps in understanding the underlying patterns and variations in the sales data over time.

        **Note:** The data starts from January 20th, which is why January 2014 has a lower value.
        """)
        fig = plot_monthly_sales_and_decomposition(data)
        st.plotly_chart(fig, use_container_width=True)   

with st.container():
        st.subheader("Revenue Treemap")
        st.markdown("""
        This treemap visualization provides an overview of the revenue distribution across different categories. 
        Each rectangle represents a category, with its size proportional to the revenue it generates. 
        Use this plot to quickly identify the top revenue-generating categories and their relative contributions.
        """)
        fig = plot_revenue_treemap(data)
        st.plotly_chart(fig, use_container_width=True) 