import streamlit as st
from functions import *

st.set_page_config(page_title="8!", page_icon=":8ball:", layout="wide", initial_sidebar_state="expanded")

st.title("Group 8!")
st.write("Arnav, Covan, Richelle, Royston, Samuel, Shirlyn, Yi Faye, Zhi Chao")
st.divider()

##################################################

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

with st.container():

    supplier_list = ['MAESA SAS', 'NINGBO SEDUNO IMP & EXP CO.LTD', 'BIGSO AB', 'Bolsius Boxmeer', 'Indo Count Industries Ltd', 'CHROMADURLIN S.A.S', 'DENIMACH LTD', 'Friedola 1888 GmbH', 'CHERRY GROUP CO.,LTD', 'HARDFORD AB']

    st.subheader("Delivery time forecast (Subgroup B Q3)")

    st.markdown("""
    **Assumptions**:
    1. Seller does not have excess stock, and places orders from supplier based on customer demand. This is to lower the risk of customers receiving poor quality items, as food items are perishable.
    2. Deliveries completed within 3 days are considered on time, any longer and the delivery will be considered late.
    3. Constant expected time taken (3 days) for items to reach customers from the seller. Late deliveries are entirely due to supplier bottlenecks. 
    """)

    supplier_name = st.selectbox(label="**Select supplier**", options=supplier_list)

    n_days = st.slider(label="**Select number of days to forecast**", value=1, step=1, min_value=1, max_value=365)
    if supplier_name and n_days:
        fig, df = get_delivery_predictions(supplier_name, n_days)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

