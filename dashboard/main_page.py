import streamlit as st

from functions import *

st.set_page_config(page_title="8!", page_icon=":8ball:", layout="wide", initial_sidebar_state="expanded")

st.title("Group 8!")
st.write("Arnav, Covan, Richelle, Royston, Samuel, Shirlyn, Yi Faye, Zhi Chao")
st.divider()

##################################################

with st.container():

    st.subheader("Delivery time forecast")

    choice = st.slider(label="Select number of days to forecast", value=1, step=1, min_value=1, max_value=28)
    st.plotly_chart(get_delivery_predictions(choice))
