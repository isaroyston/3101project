import json
import requests
from datetime import datetime, timedelta
import plotly.express as px
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
