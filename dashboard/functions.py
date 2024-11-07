import json
import requests
from datetime import datetime, timedelta
import plotly.express as px

def get_delivery_predictions(n_days):

    res = requests.get(f"http://127.0.0.1:8000/lzc_model/get_predictions/?n_days={n_days}")

    res_dict = json.loads(res.json())

    x = [datetime.today() + timedelta(days=i) for i in range(1, n_days + 1)]
    y = [i[1] for i in list(res_dict.items())]

    fig = px.line(x=x, y=y)
    fig.update_layout(xaxis_title="Date", yaxis_title="Average delivery time (days)", showlegend=False)

    return fig
