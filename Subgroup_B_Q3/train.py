import requests
import pandas as pd

from prophet import Prophet
from prophet.serialize import model_to_json

skip = 0
batch_size = 10000
df_list = []

while True:
    res = requests.get(f"http://127.0.0.1:8000/database/retrieve_values/?skip={skip}&batch_size={batch_size}")
    if not res.json():
        break
    df_list.extend(res.json())
    skip += len(res.json())
    print(f"Retrieved {skip} rows")

df = pd.DataFrame(df_list)
supplier_list = list(df['supplier'].unique())

for _supplier in supplier_list:
    df_model = df[df['supplier'] == _supplier]
    df_model = df_model[["purchase_date", "actual_delivery_time"]].groupby(by="purchase_date", as_index=False).mean()
    df_model = df_model.rename(columns={"purchase_date":"ds", "actual_delivery_time": "y"}, inplace=False)

    prophet_model = Prophet()
    print(f"Training {_supplier} model for {len(df_model)} data points")
    prophet_model.fit(df_model)

    with open(f'./models/serialized_model_{_supplier}.json', 'w') as f:
        f.write(model_to_json(prophet_model))