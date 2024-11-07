from fastapi import FastAPI
from prophet.serialize import model_from_json

with open('./lzc_model/serialized_model.json', 'r') as f:
    prophet_model = model_from_json(f.read())

app = FastAPI()

@app.get("/get_predictions/")
def get_prediction(n_days: int):
    prophet_predicted = prophet_model.predict(
        prophet_model.make_future_dataframe(
            periods=n_days, include_history=False
        )
    )["yhat"].reset_index(drop=True)
    return prophet_predicted.to_json()
