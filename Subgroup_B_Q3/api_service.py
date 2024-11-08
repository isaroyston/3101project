from fastapi import FastAPI
from prophet.serialize import model_from_json
from pydantic import BaseModel

app = FastAPI()

class InferenceData(BaseModel):
    supplier_name: str
    n_days: int

@app.post("/make_predictions/")
def make_predictions(data: InferenceData):

    inference_data = data.model_dump()

    with open(f'./Subgroup_B_Q3/models/serialized_model_{inference_data["supplier_name"]}.json', 'r') as f:
        prophet_model = model_from_json(f.read())

    prophet_predicted = prophet_model.predict(
        prophet_model.make_future_dataframe(
            periods=inference_data["n_days"], include_history=False
        )
    )["yhat"].reset_index(drop=True)

    return {"predictions": prophet_predicted.to_json()}
