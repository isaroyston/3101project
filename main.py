from fastapi import FastAPI
from database.api_service import app as database_app
from lzc_model.api_service import app as lzc_model_app

app = FastAPI()

# Include individual apps as routers
app.mount("/database", database_app)
app.mount("/lzc_model", lzc_model_app)

@app.get("/")
def root():
    return {"message": "Welcome to the main app"}



