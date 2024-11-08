from fastapi import FastAPI
from database.api_service import app as database_app
from Subgroup_B_Q3.api_service import app as lzc_model_app

app = FastAPI()

# Include individual apps as routers
app.mount("/database", database_app)
app.mount("Subgroup_B_Q3", lzc_model_app)

@app.get("/")
def root():
    return {"message": "Welcome to the main app"}



