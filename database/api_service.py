from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

import database.models
from database.schema import Ecommerce
from database.database import SessionLocal

app = FastAPI()

def get_db():

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/retrieve_values/", response_model=list[Ecommerce])
def get_values(skip: int, batch_size: int, db: Session = Depends(get_db)):
    return db.query(database.models.Ecommerce).order_by(database.models.Ecommerce.unique_row).offset(skip).limit(batch_size).all()

