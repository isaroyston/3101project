import yaml

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

with open("./config/db.yaml", 'r') as f:
    db_config = yaml.safe_load(f)

DATABASE_URL = db_config["database"]["url"]

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=True, bind=engine)
Base = declarative_base()