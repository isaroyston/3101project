from sqlalchemy import Column, Integer, String, Float, Boolean, Date

from database.database import Base

class Ecommerce(Base):

    __tablename__ = "DSA3101_TABLE_1"  # Replace with your actual table name

    unique_row = Column(Integer, primary_key=True)
    customer_key = Column(String, nullable=False)
    quantity_purchased = Column(Integer, nullable=False)
    total_price = Column(Float, nullable=False)
    purchase_date = Column(Date, nullable=False)
    time_of_purchase = Column(String, nullable=False)
    item_name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    unit_price = Column(Float, nullable=False)
    manufacturing_country = Column(String, nullable=False)
    supplier = Column(String, nullable=False)
    store_region = Column(String, nullable=False)
    store_district = Column(String, nullable=False)
    store_sub_district = Column(String, nullable=False)
    inventory_level = Column(Integer, nullable=False)
    inventory_cost = Column(Float, nullable=False)
    revenue = Column(Float, nullable=False)
    campaign_key = Column(String, nullable=True)
    mkt_chnl_key = Column(String, nullable=True)
    actual_delivery_time = Column(Integer, nullable=False)
    late_delivery = Column(Boolean, nullable=False)
    customer_reviews = Column(String, nullable=False)