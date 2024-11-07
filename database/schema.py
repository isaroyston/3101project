from pydantic import BaseModel
from typing import Optional
from datetime import date

class Ecommerce(BaseModel):

    unique_row: int
    customer_key: str
    quantity_purchased: int
    total_price: float
    purchase_date: date
    time_of_purchase: str
    item_name: str
    description: str
    unit_price: float
    manufacturing_country: str
    supplier: str
    store_region: str
    store_district: str
    store_sub_district: str
    inventory_level: int
    inventory_cost: float
    revenue: float
    campaign_key: Optional[str] = None
    mkt_chnl_key: Optional[str] = None
    actual_delivery_time: int
    late_delivery: bool
    customer_reviews: str

    class Config:
        orm_mode = True