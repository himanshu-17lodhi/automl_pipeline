from pydantic import BaseModel
from typing import Optional

class ChurnPredictionInput(BaseModel):
    age: int
    salary: float
    credit_score: int
    account_balance: float
    num_products: int
    has_credit_card: int
    is_active_member: int
    country: str
    gender: str

class PredictionOutput(BaseModel):
    prediction: int
    probability: Optional[float] = None