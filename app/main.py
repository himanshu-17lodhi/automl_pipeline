import pandas as pd
import mlflow.pyfunc
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

class PredictionRequest(BaseModel):
    age: int = Field(..., example=42)
    salary: float = Field(..., example=60000.0)
    credit_score: int = Field(..., example=700)
    account_balance: float = Field(..., example=50000.0)
    num_products: int = Field(..., example=2)
    has_credit_card: int = Field(..., example=1)
    is_active_member: int = Field(..., example=1)
    country: str = Field(..., example="Germany")
    gender: str = Field(..., example="Female")
    tenure: int = Field(..., example=5)

class PredictionResponse(BaseModel):
    prediction: int

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model_name = "Production_Model"
        logger.info(f"Loading model: {model_name}...")
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Latest")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model. Predictions will fail. Error: {e}")
    yield
    logger.info("API Shutting down.")

app = FastAPI(title="AutoML Churn API", version="1.0.0", lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
    try:
        data = pd.DataFrame([request.model_dump()])
        result = model.predict(data)
        pred = int(result[0]) if hasattr(result, '__getitem__') else int(result)
        return {"prediction": pred}
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")