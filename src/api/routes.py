from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.api.utils import load_model, make_prediction

router = APIRouter()

# Load model once when API starts
model = load_model()

# Pydantic schema for request validation
class ChurnInput(BaseModel):
    tenure: int = Field(..., example=12)
    MonthlyCharges: float = Field(..., example=70.0)
    TotalCharges: float = Field(..., example=840.0)

@router.get("/health")
def health_check():
    return {"status": "API is healthy"}

@router.post("/predict")
def predict_churn(input_data: ChurnInput):
    try:
        # Convert Pydantic object to dict
        input_dict = input_data.dict()
        # Predict and log automatically
        prediction = make_prediction(model, input_dict)
        return {"churn_prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
