from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from .utils import load_model

router = APIRouter()
model = load_model()

class ChurnRequest(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float

@router.get("/")
def read_root():
    return {"message": "Model API is running"}

@router.post("/predict")
def predict_churn(req: ChurnRequest):

    input_df = pd.DataFrame([{
        "tenure": req.tenure,
        "MonthlyCharges": req.MonthlyCharges,
        "TotalCharges": req.TotalCharges
    }])

    prediction = model.predict(input_df)[0]

    return {
        "churn_prediction": int(prediction)
    }
