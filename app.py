import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# =====================================================
# Initialize FastAPI App
# =====================================================
app = FastAPI(
    title="Churn Prediction API",
    description="API for customer churn prediction using XGBoost",
    version="1.0.0",
)

# =====================================================
# Load Model (run once at startup)
# =====================================================
MODEL_PATH = Path(__file__).parent / "api" / "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# =====================================================
# Define Input Schema (Pydantic)
# =====================================================
class CustomerData(BaseModel):
    """Schema for customer input data"""
    tenure: int
    MonthlyCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 75.5,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
            }
        }

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: int
    churn_probability: float
    message: str

# =====================================================
# Health Check Endpoint
# =====================================================
@app.get("/health")
def health_check():
    """Check if API and model are ready"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return {"status": "healthy", "model_loaded": True}

# =====================================================
# Prediction Endpoint
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    """
    Predict whether a customer will churn or not.
    
    - **prediction**: 1 = Churn, 0 = Stay
    - **churn_probability**: probability of the customer churning
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of the positive class (Churn)

        # Generate message
        if prediction == 1:
            message = f"⚠️ Customer predicted to CHURN with {probability:.2%} probability"
        else:
            message = f"✅ Customer predicted to NOT CHURN with {1-probability:.2%} probability"

        return PredictionResponse(
            prediction=int(prediction),
            churn_probability=round(float(probability), 4),
            message=message,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")