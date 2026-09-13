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
    description="API untuk prediksi churn pelanggan menggunakan XGBoost",
    version="1.0.0",
)

# =====================================================
# Load Model (run once at startup)
# =====================================================
MODEL_PATH = Path(__file__).parent / "api" / "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model berhasil dimuat dari {MODEL_PATH}")
except Exception as e:
    print(f"❌ Gagal load model: {e}")
    model = None


# =====================================================
# Define Input Schema (Pydantic)
# =====================================================
class CustomerData(BaseModel):
    """Schema untuk input data customer"""
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
    """Schema untuk response prediksi"""
    prediction: int
    churn_probability: float
    message: str


# =====================================================
# Health Check Endpoint
# =====================================================
@app.get("/health")
def health_check():
    """Check apakah API dan model ready"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia")
    return {"status": "healthy", "model_loaded": True}


# =====================================================
# Prediction Endpoint
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    """
    Prediksi apakah customer akan churn atau tidak.
    
    - **prediction**: 1 = Churn, 0 = Stay
    - **churn_probability**: probabilitas customer akan churn
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia")

    try:
        # Convert input ke DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probabilitas kelas positif (Churn)

        # Generate message
        if prediction == 1:
            message = f"⚠️ Customer diprediksi CHURN dengan probabilitas {probability:.2%}"
        else:
            message = f"✅ Customer diprediksi TIDAK CHURN dengan probabilitas {1-probability:.2%}"

        return PredictionResponse(
            prediction=int(prediction),
            churn_probability=round(float(probability), 4),
            message=message,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat prediksi: {str(e)}")