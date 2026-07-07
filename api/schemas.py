from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Age in years")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    thalach: float = Field(..., ge=0, description="Max heart rate achieved")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes)")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored (0-4)")
    thal: int = Field(..., description="Thalassemia: 3=normal, 6=fixed defect, 7=reversible")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "trestbps": 145,
                "chol": 233,
                "thalach": 150,
                "oldpeak": 2.3,
                "sex": 1,
                "cp": 1,
                "fbs": 1,
                "restecg": 2,
                "exang": 0,
                "slope": 3,
                "ca": 0,
                "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = low risk, 1 = high risk of heart disease")
    label: str
    probability: float = Field(..., description="Predicted probability of heart disease")


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
