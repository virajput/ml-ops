import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator

from heart_disease.predict import HeartDiseaseModel

from api.schemas import HealthResponse, PatientFeatures, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
)
logger = logging.getLogger("heart_disease_api")

model_holder: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_holder["model"] = HeartDiseaseModel()
        logger.info('"Model loaded successfully"')
    except FileNotFoundError as exc:
        logger.error('"Model not found: %s"' % exc)
        model_holder["model"] = None
    yield


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts the risk of heart disease from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    logger.info(
        '"method": "%s", "path": "%s", "status_code": %d, "duration_ms": %.2f'
        % (request.method, request.url.path, response.status_code, duration_ms)
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model_holder["model"] is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    model = model_holder["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    result = model.predict_one(features.model_dump())
    return PredictionResponse(
        prediction=result["prediction"],
        label="high_risk" if result["prediction"] == 1 else "low_risk",
        probability=round(result["probability"], 4),
    )


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs", "health": "/health"}
