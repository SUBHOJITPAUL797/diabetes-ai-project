"""
Diabetes Prediction API - Production Ready
FastAPI backend for diabetes prediction using pre-trained sklearn model

Features:
- Loads pre-trained sklearn model from pickle file
- Input validation using Pydantic models
- CORS enabled for frontend integration
- Comprehensive error handling & logging
- Health check endpoint
- Single & batch predictions
- Model information endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import numpy as np
import pickle
import joblib
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="ML API for predicting diabetes risk using pre-trained sklearn model",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration - Allow Cloudflare frontend
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",      # Vite dev server
    "http://localhost:8080",      # Webpack dev server
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "https://*.cloudflare.com",   # Cloudflare Pages
    "*"  # Development only - restrict in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Model Loading ====================

MODEL = None
MODEL_METADATA = {}
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

FEATURE_BOUNDS = {
    "Pregnancies": (0, 17),
    "Glucose": (0, 200),
    "BloodPressure": (0, 122),
    "Insulin": (0, 846),
    "BMI": (0, 67),
    "DiabetesPedigreeFunction": (0, 2.42),
    "Age": (21, 81)
}


def load_model():
    """
    Load sklearn model from pickle/joblib file
    Supports both .pkl and .joblib formats
    """
    global MODEL, MODEL_METADATA
    
    model_paths = [
        "models/diabetes_model.pkl",
        "models/diabetes_model.joblib",
        "diabetes_model.pkl",
        "diabetes_model.joblib",
        "./models/diabetes_model.pkl",
        "./models/diabetes_model.joblib",
    ]
    
    model_loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.joblib'):
                    MODEL = joblib.load(path)
                else:
                    with open(path, 'rb') as f:
                        MODEL = pickle.load(f)
                
                logger.info(f"✅ Model loaded successfully from: {path}")
                logger.info(f"Model type: {type(MODEL).__name__}")
                
                # Extract model metadata
                if hasattr(MODEL, 'coef_'):
                    MODEL_METADATA['coefficients'] = MODEL.coef_[0].tolist()
                    logger.info(f"Coefficients shape: {MODEL.coef_.shape}")
                
                if hasattr(MODEL, 'intercept_'):
                    MODEL_METADATA['intercept'] = float(MODEL.intercept_[0] if isinstance(MODEL.intercept_, np.ndarray) else MODEL.intercept_)
                    logger.info(f"Intercept: {MODEL_METADATA['intercept']}")
                
                if hasattr(MODEL, 'classes_'):
                    MODEL_METADATA['classes'] = MODEL.classes_.tolist()
                
                model_loaded = True
                break
                
            except Exception as e:
                logger.error(f"Error loading model from {path}: {str(e)}")
                continue
    
    if not model_loaded:
        logger.warning("⚠️  Model not found. Using mock mode for testing.")
        logger.warning("Please place your model file at: models/diabetes_model.pkl or models/diabetes_model.joblib")
        MODEL_METADATA['mock_mode'] = True


# Load model on startup
load_model()



# ==================== Pydantic Models ====================

class PredictionInput(BaseModel):
    """Input model for diabetes prediction"""
    Pregnancies: int = Field(..., ge=0, le=17, description="Number of pregnancies")
    Glucose: int = Field(..., ge=0, le=200, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: int = Field(..., ge=0, le=122, description="Diastolic blood pressure (mmHg)")
    Insulin: int = Field(..., ge=0, le=846, description="2-hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0, le=67, description="Body mass index (weight in kg/height in m²)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=2.42, description="Diabetes pedigree function")
    Age: int = Field(..., ge=21, le=81, description="Age in years")

    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "Insulin": 35,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }
        }

    @validator("BMI")
    def validate_bmi(cls, v):
        if v < 10 or v > 70:
            logger.warning(f"Unusual BMI value: {v}")
        return v

    @validator("Glucose")
    def validate_glucose(cls, v):
        if v < 70 or v > 180:
            logger.warning(f"Unusual glucose level: {v}")
        return v


class PredictionOutput(BaseModel):
    """Output model for prediction response"""
    prediction: int = Field(..., description="Prediction: 1 (Diabetic) or 0 (Non-Diabetic)")
    probability: float = Field(..., description="Probability of diabetes (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    confidence: float = Field(..., description="Model confidence in prediction (%)")
    input_features: Dict[str, Any]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    model_loaded: bool


class BatchPredictionInput(BaseModel):
    """Input model for batch predictions"""
    predictions: list[PredictionInput] = Field(..., min_items=1, max_items=1000)


class BatchPredictionOutput(BaseModel):
    """Output model for batch predictions"""
    total: int
    successful: int
    failed: int
    results: list[PredictionOutput]
    processing_time: float




# ==================== Utility Functions ====================

def calculate_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability < 0.30:
        return "Low"
    elif probability < 0.70:
        return "Medium"
    else:
        return "High"


def predict_diabetes(input_data: PredictionInput) -> tuple[int, float]:
    """
    Predict diabetes using the loaded sklearn model
    
    Args:
        input_data: PredictionInput model with patient features
        
    Returns:
        tuple: (prediction, probability)
        
    Raises:
        ValueError: If model is not loaded or prediction fails
    """
    if MODEL is None:
        raise ValueError("Model not loaded. Please ensure your model file is in the correct location.")
    
    try:
        # Prepare features in correct order
        features = np.array([[
            input_data.Pregnancies,
            input_data.Glucose,
            input_data.BloodPressure,
            input_data.Insulin,
            input_data.BMI,
            input_data.DiabetesPedigreeFunction,
            input_data.Age
        ]])
        
        # Make prediction
        prediction = MODEL.predict(features)[0]
        
        # Get probability if available
        if hasattr(MODEL, 'predict_proba'):
            probabilities = MODEL.predict_proba(features)[0]
            probability = float(probabilities[1])  # Probability of class 1 (diabetic)
        else:
            probability = float(prediction)
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        return int(prediction), probability
        
    except Exception as e:
        logger.error(f"Error in diabetes prediction: {str(e)}")
        raise ValueError(f"Prediction failed: {str(e)}")


# ==================== API Endpoints ====================

@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check"
)
async def health_check():
    """
    Health check endpoint to verify API is running and model is loaded
    
    Returns:
        HealthResponse: Status and version information
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        model_loaded=MODEL is not None
    )


@app.post(
    "/api/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
    summary="Single Diabetes Prediction",
    status_code=200
)
async def predict_single(input_data: PredictionInput):
    """
    Predict diabetes risk for a single patient
    
    Takes patient health metrics and returns diabetes prediction with confidence score.
    
    Args:
        input_data: Patient health metrics
        
    Returns:
        PredictionOutput: Prediction result with probability and risk level
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        if MODEL is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )
        
        prediction, probability = predict_diabetes(input_data)
        risk_level = calculate_risk_level(probability)
        confidence = abs(probability - 0.5) * 200  # Convert to 0-100%
        
        logger.info(f"Successfully predicted: {prediction}")
        
        return PredictionOutput(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            confidence=round(confidence, 2),
            input_features=input_data.dict(),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post(
    "/api/predict-batch",
    response_model=BatchPredictionOutput,
    tags=["Prediction"],
    summary="Batch Diabetes Predictions",
    status_code=200
)
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Predict diabetes risk for multiple patients in batch
    
    Allows batch processing of up to 1000 patient records.
    
    Args:
        batch_input: List of patient health metrics
        
    Returns:
        BatchPredictionOutput: Batch results with individual predictions
    """
    import time
    start_time = time.time()
    
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    results = []
    failed_count = 0
    
    try:
        for idx, input_data in enumerate(batch_input.predictions):
            try:
                prediction, probability = predict_diabetes(input_data)
                risk_level = calculate_risk_level(probability)
                confidence = abs(probability - 0.5) * 200
                
                results.append(PredictionOutput(
                    prediction=prediction,
                    probability=round(probability, 4),
                    risk_level=risk_level,
                    confidence=round(confidence, 2),
                    input_features=input_data.dict(),
                    timestamp=datetime.utcnow().isoformat()
                ))
                
            except Exception as e:
                failed_count += 1
                logger.warning(f"Batch item {idx} failed: {str(e)}")
        
        processing_time = time.time() - start_time
        
        return BatchPredictionOutput(
            total=len(batch_input.predictions),
            successful=len(results),
            failed=failed_count,
            results=results,
            processing_time=round(processing_time, 4)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.get(
    "/api/model-info",
    tags=["Model"],
    summary="Get Model Information"
)
async def get_model_info():
    """
    Get model information including coefficients and feature bounds
    
    Returns:
        Dictionary containing model details and feature bounds
    """
    if MODEL is None:
        return {
            "status": "model_not_loaded",
            "message": "Model file not found. Place your model at models/diabetes_model.pkl"
        }
    
    return {
        "status": "loaded",
        "model_type": type(MODEL).__name__,
        "features": FEATURE_NAMES,
        "feature_bounds": FEATURE_BOUNDS,
        "metadata": MODEL_METADATA,
        "classes": getattr(MODEL, 'classes_', [0, 1]).tolist(),
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Diabetes Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs",
        "health": "/api/health",
        "endpoints": {
            "single_prediction": "POST /api/predict",
            "batch_prediction": "POST /api/predict-batch",
            "model_info": "GET /api/model-info",
            "health_check": "GET /api/health"
        }
    }



# ==================== Startup & Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 Diabetes Prediction API Started")
    logger.info("=" * 60)
    if MODEL is not None:
        logger.info("✅ Model loaded successfully")
        logger.info(f"Features: {', '.join(FEATURE_NAMES)}")
    else:
        logger.warning("⚠️  Model not loaded - running in demo mode")
    logger.info("📚 API Docs: http://localhost:8000/api/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Diabetes Prediction API Stopped")


# ==================== Run ====================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
