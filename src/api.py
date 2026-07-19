import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from src.triage_pipeline import TriagePipeline

app = FastAPI(
    title="Sahayak Triage API",
    description="FastAPI backend for Sahayak Triage AI decision support.",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None

try:
    print("Loading TriagePipeline in FastAPI server...")
    pipeline = TriagePipeline()
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error initializing TriagePipeline: {e}")

class PatientRequest(BaseModel):
    # Demographics
    age: float = Field(default=35.0, description="Patient age in years")
    gender: str = Field(default="FEMALE", description="MALE or FEMALE")
    
    # Vitals
    triage_vital_hr: Optional[float] = Field(default=None, description="Heart rate (bpm)")
    triage_vital_sbp: Optional[float] = Field(default=None, description="Systolic blood pressure (mmHg)")
    triage_vital_dbp: Optional[float] = Field(default=None, description="Diastolic blood pressure (mmHg)")
    triage_vital_rr: Optional[float] = Field(default=None, description="Respiratory rate (breaths/min)")
    triage_vital_o2: Optional[float] = Field(default=None, description="Oxygen saturation SpO2 (%)")
    triage_vital_temp: Optional[float] = Field(default=None, description="Body temperature (°F)")
    
    # Chief Complaints (0 or 1)
    cc_breathingdifficulty: int = Field(default=0, ge=0, le=1)
    cc_breathingproblem: int = Field(default=0, ge=0, le=1)
    cc_chills: int = Field(default=0, ge=0, le=1)
    cc_coldlikesymptoms: int = Field(default=0, ge=0, le=1)
    cc_cough: int = Field(default=0, ge=0, le=1)
    cc_fever: int = Field(default=0, ge=0, le=1)
    cc_fever_75yearsorolder: int = Field(default=0, ge=0, le=1, alias="cc_fever-75yearsorolder")
    cc_fever_9weeksto74years: int = Field(default=0, ge=0, le=1, alias="cc_fever-9weeksto74years")
    cc_feverimmunocompromised: int = Field(default=0, ge=0, le=1)
    cc_nasalcongestion: int = Field(default=0, ge=0, le=1)
    cc_respiratorydistress: int = Field(default=0, ge=0, le=1)
    cc_shortnessofbreath: int = Field(default=0, ge=0, le=1)
    cc_sorethroat: int = Field(default=0, ge=0, le=1)
    cc_unresponsive: int = Field(default=0, ge=0, le=1)
    cc_urinarytractinfection: int = Field(default=0, ge=0, le=1)
    cc_woundinfection: int = Field(default=0, ge=0, le=1)

    class Config:
        populate_by_name = True

@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Check server logs.")
    return {
        "status": "healthy",
        "model_loaded": pipeline.model is not None,
        "feature_count": len(pipeline.feature_columns),
        "guideline_count": len(pipeline.guideline_embeddings)
    }

@app.post("/predict")
def predict(request: PatientRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    try:
        # Convert request to dictionary, expanding aliases
        input_data = request.model_dump(by_alias=True)
        # Run prediction
        result = pipeline.predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
