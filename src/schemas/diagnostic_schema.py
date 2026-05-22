from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class PathologyPrediction(BaseModel):
    class_id: int = Field(..., description="The numerical ID mapping to the specific pathology enum.")
    name: str = Field(..., description="Human-readable string name of the condition (e.g., Caries, Bone Loss).")
    confidence: float = Field(..., description="Model prediction probability score between 0.0 and 1.0.")

class DiagnosticUploadResponse(BaseModel):
    diagnostic_id: int = Field(..., description="Database tracking ID for the created AI diagnostic entry.")
    patient_id: int = Field(..., description="The associated patient's unique identifier.")
    file_path: str = Field(..., description="Storage destination URI path of the uploaded dental scan image.")
    predictions: List[PathologyPrediction] = Field(..., description="List of all detected dental conditions exceeding the tracking threshold.")
    gradcam_grid: List[List[float]] = Field(..., description="Lightweight 2D array matrix of float intensities used by React to render the canvas overlay.")
    scan_date: datetime

class UpdateAnnotationRequest(BaseModel):
    clinical_notes: Optional[str] = Field(None, description="The licensed dentist's professional clinical validation remarks.")
    human_verified_findings: Dict[str, Any] = Field(
        ..., 
        description="JSON object capturing manual markings, bounding box corrections, or verified class state changes."
    )

class AIDiagnosticResponse(BaseModel):
    id: int
    patient_id: int
    clinical_notes: Optional[str]
    ai_findings: Dict[str, Any]
    scan_date: datetime

    class Config:
        from_attributes = True
        