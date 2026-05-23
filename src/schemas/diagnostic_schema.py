from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class BoundingBox(BaseModel):
    x_min: float = Field(..., description="Normalized top-left X coordinate axis anchor point (0.0 to 1.0).")
    y_min: float = Field(..., description="Normalized top-left Y coordinate axis anchor point (0.0 to 1.0).")
    width: float = Field(..., description="Normalized width scale representation of the bounding region (0.0 to 1.0).")
    height: float = Field(..., description="Normalized height scale representation of the bounding region (0.0 to 1.0).")

class PathologyPrediction(BaseModel):
    class_id: int = Field(..., description="The numerical ID mapping to the specific pathology enum index.")
    name: str = Field(..., description="Human-readable string name of the condition (e.g., Caries, Bone Loss).")
    confidence: float = Field(..., description="Model prediction probability score between 0.0 and 1.0.")
    box: Optional[BoundingBox] = Field(None, description="The precise coordinate box isolating the pathology location region.")

class DiagnosticUploadResponse(BaseModel):
    diagnostic_id: int = Field(..., description="Database tracking ID for the created AI diagnostic entry.")
    patient_id: int = Field(..., description="The associated patient's unique identifier.")
    file_path: str = Field(..., description="Storage destination public URL link of the uploaded dental scan image.")
    predictions: List[PathologyPrediction] = Field(..., description="List of all isolated conditions accompanied by localization spatial blocks.")
    scan_date: datetime

class UpdateAnnotationRequest(BaseModel):
    clinical_notes: Optional[str] = Field(None, description="The licensed dentist's professional clinical validation remarks.")
    human_verified_findings: Dict[str, Any] = Field(
        ..., 
        description="JSON array capturing manual markings, bounding box adjustments, or verified class state changes."
    )

class AIDiagnosticResponse(BaseModel):
    id: int
    patient_id: int
    clinical_notes: Optional[str]
    ai_findings: Dict[str, Any]
    scan_date: datetime

    class Config:
        from_attributes = True