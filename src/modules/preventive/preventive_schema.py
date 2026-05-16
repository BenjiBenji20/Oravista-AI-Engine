from dataclasses import Field
from pydantic import BaseModel

class RiskPredictionRequest(BaseModel):
    patient_id: str


class RiskPredictionResponse(BaseModel):
    status: str | None = Field(..., examples=["success", "failed"])
    risk_score: int | None = 1
    disease_progression: str | None = None
    recommended_action:  str | None = None
  