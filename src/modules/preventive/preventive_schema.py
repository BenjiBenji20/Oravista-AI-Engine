from pydantic import BaseModel
from typing import List, Optional

from pydantic import BaseModel
from typing import List, Optional

class OralHealthRiskRequest(BaseModel):
    patient_id: str
    age: int
    sugar_intake: int
    brushing_freq: int
    smoking: int
    previous_cavities: int
    medical_history: str = ""

class OralHealthRiskResponse(BaseModel):
    patient_id: str
    risk_score: int # LLM or Joblib calculated (0-100)
    health_grade: str # e.g., "High Risk", "Moderate", "Healthy"
    disease_progression_forecast: str # LLM generated narrative
    recommended_action: str # LLM generated prevention plan


class TreatmentOutcomeRequest(BaseModel):
    patient_id: str
    procedure_name: str
    patient_age: int
    medical_history_flags: List[str]


class TreatmentOutcomeResponse(BaseModel):
    procedure_name: str
    success_probability: float # 0.0 to 100.0
    key_factors: str
    recommendation: str


class DiseaseProgressionRequest(BaseModel):
    patient_id: str
    current_condition: str
    months_untreated: int

class DiseaseProgressionResponse(BaseModel):
    patient_id: str
    current_condition: str
    progression_forecast: str
    recommended_intervention: str


class RiskStratificationRequest(BaseModel):
    clinic_id: str
    timeframe_days: int = 30

class RiskStratificationResponse(BaseModel):
    low_risk_pct: float
    medium_risk_pct: float
    high_risk_pct: float
    total_patients_analyzed: int


class NoShowPredictionRequest(BaseModel):
    appointment_id: str
    patient_id: str
    travel_distance_km: float
    historical_missed_appts: int
    appointment_day_of_week: str

class NoShowPredictionResponse(BaseModel):
    appointment_id: str
    no_show_probability: float # 0.0 to 100.0
    risk_flag: str # "High", "Medium", "Low"
    reasoning: str
