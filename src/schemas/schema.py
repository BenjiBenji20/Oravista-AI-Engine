"""
Preventative and Predictive Analytics Module — Pydantic Models
Aligned to:
  - Feature requirements (image)
  - users table schema (MySQL)
  - patient_analytics table (new, defined below)
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared Enums (mirror DB enums/constraints)
# ---------------------------------------------------------------------------

class UserRole(str, Enum):
    patient = "patient"
    admin = "admin"
    staff = "staff"
    dentist = "dentist"


class RiskLevel(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"


class AppointmentDayOfWeek(str, Enum):
    monday = "Monday"
    tuesday = "Tuesday"
    wednesday = "Wednesday"
    thursday = "Thursday"
    friday = "Friday"
    saturday = "Saturday"
    sunday = "Sunday"


# ---------------------------------------------------------------------------
# User model (mirrors `users` table — read-only reference; no password exposed)
# ---------------------------------------------------------------------------

class UserBase(BaseModel):
    """Mirrors the `users` table for safe read operations (no password)."""

    id: int
    first_name: str = Field(..., max_length=20)
    last_name: str = Field(..., max_length=20)
    email: str = Field(..., max_length=50)
    role: UserRole = UserRole.patient
    branch: str = Field(default="Main Branch", max_length=100)
    sex: Optional[str] = Field(None, max_length=10)
    dob: Optional[str] = Field(None, max_length=20)          # stored as varchar in DB
    age: Optional[int] = None
    phone: Optional[str] = Field(None, max_length=20)
    occupation: Optional[str] = Field(None, max_length=100)
    blood_type: Optional[str] = Field(default="O+", max_length=5)
    allergies: Optional[str] = None                          # TEXT in DB
    insurance: Optional[str] = Field(None, max_length=100)
    policy_number: Optional[str] = Field(None, max_length=100)
    specialty: str = Field(default="General Dentistry", max_length=100)
    status: str = Field(default="Available", max_length=20)
    profile_picture: Optional[str] = Field(None, max_length=255)
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# 1. Patient Oral Health Risk Score
#    Aggregates lifestyle data, genetics, and clinical history
#    to generate a numerical health grade for each patient.
# ---------------------------------------------------------------------------

class OralHealthRiskRequest(BaseModel):
    """
    All fields sourced from the `users` table + patient_analytics table.
    `patient_id` maps to `users.id`.
    """
    patient_id: int                            # users.id (int, not str)
    age: int = Field(..., ge=1, le=120)        # users.age
    sex: Optional[str] = None                  # users.sex
    blood_type: Optional[str] = None           # users.blood_type
    allergies: Optional[str] = None            # users.allergies (TEXT)
    occupation: Optional[str] = None           # users.occupation — occupational hazards

    # From patient_analytics table
    sugar_intake_score: int = Field(..., ge=0, le=10,
        description="Daily sugar intake on a 0–10 scale")
    brushing_frequency: int = Field(..., ge=0, le=5,
        description="Times patient brushes teeth per day")
    flossing_frequency: int = Field(default=0, ge=0, le=3,
        description="Times patient flosses per day")
    smoking: bool = Field(...,
        description="Whether the patient currently smokes")
    alcohol_use: bool = Field(default=False)
    previous_cavities: int = Field(..., ge=0)
    previous_extractions: int = Field(default=0, ge=0)
    family_history_dental_disease: bool = Field(default=False,
        description="Genetic / family history flag")
    last_dental_visit_months_ago: Optional[int] = Field(None, ge=0)
    medical_history_notes: str = Field(default="",
        description="Free-text notes from clinical history")


class LLMCheckupAssessment(BaseModel):
    risk_score: int = Field(..., description="0-100 composite risk score")
    health_grade: str = Field(..., description="e.g., 'Healthy', 'High Risk'")
    risk_level: RiskLevel
    disease_progression_forecast: str = Field(..., description="Likely disease progression")
    recommended_action: str = Field(..., description="Personalized prevention plan")


class OralHealthRiskResponse(BaseModel):
    patient_id: int
    risk_score: int = Field(..., ge=0, le=100,
        description="Composite score: 0=Healthy, 100=Extreme Risk")
    health_grade: str = Field(...,
        description="e.g. 'Healthy', 'Moderate Risk', 'High Risk'")
    risk_level: RiskLevel
    disease_progression_forecast: str = Field(...,
        description="LLM-generated narrative of likely disease progression")
    recommended_action: str = Field(...,
        description="LLM-generated personalised prevention plan")
    generated_at: datetime = Field(default_factory=datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 2. Treatment Outcome Prediction
#    Uses historical data to estimate the success probability
#    of specific procedures.
# ---------------------------------------------------------------------------

class TreatmentOutcomeRequest(BaseModel):
    patient_id: int                             # users.id
    procedure_name: str = Field(..., max_length=150)
    patient_age: int = Field(..., ge=1, le=120) # users.age
    sex: Optional[str] = None                  # users.sex
    blood_type: Optional[str] = None           # users.blood_type
    allergies: Optional[str] = None            # users.allergies
    insurance: Optional[str] = None            # users.insurance
    medical_history_flags: List[str] = Field(default_factory=list,
        description="Known conditions, e.g. ['diabetes', 'hypertension']")
    previous_procedure_outcomes: Optional[str] = Field(None,
        description="Short narrative of past relevant procedures")


class TreatmentOutcomeResponse(BaseModel):
    patient_id: int
    procedure_name: str
    success_probability: float = Field(..., ge=0.0, le=100.0,
        description="Percentage probability of success (0–100)")
    confidence_level: str = Field(...,
        description="e.g. 'High', 'Medium', 'Low' model confidence")
    key_factors: str = Field(...,
        description="Main factors influencing the prediction")
    recommendation: str = Field(...,
        description="LLM-generated recommendation for the dentist")
    generated_at: datetime = Field(default_factory=datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 3. Disease Progression Forecasting
#    Projects how a current condition might evolve if left untreated.
# ---------------------------------------------------------------------------

class DiseaseProgressionRequest(BaseModel):
    patient_id: int                             # users.id
    current_condition: str = Field(..., max_length=200)
    severity: RiskLevel = Field(default=RiskLevel.medium,
        description="Current severity level of the condition")
    months_untreated: int = Field(..., ge=0,
        description="Simulated months without treatment")
    patient_age: Optional[int] = Field(None, ge=1, le=120)  # users.age
    smoking: bool = Field(default=False)
    diabetes: bool = Field(default=False,
        description="Known to accelerate oral disease")
    medical_history_notes: str = Field(default="")


class DiseaseProgressionResponse(BaseModel):
    patient_id: int
    current_condition: str
    severity_at_start: RiskLevel
    projected_severity: RiskLevel = Field(...,
        description="Predicted severity after `months_untreated`")
    progression_forecast: str = Field(...,
        description="LLM-generated step-by-step disease timeline")
    recommended_intervention: str = Field(...,
        description="LLM-generated urgency and treatment suggestion")
    forecast_horizon_months: int
    generated_at: datetime = Field(default_factory=datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 4. Risk Stratification
#    Segregates the patient database into high, medium, and low-risk
#    categories, allowing the clinic to prioritise urgent care.
# ---------------------------------------------------------------------------

class RiskStratificationRequest(BaseModel):
    branch: Optional[str] = Field(None, max_length=100,
        description="Filter by users.branch; None = all branches")
    timeframe_days: int = Field(default=30, ge=1,
        description="Look-back window for scoring recent patients")
    include_patient_list: bool = Field(default=False,
        description="If True, response includes per-patient breakdown")


class PatientRiskSummary(BaseModel):
    """Embedded per-patient row when include_patient_list=True."""
    patient_id: int
    full_name: str
    risk_score: int
    risk_level: RiskLevel
    last_visit_date: Optional[date] = None


class RiskStratificationResponse(BaseModel):
    branch: Optional[str] = None
    timeframe_days: int
    total_patients_analyzed: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    low_risk_pct: float = Field(..., ge=0.0, le=100.0)
    medium_risk_pct: float = Field(..., ge=0.0, le=100.0)
    high_risk_pct: float = Field(..., ge=0.0, le=100.0)
    high_risk_patients: Optional[List[PatientRiskSummary]] = None
    generated_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    @model_validator(mode="after")
    def percentages_sum_to_100(self) -> "RiskStratificationResponse":
        total = self.low_risk_pct + self.medium_risk_pct + self.high_risk_pct
        if self.total_patients_analyzed > 0 and not (99.9 <= total <= 100.1):
            raise ValueError(f"Risk percentages must sum to ~100, got {total}")
        return self


# ---------------------------------------------------------------------------
# 5. No-Show Prediction
#    Analyses past attendance patterns and external factors to flag
#    appointments with a high probability of being missed,
#    triggering automated reminders.
# ---------------------------------------------------------------------------

class NoShowPredictionRequest(BaseModel):
    appointment_id: str = Field(..., max_length=50)
    patient_id: int                             # users.id
    branch: Optional[str] = Field(None, max_length=100)     # users.branch

    # Appointment context
    appointment_day_of_week: AppointmentDayOfWeek
    appointment_time_of_day: Optional[str] = Field(None,
        description="e.g. 'Morning', 'Afternoon', 'Evening'")

    # Patient factors (sourced from users + patient_analytics tables)
    travel_distance_km: float = Field(..., ge=0.0)
    historical_missed_appointments: int = Field(..., ge=0,
        description="Count of past missed appointments for this patient")
    total_past_appointments: int = Field(..., ge=0,
        description="Total appointments ever scheduled for this patient")
    insurance: Optional[str] = None            # users.insurance
    age: Optional[int] = Field(None, ge=1, le=120)  # users.age
    occupation: Optional[str] = None          # users.occupation

    # External / contextual factors
    days_until_appointment: int = Field(..., ge=0)
    reminder_sent: bool = Field(default=False)
    weather_risk_flag: bool = Field(default=False,
        description="True if inclement weather is forecast")

    @field_validator("total_past_appointments")
    @classmethod
    def total_gte_missed(cls, v: int, info) -> int:
        missed = info.data.get("historical_missed_appointments", 0)
        if v < missed:
            raise ValueError(
                "total_past_appointments cannot be less than "
                "historical_missed_appointments"
            )
        return v


class NoShowPredictionResponse(BaseModel):
    appointment_id: str = Field(..., max_length=20)
    patient_id: int
    no_show_probability: float = Field(..., ge=0.0, le=100.0,
        description="Probability (0–100) the patient will miss the appointment")
    risk_flag: RiskLevel
    reasoning: str = Field(...,
        description="LLM-generated explanation of top contributing factors")
    automated_reminder_triggered: bool = Field(...,
        description="True if probability exceeded the clinic's threshold")
    generated_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    
    
class AnalyticsPromptLog(BaseModel):
    module: str
    reference_id: int
    prompt_sent: str
    raw_llm_response: str
    model_used: str = "claude-sonnet-4-20250514"
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    