from src.database.base import Base
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, Boolean,
    Enum, DateTime, Date, Numeric, SmallInteger,
    ForeignKey, UniqueConstraint, func
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
import enum


# ---------------------------------------------------------------------------
# Enums — mirror DB ENUM columns
# ---------------------------------------------------------------------------

class RiskLevelEnum(str, enum.Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"


class AppointmentDayEnum(str, enum.Enum):
    Monday = "Monday"
    Tuesday = "Tuesday"
    Wednesday = "Wednesday"
    Thursday = "Thursday"
    Friday = "Friday"
    Saturday = "Saturday"
    Sunday = "Sunday"


class AnalyticsModuleEnum(str, enum.Enum):
    oral_health_risk = "oral_health_risk"
    treatment_outcome = "treatment_outcome"
    disease_progression = "disease_progression"
    risk_stratification = "risk_stratification"
    no_show_prediction = "no_show_prediction"
    
    
class UserRoleEnum(enum.Enum):
    patient = "patient"
    admin = "admin"
    staff = "staff"
    dentist = "dentist"


# ---------------------------------------------------------------------------
# PatientAnalytics
# Table: patient_analytics
# FK: patient_id → users.id (one-to-one)
# Used as LLM prompt context source for all five analytics modules.
# ---------------------------------------------------------------------------

class PatientAnalytics(Base):
    __tablename__ = "patient_analytics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        unique=True,                        # one row per patient
    )

    # Lifestyle
    sugar_intake_score = Column(SmallInteger, nullable=False, default=5)
    brushing_frequency = Column(SmallInteger, nullable=False, default=2)
    flossing_frequency = Column(SmallInteger, nullable=False, default=0)
    smoking = Column(Boolean, nullable=False, default=False)
    alcohol_use = Column(Boolean, nullable=False, default=False)

    # Clinical history
    previous_cavities = Column(SmallInteger, nullable=False, default=0)
    previous_extractions = Column(SmallInteger, nullable=False, default=0)
    last_dental_visit_months_ago = Column(SmallInteger, nullable=True)
    medical_history_notes = Column(Text, nullable=True)

    # Genetics / family history
    family_history_dental_disease = Column(Boolean, nullable=False, default=False)

    # Systemic conditions
    diabetes = Column(Boolean, nullable=False, default=False)

    # No-show factors
    travel_distance_km = Column(Numeric(6, 2), nullable=True)
    historical_missed_appointments = Column(SmallInteger, nullable=False, default=0)
    total_past_appointments = Column(SmallInteger, nullable=False, default=0)

    # Treatment history
    previous_procedure_outcomes = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    patient = relationship("User", back_populates="analytics")


# ---------------------------------------------------------------------------
# OralHealthRiskScore
# Table: oral_health_risk_scores
# FK: patient_id → users.id (one-to-many — one patient, many scored runs)
# Persists each OralHealthRiskResponse LLM result.
# ---------------------------------------------------------------------------

class OralHealthRiskScore(Base):
    __tablename__ = "oral_health_risk_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )

    risk_score = Column(SmallInteger, nullable=False)           # 0–100
    health_grade = Column(String(30), nullable=False)           # e.g. "High Risk"
    risk_level = Column(Enum(RiskLevelEnum, name="risk_level_enum"), nullable=False)
    disease_progression_forecast = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=False)
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    patient = relationship("User", back_populates="oral_health_risk_scores")
    prompt_log = relationship(
        "AnalyticsPromptLog",
        primaryjoin="and_(AnalyticsPromptLog.module=='oral_health_risk', "
                    "foreign(AnalyticsPromptLog.reference_id)==OralHealthRiskScore.id)",
        viewonly=True,
    )


# ---------------------------------------------------------------------------
# TreatmentOutcomePrediction
# Table: treatment_outcome_predictions
# FK: patient_id → users.id (one-to-many)
# Persists each TreatmentOutcomeResponse LLM result.
# ---------------------------------------------------------------------------

class TreatmentOutcomePrediction(Base):
    __tablename__ = "treatment_outcome_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )

    procedure_name = Column(String(150), nullable=False)
    success_probability = Column(Numeric(5, 2), nullable=False)  # 0.00–100.00
    confidence_level = Column(Enum(RiskLevelEnum, name="risk_level_enum"), nullable=False)
    key_factors = Column(Text, nullable=False)
    recommendation = Column(Text, nullable=False)
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    patient = relationship("User", back_populates="treatment_outcome_predictions")
    prompt_log = relationship(
        "AnalyticsPromptLog",
        primaryjoin="and_(AnalyticsPromptLog.module=='treatment_outcome', "
                    "foreign(AnalyticsPromptLog.reference_id)==TreatmentOutcomePrediction.id)",
        viewonly=True,
    )


# ---------------------------------------------------------------------------
# DiseaseProgressionForecast
# Table: disease_progression_forecasts
# FK: patient_id → users.id (one-to-many)
# Persists each DiseaseProgressionResponse LLM result.
# ---------------------------------------------------------------------------

class DiseaseProgressionForecast(Base):
    __tablename__ = "disease_progression_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )

    current_condition = Column(String(200), nullable=False)
    severity_at_start = Column(Enum(RiskLevelEnum, name="risk_level_enum"), nullable=False)
    projected_severity = Column(Enum(RiskLevelEnum, name="risk_level_enum"), nullable=False)
    months_untreated = Column(SmallInteger, nullable=False)
    progression_forecast = Column(Text, nullable=False)
    recommended_intervention = Column(Text, nullable=False)
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    patient = relationship("User", back_populates="disease_progression_forecasts")
    prompt_log = relationship(
        "AnalyticsPromptLog",
        primaryjoin="and_(AnalyticsPromptLog.module=='disease_progression', "
                    "foreign(AnalyticsPromptLog.reference_id)==DiseaseProgressionForecast.id)",
        viewonly=True,
    )


# ---------------------------------------------------------------------------
# RiskStratificationReport
# Table: risk_stratification_reports
# No patient FK — clinic-level aggregate snapshot.
# branch is a soft match to users.branch (not enforced by FK).
# ---------------------------------------------------------------------------

class RiskStratificationReport(Base):
    __tablename__ = "risk_stratification_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    branch = Column(String(100), nullable=True)                 # None = all branches
    timeframe_days = Column(SmallInteger, nullable=False, default=30)

    total_patients_analyzed = Column(Integer, nullable=False)
    low_risk_count = Column(Integer, nullable=False)
    medium_risk_count = Column(Integer, nullable=False)
    high_risk_count = Column(Integer, nullable=False)

    low_risk_pct = Column(Numeric(5, 2), nullable=False)
    medium_risk_pct = Column(Numeric(5, 2), nullable=False)
    high_risk_pct = Column(Numeric(5, 2), nullable=False)

    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    patient_rows = relationship(
        "RiskStratificationPatientRow",
        back_populates="report",
        cascade="all, delete-orphan",
    )
    prompt_log = relationship(
        "AnalyticsPromptLog",
        primaryjoin="and_(AnalyticsPromptLog.module=='risk_stratification', "
                    "foreign(AnalyticsPromptLog.reference_id)==RiskStratificationReport.id)",
        viewonly=True,
    )


# ---------------------------------------------------------------------------
# RiskStratificationPatientRow
# Table: risk_stratification_patient_rows
# FK: report_id → risk_stratification_reports.id (many-to-one)
# FK: patient_id → users.id (many-to-one)
# Bridge between the aggregate report and individual patients.
# ---------------------------------------------------------------------------

class RiskStratificationPatientRow(Base):
    __tablename__ = "risk_stratification_patient_rows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(
        Integer,
        ForeignKey("risk_stratification_reports.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )
    patient_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )

    risk_score = Column(SmallInteger, nullable=False)
    risk_level = Column(Enum(RiskLevelEnum, name="risk_level_enum"), nullable=False)
    last_visit_date = Column(Date, nullable=True)

    # Relationships
    report = relationship("RiskStratificationReport", back_populates="patient_rows")
    patient = relationship("User", back_populates="stratification_rows")


# ---------------------------------------------------------------------------
# NoShowPrediction
# Table: no_show_predictions
# FK: appointment_id → appointments.booking_ref  (VARCHAR 20 natural key)
# FK: patient_id → users.id
# ---------------------------------------------------------------------------

class NoShowPrediction(Base):
    __tablename__ = "no_show_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # CORRECTED: FK → appointments.booking_ref (varchar 20), not a synthetic UUID
    appointment_id = Column(
        Integer,
        ForeignKey("appointments.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        unique=True,                        # one prediction per appointment
    )
    patient_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )
    branch = Column(String(100), nullable=True)

    # Appointment context
    appointment_day_of_week = Column(Enum(AppointmentDayEnum, name="appointment_day_enum"), nullable=False)
    appointment_time_of_day = Column(String(20), nullable=True)
    days_until_appointment = Column(SmallInteger, nullable=False)
    reminder_sent = Column(Boolean, nullable=False, default=False)
    weather_risk_flag = Column(Boolean, nullable=False, default=False)

    # Snapshot of patient_analytics at prediction time
    travel_distance_km = Column(Numeric(6, 2), nullable=True)
    historical_missed_appointments = Column(SmallInteger, nullable=False, default=0)
    total_past_appointments = Column(SmallInteger, nullable=False, default=0)

    # LLM result
    no_show_probability = Column(Numeric(5, 2), nullable=False)
    risk_flag = Column(Enum(RiskLevelEnum, name="risk_level_enum"), nullable=False)
    reasoning = Column(Text, nullable=False)
    automated_reminder_triggered = Column(Boolean, nullable=False, default=False)
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    appointment = relationship("Appointment", back_populates="no_show_prediction")
    patient = relationship("User", back_populates="no_show_predictions")
    prompt_log = relationship(
        "AnalyticsPromptLog",
        primaryjoin="and_(AnalyticsPromptLog.module=='no_show_prediction', "
                    "foreign(AnalyticsPromptLog.reference_id)==NoShowPrediction.id)",
        viewonly=True,
    )


# ---------------------------------------------------------------------------
# AnalyticsPromptLog
# Table: analytics_prompt_logs
# Polymorphic soft-reference log — no FK on reference_id by design.
# module enum tells you which result table reference_id points to.
# ---------------------------------------------------------------------------

class AnalyticsPromptLog(Base):
    __tablename__ = "analytics_prompt_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    module = Column(Enum(AnalyticsModuleEnum, name="analytics_module_enum"), nullable=False)
    reference_id = Column(Integer, nullable=False)              # soft FK — polymorphic
    prompt_sent = Column(Text, nullable=False)
    raw_llm_response = Column(Text, nullable=False)
    model_used = Column(String(50), nullable=False, default="claude-sonnet-4-20250514")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(20), nullable=False)
    last_name = Column(String(20), nullable=False)
    email = Column(String(50), nullable=False, unique=True)
    role = Column(Enum(UserRoleEnum, name="user_role_enum"), nullable=False, default=UserRoleEnum.patient)
    branch = Column(String(100), nullable=True, default="Main Branch")
    password = Column(String(255), nullable=False)
    created_at = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )
    sex = Column(String(10), nullable=True)
    dob = Column(String(20), nullable=True)
    age = Column(Integer, nullable=True)
    phone = Column(String(20), nullable=True)
    occupation = Column(String(100), nullable=True)
    blood_type = Column(String(5), nullable=True, default="O+")
    allergies = Column(Text, nullable=True)
    insurance = Column(String(100), nullable=True)
    policy_number = Column(String(100), nullable=True)
    specialty = Column(String(100), nullable=True, default="General Dentistry")
    status = Column(String(20), nullable=True, default="Available")
    profile_picture = Column(String(255), nullable=True)

    # ---------------------------------------------------------
    # Analytics Relationships
    # ---------------------------------------------------------

    # One-to-one: every patient has one analytics profile
    analytics = relationship(
        "PatientAnalytics",
        back_populates="patient",
        uselist=False, 
        cascade="all, delete-orphan",
    )

    # One-to-many: audit log of every risk score run for this patient
    oral_health_risk_scores = relationship(
        "OralHealthRiskScore",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    # One-to-many: every treatment prediction for this patient
    treatment_outcome_predictions = relationship(
        "TreatmentOutcomePrediction",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    # One-to-many: every disease forecast for this patient
    disease_progression_forecasts = relationship(
        "DiseaseProgressionForecast",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    # One-to-many: rows this patient appears in across stratification reports
    stratification_rows = relationship(
        "RiskStratificationPatientRow",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    # One-to-many: no-show predictions for this patient's appointments
    no_show_predictions = relationship(
        "NoShowPrediction",
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    

class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    booking_ref = Column(String(20), unique=True, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    service_type = Column(String(255), nullable=False)
    dentist_name = Column(String(255), nullable=False)
    appointment_date = Column(Date, nullable=False)
    appointment_time = Column(String(50), nullable=False)
    status = Column(String(50), default="Pending")
    created_at = Column(
        DateTime, 
        nullable=False, 
        server_default=func.now()
    )
    amount = Column(Numeric(10, 2), default=0.00)
    branch = Column(String(255), default="Main Branch")

    # Reference back to the User (Patient)
    patient = relationship("User", backref="appointments")

    # One-to-one: one no-show prediction per appointment (unique on appointment_id)
    no_show_prediction = relationship(
        "NoShowPrediction",
        back_populates="appointment",
        uselist=False,                      # one-to-one (appointment_id is unique)
        cascade="all, delete-orphan",
    )
    

class PatientRecord(Base):
    __tablename__ = "patient_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    upload_date = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )

    # Relationship back to the User model
    patient = relationship("User", back_populates="records")


class AIDiagnostic(Base):
    __tablename__ = "ai_diagnostics"

    # Mapping 'diagnosis_id' from MySQL to 'id' for ORM consistency
    id = Column("diagnosis_id", Integer, primary_key=True, autoincrement=True)
    patient_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
    )
    clinical_notes = Column(Text, nullable=True)
    
    # Using JSONB for optimized PostgreSQL performance
    ai_findings = Column(JSONB, nullable=True, server_default='{}')
    
    scan_date = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )

    # Relationship back to the User model
    patient = relationship("User", back_populates="diagnostics")