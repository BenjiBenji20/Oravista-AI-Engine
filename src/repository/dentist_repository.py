from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from sqlalchemy.orm import joinedload, selectinload
from src.models.model import AnalyticsPromptLog, Appointment, NoShowPrediction, PatientAnalytics, RiskStratificationPatientRow, RiskStratificationReport, TreatmentOutcomePrediction, User, OralHealthRiskScore

class DentistRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_dashboard_patients(self, dentist_branch: str):
        """Fetches the latest risk record for every patient in the dentist's branch."""
        subquery = (
            select(func.max(OralHealthRiskScore.id))
            .group_by(OralHealthRiskScore.patient_id)
            .scalar_subquery()
        )

        query = (
            select(User, OralHealthRiskScore)
            .join(OralHealthRiskScore, User.id == OralHealthRiskScore.patient_id)
            .where(
                User.role == "patient",
                OralHealthRiskScore.id.in_(subquery)
            )
        )
        
        result = await self.db.execute(query)
        return result.all()


    async def get_latest_treatment_prediction_by_patient(self, patient_id: int) -> TreatmentOutcomePrediction | None:
        """
        Looks up the single most recent treatment outcome prediction for a patient.
        Used as a defensive cache to avoid duplicate LLM processing calls.
        """
        stmt = (
            select(TreatmentOutcomePrediction)
            .where(TreatmentOutcomePrediction.patient_id == patient_id)
            .order_by(TreatmentOutcomePrediction.id.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()


    async def save_treatment_prediction(self, prediction: TreatmentOutcomePrediction) -> TreatmentOutcomePrediction:
        self.db.add(prediction)
        await self.db.commit()
        await self.db.refresh(prediction)
        return prediction


    async def get_treatment_prediction(self, prediction_id: int) -> TreatmentOutcomePrediction:
        """Finds a specific prediction by its primary key ID (For detail views)."""
        stmt = select(TreatmentOutcomePrediction).where(TreatmentOutcomePrediction.id == prediction_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()


    async def get_all_treatment_predictions(self) -> list[TreatmentOutcomePrediction]:
        stmt = select(TreatmentOutcomePrediction).order_by(TreatmentOutcomePrediction.id.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())


    async def save_prompt_log(self, log: AnalyticsPromptLog):
        self.db.add(log)
        await self.db.commit()


    async def get_latest_patient_risk_profiles(self, branch: str) -> list:
        """Fetches the absolute latest risk score records for unique patients filtered by branch."""
        # Isolate the newest single score ID per unique patient
        latest_score_subquery = (
            select(func.max(OralHealthRiskScore.id))
            .group_by(OralHealthRiskScore.patient_id)
            .scalar_subquery()
        )

        # Gather users matching branch criteria carrying those explicit score IDs
        stmt = (
            select(User, OralHealthRiskScore)
            .join(OralHealthRiskScore, User.id == OralHealthRiskScore.patient_id)
            .where(
                User.role == "patient",
                User.branch == branch,
                OralHealthRiskScore.id.in_(latest_score_subquery)
            )
        )
        result = await self.db.execute(stmt)
        return result.all()


    async def save_stratification_report(self, report: RiskStratificationReport) -> RiskStratificationReport:
        self.db.add(report)
        await self.db.commit()
        await self.db.refresh(report)
        return report


    async def get_patients_by_tier(self, report_id: int, risk_level: str) -> list[RiskStratificationPatientRow]:
            """Fetches the structural list of patients mapped inside a specific snapshot tier with joined user references."""
            stmt = (
                select(RiskStratificationPatientRow)
                .options(selectinload(RiskStratificationPatientRow.patient)) # Explicitly load user data eagerly
                .where(
                    RiskStratificationPatientRow.report_id == report_id,
                    RiskStratificationPatientRow.risk_level == risk_level
                )
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        

    async def get_no_show_prediction_by_appointment(self, appointment_id: int) -> Optional[NoShowPrediction]:
        """Fetches an existing no-show assessment using relationship loaders to fetch metadata."""
        stmt = (
            select(NoShowPrediction)
            .options(joinedload(NoShowPrediction.patient), joinedload(NoShowPrediction.appointment))
            .where(NoShowPrediction.appointment_id == appointment_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()


    async def get_appointment_context_for_prediction(self, appointment_id: int):
        """
        Collects comprehensive operational elements across Appointment, User, 
        and PatientAnalytics tables in a single query execution block.
        """
        stmt = (
            select(Appointment, User, PatientAnalytics)
            .join(User, Appointment.user_id == User.id)
            .join(PatientAnalytics, User.id == PatientAnalytics.patient_id)
            .where(Appointment.id == appointment_id)
        )
        result = await self.db.execute(stmt)
        return result.one_or_none()


    async def save_no_show_prediction(self, prediction: NoShowPrediction) -> NoShowPrediction:
        self.db.add(prediction)
        await self.db.commit()
        await self.db.refresh(prediction)
        return prediction


    async def get_upcoming_no_show_queue(self, branch: str) -> list:
        """
        Fetches all upcoming appointments for a specific branch, joining patient info 
        and analytics context, along with any existing predictions.
        """
        current_date = datetime.now(timezone.utc).date()

        stmt = (
            select(Appointment)
            .options(
                joinedload(Appointment.patient).joinedload(User.analytics),
                joinedload(Appointment.no_show_prediction)
            )
            # 1. Join the user (required)
            .join(User, Appointment.user_id == User.id)
            # 2. Use OUTER JOIN for analytics so rows don't disappear if analytics is missing
            .outerjoin(PatientAnalytics, User.id == PatientAnalytics.patient_id)
            .where(
                # Make sure user_id 80 or 1 is actually set to UserRoleEnum.patient
                User.role == "patient", 
                Appointment.branch == branch
            )
            .order_by(Appointment.appointment_date.desc())
        )
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
        