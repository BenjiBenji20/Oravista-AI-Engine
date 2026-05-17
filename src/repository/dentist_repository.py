from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from src.schemas.schema import OralHealthRiskRequest
from src.models.model import AnalyticsPromptLog, PatientAnalytics, TreatmentOutcomePrediction, User, OralHealthRiskScore
from sqlalchemy.orm import selectinload

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