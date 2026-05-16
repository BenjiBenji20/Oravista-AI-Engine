from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from src.models.model import PatientAnalytics, OralHealthRiskScore, AnalyticsPromptLog
from src.schemas.schema import OralHealthRiskRequest

class PatientRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_patient_analytics(self, patient_id: int) -> PatientAnalytics | None:
        result = await self.db.execute(select(PatientAnalytics).where(PatientAnalytics.patient_id == patient_id))
        return result.scalar_one_or_none()

    async def upsert_patient_analytics(self, request: OralHealthRiskRequest) -> PatientAnalytics:
        analytics = await self.get_patient_analytics(request.patient_id)
        if not analytics:
            analytics = PatientAnalytics(patient_id=request.patient_id)
            self.db.add(analytics)
        
        # Update fields from request
        analytics.sugar_intake_score = request.sugar_intake_score
        analytics.brushing_frequency = request.brushing_frequency
        analytics.flossing_frequency = request.flossing_frequency
        analytics.smoking = request.smoking
        analytics.alcohol_use = request.alcohol_use
        analytics.previous_cavities = request.previous_cavities
        analytics.previous_extractions = request.previous_extractions
        analytics.family_history_dental_disease = request.family_history_dental_disease
        analytics.last_dental_visit_months_ago = request.last_dental_visit_months_ago
        analytics.medical_history_notes = request.medical_history_notes

        await self.db.commit()
        await self.db.refresh(analytics)
        return analytics

    async def save_oral_health_risk_score(self, score: OralHealthRiskScore) -> OralHealthRiskScore:
        self.db.add(score)
        await self.db.commit()
        await self.db.refresh(score)
        return score

    async def save_prompt_log(self, log: AnalyticsPromptLog) -> AnalyticsPromptLog:
        self.db.add(log)
        await self.db.commit()
        await self.db.refresh(log)
        return log
