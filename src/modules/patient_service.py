from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.repository.patient_repository import PatientRepository
from src.agents.patient_agent import PatientCheckupAgent
from src.schemas.schema import OralHealthRiskRequest, OralHealthRiskResponse
from src.models.model import OralHealthRiskScore, AnalyticsPromptLog, AnalyticsModuleEnum, PatientAnalytics
import logging

logger = logging.getLogger(__name__)

class PatientService:
    def __init__(self, db: AsyncSession):
        self.repository = PatientRepository(db)

    async def get_patient_analytics(self, patient_id: int) -> OralHealthRiskRequest:
        # Use the new repository method that includes the User profile
        analytics = await self.repository.get_patient_analytics_with_profile(patient_id)
        
        if not analytics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analytics for patient {patient_id} not found."
            )
        
        # Access User fields via the 'patient' relationship
        user_profile = analytics.patient 

        return OralHealthRiskRequest(
            patient_id=analytics.patient_id, 
            age=user_profile.age,         
            sex=user_profile.sex,         
            blood_type=user_profile.blood_type,
            allergies=user_profile.allergies,
            occupation=user_profile.occupation,
            sugar_intake_score=analytics.sugar_intake_score,
            brushing_frequency=analytics.brushing_frequency, 
            flossing_frequency=analytics.flossing_frequency, 
            smoking=analytics.smoking, 
            alcohol_use=analytics.alcohol_use, 
            previous_cavities=analytics.previous_cavities, 
            previous_extractions=analytics.previous_extractions, 
            family_history_dental_disease=analytics.family_history_dental_disease, 
            last_dental_visit_months_ago=analytics.last_dental_visit_months_ago, 
            medical_history_notes=analytics.medical_history_notes,
        )
        
        
    async def get_patient_oral_health_risk_scores(self, patient_id: int) -> OralHealthRiskResponse:
        risk_data = await self.repository.get_patient_oral_health_risk(patient_id)

        if not risk_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No risk assessment found for patient ID {patient_id}."
            )

        return OralHealthRiskResponse(
            patient_id=risk_data.patient_id,
            risk_score=risk_data.risk_score,
            health_grade=risk_data.health_grade,
            risk_level=risk_data.risk_level,
            disease_progression_forecast=risk_data.disease_progression_forecast,
            recommended_action=risk_data.recommended_action,
            generated_at=risk_data.generated_at
        )         
        
