from fastapi import HTTPException

from src.agents.treatment_prediction_agent import TreatmentPredictionAgent
from src.models.model import AnalyticsModuleEnum, AnalyticsPromptLog, TreatmentOutcomePrediction
from src.repository.dentist_repository import DentistRepository
from src.schemas.schema import DentistDashboardPatient, DentistDashboardResponse, OralHealthRiskRequest,  TreatmentOutcomeDetailResponse, TreatmentOutcomeResponse
import logging

logger = logging.getLogger(__name__)
class DentistService:
    def __init__(self, db):
        self.repository = DentistRepository(db)
        self.agent = TreatmentPredictionAgent()

    async def get_dentist_dashboard(self, dentist_id: int, branch: str) -> DentistDashboardResponse:
        # Fetch raw data from repo
        records = await self.repository.get_dashboard_patients(branch)
        
        patient_list = []
        for user, risk in records:
            patient_list.append(
                DentistDashboardPatient(
                    patient_id=user.id,
                    name=f"{user.first_name} {user.last_name}",
                    score=risk.risk_score,
                    issue=risk.health_grade or "General Checkup",
                    progression=risk.disease_progression_forecast or "No forecast available.",
                    action=risk.recommended_action or "Routine cleaning."
                )
            )
            
        return DentistDashboardResponse(
            dentist_id=dentist_id,
            patients=patient_list
        )


    async def predict_treatment_outcomes(self, request: OralHealthRiskRequest) -> TreatmentOutcomeResponse:
        try:
            if not request:
                raise HTTPException(status_code=400, detail="No health risk data provided in request.")

            # 1. Cache Check: Skip LLM call if a prediction record already exists for this patient
            existing_prediction = await self.repository.get_latest_treatment_prediction_by_patient(request.patient_id)
            if existing_prediction:
                return TreatmentOutcomeResponse(
                    id=existing_prediction.id,
                    patient_id=existing_prediction.patient_id,
                    procedure_name=existing_prediction.procedure_name,
                    success_probability=float(existing_prediction.success_probability),
                    key_factors=existing_prediction.key_factors,
                    generated_at=existing_prediction.generated_at
                )

            # 2. Convert current profile into clean payload context for LLM execution
            context = request.model_dump()

            # 3. Call the agent to diagnose the most critical procedure and compute regression probabilities
            outcome, prompt, raw_response = await self.agent.predict_outcome(context)

            # 4. Map the response confidence string to match DB Enum constraints ("Low", "Medium", "High")
            confidence_enum = outcome.confidence_level.capitalize()
            
            db_prediction = TreatmentOutcomePrediction(
                patient_id=request.patient_id,
                procedure_name=outcome.procedure_name,
                success_probability=outcome.success_probability,
                confidence_level=confidence_enum, 
                key_factors=outcome.key_factors,
                recommendation=outcome.recommendation
            )
            
            saved_prediction = await self.repository.save_treatment_prediction(db_prediction)
            
            # 5. Log the transactional history for compliance audit trails
            prompt_log = AnalyticsPromptLog(
                module=AnalyticsModuleEnum.treatment_outcome,
                reference_id=saved_prediction.id,
                prompt_sent=prompt,
                raw_llm_response=raw_response,
                model_used=self.agent.model_name
            )
            await self.repository.save_prompt_log(prompt_log)
            
            # 6. Construct and return a lean object back to the client router
            return TreatmentOutcomeResponse(
                id=saved_prediction.id,
                patient_id=saved_prediction.patient_id,
                procedure_name=saved_prediction.procedure_name,
                success_probability=float(saved_prediction.success_probability),
                key_factors=saved_prediction.key_factors,
                generated_at=saved_prediction.generated_at
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error predicting treatment outcome: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to predict treatment outcome: {str(e)}")
            
    async def get_prediction_details(self, patient_id: int, prediction_id: int) -> TreatmentOutcomeDetailResponse:
        prediction = await self.repository.get_treatment_prediction(prediction_id)
        if not prediction or prediction.patient_id != patient_id:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
        return TreatmentOutcomeDetailResponse(
            id=prediction.id,
            patient_id=prediction.patient_id,
            procedure_name=prediction.procedure_name,
            success_probability=float(prediction.success_probability),
            key_factors=prediction.key_factors,
            confidence_level=prediction.confidence_level.value, 
            recommendation=prediction.recommendation,
            generated_at=prediction.generated_at
        )
        