from decimal import Decimal

from fastapi import HTTPException

from src.agents.treatment_prediction_agent import TreatmentPredictionAgent
from src.agents.patient_agent import PatientCheckupAgent
from src.models.model import AnalyticsModuleEnum, AnalyticsPromptLog, RiskStratificationPatientRow, RiskStratificationReport, TreatmentOutcomePrediction
from src.repository.dentist_repository import DentistRepository
from src.schemas.schema import *
import logging

logger = logging.getLogger(__name__)
class DentistService:
    def __init__(self, db):
        self.repository = DentistRepository(db)
        self.agent = TreatmentPredictionAgent()
        self.patient_agent = PatientCheckupAgent()  


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


    async def generate_risk_stratification(self, request: RiskStratificationRequest) -> RiskStratificationResponse:
        try:
            # 1. Fetch latest raw risk profile entries for the branch
            raw_profiles = await self.repository.get_latest_patient_risk_profiles(request.branch)
            total_patients = len(raw_profiles)

            if total_patients == 0:
                raise HTTPException(status_code=404, detail=f"No patients with completed risk scores found in branch: {request.branch}")

            # 2. Segregate profiles into risk tiers based on their DB Enum string value
            low_tier = [r for r in raw_profiles if r.OralHealthRiskScore.risk_level == "Low"]
            med_tier = [r for r in raw_profiles if r.OralHealthRiskScore.risk_level == "Medium"]
            high_tier = [r for r in raw_profiles if r.OralHealthRiskScore.risk_level == "High"]

            # 3. Calculate mathematical percentages (Rounded cleanly to 2 decimal points)
            low_pct = round((len(low_tier) / total_patients) * 100, 2)
            med_pct = round((len(med_tier) / total_patients) * 100, 2)
            
            # Defensive math: force high tier adjustment to guarantee the sum equals exactly 100.00%
            high_pct = round(100.00 - (low_pct + med_pct), 2)

            # 4. Save aggregate snapshot report data to DB
            db_report = RiskStratificationReport(
                branch=request.branch,
                timeframe_days=request.timeframe_days,
                total_patients_analyzed=total_patients,
                low_risk_count=len(low_tier),
                medium_risk_count=len(med_tier),
                high_risk_count=len(high_tier),
                low_risk_pct=Decimal(str(low_pct)),
                medium_risk_pct=Decimal(str(med_pct)),
                high_risk_pct=Decimal(str(high_pct))
            )
            saved_report = await self.repository.save_stratification_report(db_report)

            # 5. Build individual drill-down lookup mappings for every patient analyzed
            for profile in raw_profiles:
                patient_row = RiskStratificationPatientRow(
                    report_id=saved_report.id,
                    patient_id=profile.User.id,
                    risk_score=profile.OralHealthRiskScore.risk_score,
                    risk_level=profile.OralHealthRiskScore.risk_level,
                    last_visit_date=None # Can hook to real last appointment date if available
                )
                self.repository.db.add(patient_row)
            
            # Commit the granular batch mapping rows
            await self.repository.db.commit()

            return RiskStratificationResponse(
                id=saved_report.id,
                branch=saved_report.branch,
                timeframe_days=saved_report.timeframe_days,
                total_patients_analyzed=saved_report.total_patients_analyzed,
                low_risk_count=saved_report.low_risk_count,
                medium_risk_count=saved_report.medium_risk_count,
                high_risk_count=saved_report.high_risk_count,
                low_risk_pct=float(saved_report.low_risk_pct),
                medium_risk_pct=float(saved_report.medium_risk_pct),
                high_risk_pct=float(saved_report.high_risk_pct),
                generated_at=saved_report.generated_at
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating risk stratification: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate risk stratification report: {str(e)}")


    async def get_patients_in_stratification_tier(self, report_id: int, risk_level: str) -> List[PatientRiskSummary]:
        try:
            # Fetch mapping rows from repository using our optimized relationship loader
            rows = await self.repository.get_patients_by_tier(report_id, risk_level)
            
            # Map structural components into clean API summaries
            return [
                PatientRiskSummary(
                    patient_id=row.patient_id,
                    full_name=f"{row.patient.first_name} {row.patient.last_name}",
                    risk_score=row.risk_score,
                    risk_level=row.risk_level.value if hasattr(row.risk_level, 'value') else str(row.risk_level),
                    last_visit_date=row.last_visit_date
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error extracting target tier patients: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch drill-down patient profiles: {str(e)}")
        

    async def process_checkup(self, request: OralHealthRiskRequest) -> OralHealthRiskResponse:
        try:
            # 1. Update/Save the patient analytics with the new checkup data
            await self.repository.upsert_patient_analytics(request)

            # 2. Call the patient_agent with the request data
            # Convert request to dict for LLM context
            context = request.model_dump()
            assessment, prompt, raw_response = await self.patient_agent.analyze_patient_risk(context)

            # 3. Save the risk score
            score_model = OralHealthRiskScore(
                patient_id=request.patient_id,
                risk_score=assessment.risk_score,
                health_grade=assessment.health_grade,
                risk_level=assessment.risk_level.value, # Need to use .value for the string
                disease_progression_forecast=assessment.disease_progression_forecast,
                recommended_action=assessment.recommended_action
            )
            saved_score = await self.repository.save_oral_health_risk_score(score_model)

            # 4. Save the prompt log
            prompt_log = AnalyticsPromptLog(
                module=AnalyticsModuleEnum.oral_health_risk,
                reference_id=saved_score.id,
                prompt_sent=prompt,
                raw_llm_response=raw_response,
                model_used=self.patient_agent.model_name
            )
            await self.repository.save_prompt_log(prompt_log)

            # 5. Build and return the response
            response = OralHealthRiskResponse(
                patient_id=request.patient_id,
                risk_score=saved_score.risk_score,
                health_grade=saved_score.health_grade,
                risk_level=saved_score.risk_level,
                disease_progression_forecast=saved_score.disease_progression_forecast,
                recommended_action=saved_score.recommended_action,
                generated_at=saved_score.generated_at
            )
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in process_checkup: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process checkup: {str(e)}")
      