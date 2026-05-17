import logging
from fastapi import HTTPException
from typing import List, Optional
from datetime import datetime, timezone
from src.agents.no_show_prediction_agent import NoShowPredictionAgent
from src.repository.dentist_repository import DentistRepository
from src.schemas.schema import NoShowDashboardResponse
from src.models.model import NoShowPrediction, AnalyticsPromptLog, AnalyticsModuleEnum

logger = logging.getLogger(__name__)

class NoShowPredictionService:
    def __init__(self, db):
        self.db = db
        self.repository = DentistRepository(db)
        self.agent = NoShowPredictionAgent()

    async def process_no_show_prediction(self, appointment_id: int, weather_risk: bool = False) -> NoShowDashboardResponse:
        try:
            # 1. Cache Check: Check if an existing assessment is already available for this appointment
            existing_prediction = await self.repository.get_no_show_prediction_by_appointment(appointment_id)
            if existing_prediction:
                return self._map_to_dashboard_response(existing_prediction)

            # 2. Extract context data from Appointment, User, and PatientAnalytics tables
            data = await self.repository.get_appointment_context_for_prediction(appointment_id)
            if not data:
                raise HTTPException(status_code=404, detail="Appointment data context could not be collected.")

            appointment, user, analytics = data

            # 3. Assemble the structural payload context for Gemini
            lead_days = (appointment.appointment_date - datetime.now(timezone.utc).date()).days
            lead_days = max(0, lead_days)
            
            day_of_week = appointment.appointment_date.strftime("%A")
            
            context_payload = {
                "appointment_day_of_week": day_of_week,
                "appointment_time_of_day": appointment.appointment_time,
                "days_until_appointment": lead_days,
                "travel_distance_km": float(analytics.travel_distance_km) if analytics.travel_distance_km else 0.0,
                "historical_missed_appointments": analytics.historical_missed_appointments,
                "total_past_appointments": analytics.total_past_appointments,
                "age": user.age,
                "occupation": user.occupation,
                "weather_risk_flag": weather_risk
            }

            # 4. Generate prediction via Gemini Agent
            assessment, prompt, raw_response = await self.agent.predict_no_show(context_payload)

            # 5. Determine if an automated reminder is triggered (e.g., probability threshold > 70%)
            auto_reminder = True if assessment.no_show_probability >= 70.0 else False

            # 6. Save data to the no_show_predictions table
            db_prediction = NoShowPrediction(
                appointment_id=appointment.id,
                patient_id=user.id,
                branch=appointment.branch,
                appointment_day_of_week=day_of_week,
                appointment_time_of_day=appointment.appointment_time,
                days_until_appointment=lead_days,
                reminder_sent=appointment.status == "Reminder Sent",
                weather_risk_flag=weather_risk,
                travel_distance_km=analytics.travel_distance_km,
                historical_missed_appointments=analytics.historical_missed_appointments,
                total_past_appointments=analytics.total_past_appointments,
                no_show_probability=assessment.no_show_probability,
                risk_flag=assessment.risk_flag.capitalize(), # "High", "Medium", "Low"
                reasoning=assessment.reasoning,
                automated_reminder_triggered=auto_reminder
            )
            saved_prediction = await self.repository.save_no_show_prediction(db_prediction)

            # 7. Log transactional audit metrics
            prompt_log = AnalyticsPromptLog(
                module=AnalyticsModuleEnum.no_show_prediction,
                reference_id=saved_prediction.id,
                prompt_sent=prompt,
                raw_llm_response=raw_response,
                model_used=self.agent.model_name
            )
            await self.repository.save_prompt_log(prompt_log)

            return self._map_to_dashboard_response(saved_prediction, user_fullname=f"{user.first_name} {user.last_name}", appt_time=f"{appointment.appointment_date} {appointment.appointment_time}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error predicting appointment cancellation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate no-show prediction: {str(e)}")

    def _map_to_dashboard_response(self, pred: NoShowPrediction, user_fullname: str = None, appt_time: str = None) -> NoShowDashboardResponse:
        """Helper to safely format structural data into the target UI schema format."""
        # Fallbacks if loading straight from an existing cached relation object
        display_name = user_fullname or f"{pred.patient.first_name} {pred.patient.last_name}"
        display_time = appt_time or f"{pred.appointment.appointment_date} {pred.appointment.appointment_time}"
        
        status_string = "Reminder Sent" if (pred.automated_reminder_triggered or pred.reminder_sent) else "Pending"

        return NoShowDashboardResponse(
            appointment_id=pred.appointment_id,
            time=display_time,
            patient=display_name,
            probability=float(pred.no_show_probability),
            reason=pred.reasoning,
            status=status_string
        )


    async def get_branch_no_show_queue(self, branch: str) -> List[NoShowDashboardResponse]:
        """
        Retrieves the upcoming schedule list formatted directly for the UI scaffolding,
        exposing the structural appointment_id for the frontend to target.
        """
        appointments = await self.repository.get_upcoming_no_show_queue(branch)
        
        queue_response = []
        for appt in appointments:
            patient_name = f"{appt.patient.first_name} {appt.patient.last_name}"
            display_time = f"{appt.appointment_date} {appt.appointment_time}"
            
            # If an AI prediction has already been generated and saved for this appointment
            if appt.no_show_prediction:
                pred = appt.no_show_prediction
                status_string = "Reminder Sent" if (pred.automated_reminder_triggered or pred.reminder_sent) else "Processed"
                probability = float(pred.no_show_probability)
                reason = pred.reasoning
            else:
                # Prediction hasn't been run yet—provide baseline structural data to the UI
                status_string = "No Prediction Run"
                probability = 0.0
                reason = "Click 'Predict' to evaluate risk metrics"

            queue_response.append(
                NoShowDashboardResponse(
                    appointment_id=appt.id, # Front-end captures this ID to send to POST trigger
                    time=display_time,
                    patient=patient_name,
                    probability=probability,
                    reason=reason,
                    status=status_string
                )
            )
            
        return queue_response
        