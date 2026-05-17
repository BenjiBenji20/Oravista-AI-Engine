from src.repository.dentist_repository import DentistRepository
from src.schemas.schema import DentistDashboardPatient, DentistDashboardResponse

class DentistService:
    def __init__(self, db):
        self.repository = DentistRepository(db)

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