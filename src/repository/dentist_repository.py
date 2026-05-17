from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from src.models.model import User, OralHealthRiskScore

class DentistRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_dashboard_patients(self, dentist_branch: str):
        """
        Fetches the latest risk record for every patient in the dentist's branch.
        RBAC: Filtered by branch at the database level.
        """
        # Subquery to get the latest score ID per patient
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