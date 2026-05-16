from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.session import get_async_db
from src.modules.patient_service import PatientService
from src.schemas.schema import OralHealthRiskRequest, OralHealthRiskResponse

router = APIRouter(prefix="/api/patient", tags=["patient"])

@router.post("/check-up", response_model=OralHealthRiskResponse)
async def health_test(
    request: OralHealthRiskRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """
    patient -> answer diagnostic tests -> llm -> response
    """
    service = PatientService(db)
    return await service.process_checkup(request)
