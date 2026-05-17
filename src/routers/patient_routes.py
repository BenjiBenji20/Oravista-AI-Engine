from typing import Tuple

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.session import get_async_db
from src.modules.patient_service import PatientService
from src.schemas.schema import *

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

@router.get("/get/{patient_id}/analytics", response_model=OralHealthRiskRequest)
async def get_patient_analytics(
    patient_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    service = PatientService(db)
    return await service.get_patient_analytics(patient_id)

@router.get("/get/{patient_id}/oral-health-risk", response_model=OralHealthRiskResponse)
async def get_patient_risk( 
    patient_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    service = PatientService(db)
    return await service.get_patient_oral_health_risk_scores(patient_id)

