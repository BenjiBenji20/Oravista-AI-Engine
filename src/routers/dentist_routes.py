from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.session import get_async_db
from src.modules.dentist_service import DentistService
from src.schemas.schema import DentistDashboardResponse, OralHealthRiskRequest, TreatmentOutcomeDetailResponse, TreatmentOutcomeResponse

router = APIRouter(prefix="/api/dentist", tags=["Dentist Dashboard"])

@router.get("/dashboard/predict-risk-queue/{dentist_id}", response_model=DentistDashboardResponse)
async def get_dashboard(
    dentist_id: int, 
    branch: str = "Main Branch", 
    db: AsyncSession = Depends(get_async_db)
):
    service = DentistService(db)
    return await service.get_dentist_dashboard(dentist_id, branch)


@router.post("/dashboard/predict-treatment-outcome", response_model=TreatmentOutcomeResponse)
async def trigger_treatment_predictions(
    request: OralHealthRiskRequest, 
    db: AsyncSession = Depends(get_async_db)
):
    service = DentistService(db)
    return await service.predict_treatment_outcomes(request)


@router.get("/dashboard/{patient_id}/predict-treatment-outcome/{prediction_id}", response_model=TreatmentOutcomeDetailResponse)
async def get_prediction_details(
    patient_id: int,
    prediction_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    service = DentistService(db)
    return await service.get_prediction_details(patient_id, prediction_id)
