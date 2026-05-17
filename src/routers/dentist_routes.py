from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.session import get_async_db
from src.modules.dentist_service import DentistService
from src.schemas.schema import DentistDashboardResponse, OralHealthRiskRequest, PatientRiskSummary, RiskStratificationRequest, RiskStratificationResponse, TreatmentOutcomeDetailResponse, TreatmentOutcomeResponse

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


@router.post("/dashboard/risk-stratification", response_model=RiskStratificationResponse)
async def trigger_risk_stratification(
    request: RiskStratificationRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """Calculates and snapshots the branch-level patient distribution categories."""
    print(f"\n\n BRANCH: {request.branch}\n\n")
    service = DentistService(db)
    return await service.generate_risk_stratification(request)


@router.get("/dashboard/risk-stratification/{report_id}/patients", response_model=List[PatientRiskSummary])
async def get_patients_by_risk_tier(
    report_id: int,
    risk_level: str = Query(..., description="Must be 'Low', 'Medium', or 'High'"),
    db: AsyncSession = Depends(get_async_db)
):
    """Drill-down endpoint: Fetches patients matching a specific category from a snapshot report."""
    if risk_level not in ["Low", "Medium", "High"]:
        raise HTTPException(status_code=400, detail="Invalid risk_level. Use 'Low', 'Medium', or 'High'.")
        
    service = DentistService(db)
    return await service.get_patients_in_stratification_tier(report_id, risk_level)