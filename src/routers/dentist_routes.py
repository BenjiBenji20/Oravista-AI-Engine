from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.session import get_async_db
from src.modules.dentist_service import DentistService
from src.schemas.schema import DentistDashboardResponse

router = APIRouter(prefix="/api/dentist", tags=["Dentist Dashboard"])

@router.get("/dashboard/predict-risk-queue/{dentist_id}", response_model=DentistDashboardResponse)
async def get_dashboard(
    dentist_id: int, 
    branch: str = "Main Branch", 
    db: AsyncSession = Depends(get_async_db)
):
    service = DentistService(db)
    return await service.get_dentist_dashboard(dentist_id, branch)
