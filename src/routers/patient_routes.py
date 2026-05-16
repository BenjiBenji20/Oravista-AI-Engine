from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.session import get_async_db

router = APIRouter("/api/patient")

@router.post("/health-test")
async def health_test(
    db: AsyncSession = Depends(get_async_db)
):
    """
    patient -> answer diagnostic tests -> llm -> response
    """
    



