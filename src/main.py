from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from src.database.session import engine

from sqlalchemy.ext.asyncio import AsyncSession
from src.models.model import *

@asynccontextmanager
async def life_span(app: FastAPI):
    try:
        # Create database tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            print("Tables created successfully!")
        
        yield
        
    finally:
        
        await engine.dispose()
        print("Application shutdown complete")
       

app = FastAPI(
    title="Capstone project module b",
    lifespan=life_span
)

from src.routers.patient_routes import router as patient_router
app.include_router(patient_router)
