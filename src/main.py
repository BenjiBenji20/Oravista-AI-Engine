from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from src.database.session import engine
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.model import *
from sqlalchemy import text

@asynccontextmanager
async def life_span(app: FastAPI):
    try:
        # Create database tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("SELECT 1"))
            print("\n\nPostgres connected successfully!")
        
        yield
        
    finally:
        
        await engine.dispose()
        print("Application shutdown complete")
       

app = FastAPI(
    title="Capstone project module b",
    lifespan=life_span
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React local
        "http://localhost:5173",   # Vite local
        "https://oravista.vercel.app", # Deployment preview
        "https://oravista.site"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

from src.routers.patient_routes import router as patient_router
from src.routers.dentist_routes import router as dentist_router
from src.routers.diagnostic_imaging_router import router as diagnostic_imaging_router
app.include_router(patient_router)
app.include_router(dentist_router)
app.include_router(diagnostic_imaging_router)
