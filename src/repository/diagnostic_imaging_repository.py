from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
from src.models.model import User, PatientRecord, AIDiagnostic

class DiagnosticImagingRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def verify_patient_exists(self, patient_id: int) -> bool:
        """Ensures file metadata maps to a legitimate client entry."""
        query = select(User.id).where(User.id == patient_id, User.role == "patient")
        result = await self.db.execute(query)
        return result.scalar() is not None

    async def create_diagnostic_entry(self, patient_id: int, file_name: str, file_path: str, ai_findings: dict) -> AIDiagnostic:
        """Atomically commits tracking elements across administrative and AI diagnostic logs."""
        # 1. Archive file reference row within standard patient records directory
        record_entry = PatientRecord(
            user_id=patient_id,
            file_name=file_name,
            file_path=file_path
        )
        self.db.add(record_entry)
        
        # 2. Bind the prediction payload directly inside the dedicated diagnostic engine log
        diagnostic_entry = AIDiagnostic(
            patient_id=patient_id,
            ai_findings=ai_findings,
            clinical_notes=None
        )
        self.db.add(diagnostic_entry)
        
        # Flush and write structural states down to PostgreSQL instance
        await self.db.commit()
        await self.db.refresh(diagnostic_entry)
        return diagnostic_entry

    async def get_diagnostic_by_id(self, diagnostic_id: int) -> AIDiagnostic:
        """Retrieves raw data records to support targeted modification checks."""
        query = select(AIDiagnostic).where(AIDiagnostic.id == diagnostic_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def update_diagnostic_records(self, diagnostic_id: int, clinical_notes: str, ai_findings: dict) -> AIDiagnostic:
        """Updates clinical annotations and modifies findings inside JSONB arrays."""
        stmt = (
            update(AIDiagnostic)
            .where(AIDiagnostic.id == diagnostic_id)
            .values(clinical_notes=clinical_notes, ai_findings=ai_findings)
            .returning(AIDiagnostic)
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.scalar_one()
        