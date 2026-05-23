import os
import secrets
import random
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from supabase import create_client, Client

from src.repository.diagnostic_imaging_repository import DiagnosticImagingRepository
from src.schemas.diagnostic_schema import DiagnosticUploadResponse, PathologyPrediction, BoundingBox
from src.core.settings import settings

# Initialize the Supabase Client
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_SERVICE_ROLE_KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase configuration environment variables.")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class DiagnosticImagingService:
    def __init__(self, db: AsyncSession):
        self.repository = DiagnosticImagingRepository(db)

    async def process_and_log_scan(self, patient_id: int, file: UploadFile) -> DiagnosticUploadResponse:
        # 1. Verify target patient entity exists before doing network work
        patient_exists = await self.repository.verify_patient_exists(patient_id)
        if not patient_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Patient with ID {patient_id} does not exist in the system records."
            )

        # 2. Generate a secure, unique filename to prevent overwrites
        file_ext = os.path.splitext(file.filename)[1]
        safe_filename = f"{secrets.token_hex(8)}{file_ext}"

        try:
            # 3. Read raw bytes directly from memory
            file_bytes = await file.read()

            # 4. Upload the file stream directly to your Supabase Storage Bucket
            supabase_client.storage.from_("dental-scans").upload(
                path=safe_filename,
                file=file_bytes,
                file_options={"content-type": file.content_type}
            )

            # 5. Extract the direct public asset URL link
            public_url = supabase_client.storage.from_("dental-scans").get_public_url(safe_filename)

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to stream asset to Supabase Cloud Storage: {str(e)}"
            )

        # 6. Run your AI Machine Learning Code Logic here
        # Simulated Object Detection predictions generating normalized bounding boxes
        simulated_predictions = [
            PathologyPrediction(
                class_id=0, 
                name="Caries", 
                confidence=round(random.uniform(0.75, 0.98), 2),
                box=BoundingBox(
                    x_min=0.22,  # 22% from the left edge of the image
                    y_min=0.45,  # 45% from the top edge of the image
                    width=0.08,  # Box takes up 8% of the image's total width
                    height=0.06  # Box takes up 6% of the image's total height
                )
            ),
            PathologyPrediction(
                class_id=13, 
                name="Bone Loss", 
                confidence=round(random.uniform(0.60, 0.88), 2),
                box=BoundingBox(
                    x_min=0.55, 
                    y_min=0.62, 
                    width=0.15, 
                    height=0.05
                )
            )
        ]

        # Structure payload object maps to bind inside PostgreSQL JSONB formats natively
        ai_findings_payload = {
            "predictions": [p.model_dump() for p in simulated_predictions],
            "human_verified": False
        }

        # 7. Persist the record. 'public_url' is saved directly into 'file_path'
        diagnostic_record = await self.repository.create_diagnostic_entry(
            patient_id=patient_id,
            file_name=file.filename,
            file_path=public_url,
            ai_findings=ai_findings_payload
        )

        return DiagnosticUploadResponse(
            diagnostic_id=diagnostic_record.id,
            patient_id=diagnostic_record.patient_id,
            file_path=public_url,
            predictions=simulated_predictions,
            scan_date=diagnostic_record.scan_date
        )

    async def apply_dentist_annotations(self, diagnostic_id: int, clinical_notes: str, verified_findings: dict):
        record = await self.repository.get_diagnostic_by_id(diagnostic_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"AI Diagnostic report entry with ID {diagnostic_id} cannot be isolated."
            )

        # Copy the existing JSONB data structure safely
        updated_findings = dict(record.ai_findings) if record.ai_findings else {}
        updated_findings["human_verified"] = True

        # Safely extract the raw array list regardless of whether the agent/client 
        # sent it wrapped under an "annotations" key or as a direct body dict.
        if "annotations" in verified_findings:
            actual_array = verified_findings["annotations"]
        else:
            actual_array = verified_findings

        # Enforce a uniform, flat array mapping inside the database column
        updated_findings["annotations"] = actual_array

        return await self.repository.update_diagnostic_records(
            diagnostic_id=diagnostic_id,
            clinical_notes=clinical_notes,
            ai_findings=updated_findings
        )