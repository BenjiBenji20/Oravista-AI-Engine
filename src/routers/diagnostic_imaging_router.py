from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.session import get_async_db
from src.schemas.diagnostic_schema import DiagnosticUploadResponse, UpdateAnnotationRequest, AIDiagnosticResponse
from src.modules.diagnostic_imaging_service import DiagnosticImagingService

router = APIRouter(prefix="/api/diagnostic-imaging", tags=["Dentist Only Diagnostic Imaging"])

@router.post("/upload", response_model=DiagnosticUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dental_scan(
    patient_id: int = Form(..., description="Target patient ID linked to the upload file."),
    file: UploadFile = File(..., description="Raw radiograph panoramic X-ray or intraoral snapshot image file."),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Ingests raw dental images, handles stateless streaming transfers to Supabase storage,
    runs localized pathology object detection, and maps prediction bounding box arrays.
    """
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a standard PNG or JPEG image file."
        )

    service = DiagnosticImagingService(db)
    return await service.process_and_log_scan(patient_id=patient_id, file=file)


@router.put("/{diagnostic_id}/annotate", response_model=AIDiagnosticResponse)
async def update_dentist_annotation(
    diagnostic_id: int,
    payload: UpdateAnnotationRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Saves professional corrections, annotations, and final manual marking metrics
    to ensure absolute human expert control over AI diagnostic findings.
    """
    service = DiagnosticImagingService(db)
    return await service.apply_dentist_annotations(
        diagnostic_id=diagnostic_id,
        clinical_notes=payload.clinical_notes,
        verified_findings=payload.human_verified_findings
    )