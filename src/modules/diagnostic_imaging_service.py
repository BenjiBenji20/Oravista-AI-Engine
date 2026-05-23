import os
import secrets
import asyncio
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from supabase import create_client, Client

from src.repository.diagnostic_imaging_repository import DiagnosticImagingRepository
from src.schemas.diagnostic_schema import DiagnosticUploadResponse, PathologyPrediction, BoundingBox
from src.agents.diagnostic_imaging_agent import DiagnosticImagingAgent
from src.core.settings import settings

# 1. Cloud Storage Credentials Check
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_SERVICE_ROLE_KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing required Supabase integration credentials.")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# 2. Local File System Paths Mappings (Corrected to src/weights)
# ---------------------------------------------------------------------------
# Locates the current directory of this service file (src/modules)
CURRENT_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolves to the parent 'src' directory dynamically
SRC_DIR = os.path.dirname(CURRENT_MODULE_DIR)

# Builds absolute paths targeting the weights folder inside src/
ONNX_PATH = os.path.join(SRC_DIR, "weights", "oravista_dental_image_diagnosis_v1.onnx")
ANCHORS_PATH = os.path.join(SRC_DIR, "weights", "anchors.npy") # Plural matching your filesystem

if not os.path.exists(ONNX_PATH) or not os.path.exists(ANCHORS_PATH):
    raise FileNotFoundError(
        f"Missing core engine runtime weights or matrices.\n"
        f"Checked Absolute Locations:\n"
        f" -> ONNX: {ONNX_PATH}\n"
        f" -> Anchors: {ANCHORS_PATH}"
    )

print("🚀 Booting up local high-performance RetinaNet Engine inside Codespace...")
ORT_SESSION = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
STATIC_ANCHORS = np.load(ANCHORS_PATH)
print("🎯 ONNX weights and anchor matrices successfully loaded and synchronized.")

CLASS_MAPPING = {0: "Caries", 1: "Filling", 2: "Impacted", 3: "Crown", 4: "Calculus", 13: "Bone Loss"}


# 3. Synchronous NumPy Matrix Processing Block
def decode_and_nms_pipeline(onnx_outputs, anchors, conf_thresh=0.40, iou_thresh=0.30):
    raw_deltas = onnx_outputs[0][0]
    raw_logits = onnx_outputs[1][0]

    # Compute Sigmoid probabilities
    confidences = 1 / (1 + np.exp(-raw_logits))
    best_class_ids = np.argmax(confidences, axis=-1)
    best_scores = np.max(confidences, axis=-1)

    valid_mask = best_scores > conf_thresh
    if not np.any(valid_mask):
        return []

    filtered_deltas = raw_deltas[valid_mask]
    filtered_anchors = anchors[valid_mask]
    filtered_scores = best_scores[valid_mask]
    filtered_classes = best_class_ids[valid_mask]

    a_cx, a_cy, a_w, a_h = filtered_anchors[:, 0], filtered_anchors[:, 1], filtered_anchors[:, 2], filtered_anchors[:, 3]
    d_cx, d_cy, d_w, d_h = filtered_deltas[:, 0], filtered_deltas[:, 1], filtered_deltas[:, 2], filtered_deltas[:, 3]

    cx = d_cx * a_w + a_cx
    cy = d_cy * a_h + a_cy
    w = np.exp(d_w) * a_w
    h = np.exp(d_h) * a_h

    x_min = cx - (w / 2.0)
    y_min = cy - (h / 2.0)
    x_max = cx + (w / 2.0)
    y_max = cy + (h / 2.0)

    final_detections = []
    unique_classes = np.unique(filtered_classes)

    for c in unique_classes:
        class_idx = np.where(filtered_classes == c)[0]
        c_x1, c_y1, c_x2, c_y2 = x_min[class_idx], y_min[class_idx], x_max[class_idx], y_max[class_idx]
        c_scores = filtered_scores[class_idx]

        areas = (c_x2 - c_x1) * (c_y2 - c_y1)
        order = c_scores.argsort()[::-1]

        keep_indices = []
        while order.size > 0:
            idx = order[0]
            keep_indices.append(idx)

            xx1 = np.maximum(c_x1[idx], c_x1[order[1:]])
            yy1 = np.maximum(c_y1[idx], c_y1[order[1:]])
            xx2 = np.minimum(c_x2[idx], c_x2[order[1:]])
            yy2 = np.minimum(c_y2[idx], c_y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h

            iou = intersection / (areas[idx] + areas[order[1:]] - intersection + 1e-8)
            order = order[np.where(iou <= iou_thresh)[0] + 1]

        for idx in keep_indices:
            global_idx = class_idx[idx]
            final_detections.append({
                "class_id": int(c),
                "name": CLASS_MAPPING.get(int(c), "Unknown"),
                "confidence": float(filtered_scores[global_idx]),
                "box": {
                    "x_min": float(max(0.0, min(1.0, x_min[global_idx] / 640.0))),
                    "y_min": float(max(0.0, min(1.0, y_min[global_idx] / 640.0))),
                    "width": float(max(0.0, min(1.0, w[global_idx] / 640.0))),
                    "height": float(max(0.0, min(1.0, h[global_idx] / 640.0)))
                }
            })

    return final_detections


# 4. Asynchronous Core Service Orchestrator
class DiagnosticImagingService:
    def __init__(self, db: AsyncSession):
        self.repository = DiagnosticImagingRepository(db)
        self.agent = DiagnosticImagingAgent()

    async def process_and_log_scan(self, patient_id: int, file: UploadFile) -> DiagnosticUploadResponse:
        print(f"\n[DATA FLOW] 1. Initializing diagnostic ingestion request for Patient ID: {patient_id}")
        
        patient_exists = await self.repository.verify_patient_exists(patient_id)
        if not patient_exists:
            raise HTTPException(status_code=404, detail="Target patient records missing from system registry.")

        file_ext = os.path.splitext(file.filename)[1]
        safe_filename = f"{secrets.token_hex(8)}{file_ext}"

        # Ingest file bytes into memory
        file_bytes = await file.read()
        
        print("[DATA FLOW] 2. Streaming raw image binary up to Supabase Storage Bucket...")
        try:
            supabase_client.storage.from_("dental-scans").upload(
                path=safe_filename,
                file=file_bytes,
                file_options={"content-type": file.content_type}
            )
            public_url = supabase_client.storage.from_("dental-scans").get_public_url(safe_filename)
            print(f"[DATA FLOW] -> Public URL secured: {public_url}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cloud asset pipeline streaming stall: {str(e)}")

        print("[DATA FLOW] 3. Initializing local ResNet-50 / ONNX forward inference mapping pass...")
        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (640, 640))
            input_data = np.expand_dims(resized_img, axis=0).astype(np.float32)

            input_name = ORT_SESSION.get_inputs()[0].name
            onnx_outputs = await asyncio.to_thread(lambda: ORT_SESSION.run(None, {input_name: input_data}))
            
            # Extract localized detections breaking the 0.40 confidence barrier
            local_detections = decode_and_nms_pipeline(onnx_outputs, STATIC_ANCHORS, conf_thresh=0.40)
            print(f"[DATA FLOW] -> Local engine evaluation completed. Found {len(local_detections)} candidate boxes.")
        except Exception as e:
            print(f"⚠️ Local engine crashed: {str(e)}. Defaulting straight to fallback pipeline.")
            local_detections = []

        # -------------------------------------------------------------------
        # EVALUATE FALLBACK LOGIC TRIGGERS (THRESHOLD = 0.40)
        # -------------------------------------------------------------------
        highest_score = max([d["confidence"] for d in local_detections]) if local_detections else 0.0
        
        trigger_fallback = False
        trigger_reason = ""

        if len(local_detections) == 0:
            trigger_fallback = True
            trigger_reason = "Trigger 1: Zero local detections broke the 0.40 confidence threshold."
        elif len(local_detections) > 8:
            trigger_fallback = True
            trigger_reason = "Trigger 2: Local detector hallucinated excessive feature boxes (>8 instances)."
        elif 0.40 <= highest_score <= 0.48:
            trigger_fallback = True
            trigger_reason = "Trigger 3: Detections are clustered entirely inside the low-certainty gray-zone (0.40 to 0.48)."

        final_predictions = []
        final_notes = ""

        if trigger_fallback:
            print(f"\n[🚨 HYBRID FALLBACK ACTUATED] Reason: {trigger_reason}")
            print("[DATA FLOW] Routing raw file stream directly to Gemini Multimodal vision engine...")
            
            gemini_result, _, _ = await self.agent.analyze_image_multimodal(file_bytes)
            final_notes = gemini_result.clinical_notes

            for pred in gemini_result.predictions:
                final_predictions.append(
                    PathologyPrediction(
                        class_id=pred.class_id,
                        name=pred.name,
                        confidence=round(pred.confidence, 2),
                        box=BoundingBox(
                            x_min=pred.box.x_min,
                            y_min=pred.box.y_min,
                            width=pred.box.width,
                            height=pred.box.height
                        )
                    )
                )
            print(f"[DATA FLOW] -> Gemini diagnostics completed. Captured {len(final_predictions)} validated anomalies.")
        else:
            print("\n[🎯 LOCAL DISPATCH SUCCESSFUL] Custom ResNet model outputs are stable.")
            print("[DATA FLOW] Requesting text-only clinical advisory summary notes from Gemini...")
            
            # Package findings for the text-only notes helper
            findings_summary = [{"class_id": d["class_id"], "name": d["name"], "confidence": d["confidence"]} for d in local_detections]
            final_notes = await self.agent.generate_notes_from_findings(findings_summary)

            for det in local_detections:
                final_predictions.append(
                    PathologyPrediction(
                        class_id=det["class_id"],
                        name=det["name"],
                        confidence=round(det["confidence"], 2),
                        box=BoundingBox(**det["box"])
                    )
                )

        # -------------------------------------------------------------------
        # DATABASE COMMIT LAYER
        # -------------------------------------------------------------------
        ai_findings_payload = {
            "predictions": [p.model_dump() for p in final_predictions],
            "human_verified": False,
            "fallback_triggered": trigger_fallback
        }

        print("[DATA FLOW] 4. Committing finalized data models to relational PostgreSQL tables...")
        diagnostic_record = await self.repository.create_diagnostic_entry(
            patient_id=patient_id,
            file_name=file.filename,
            file_path=public_url,
            ai_findings=ai_findings_payload
        )
        
        # Manually save the clinical recommendation notes to the records
        await self.repository.update_diagnostic_records(
            diagnostic_id=diagnostic_record.id,
            clinical_notes=final_notes,
            ai_findings=ai_findings_payload
        )

        print("[DATA FLOW] 5. Request completed. Forwarding structured output schemas back to React dashboard.\n")
        return DiagnosticUploadResponse(
            diagnostic_id=diagnostic_record.id,
            patient_id=diagnostic_record.patient_id,
            file_path=public_url,
            predictions=final_predictions,
            clinical_notes=final_notes,
            scan_date=diagnostic_record.scan_date
        )

    async def apply_dentist_annotations(self, diagnostic_id: int, clinical_notes: str, verified_findings: dict):
        record = await self.repository.get_diagnostic_by_id(diagnostic_id)
        if not record:
            raise HTTPException(status_code=404, detail="AI Diagnostic record not found.")

        updated_findings = dict(record.ai_findings) if record.ai_findings else {}
        updated_findings["human_verified"] = True
        
        actual_array = verified_findings["annotations"] if "annotations" in verified_findings else verified_findings
        updated_findings["annotations"] = actual_array

        return await self.repository.update_diagnostic_records(
            diagnostic_id=diagnostic_id,
            clinical_notes=clinical_notes,
            ai_findings=updated_findings
        )
