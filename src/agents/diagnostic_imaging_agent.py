import json
import google.generativeai as genai
from typing import List
from pydantic import BaseModel, Field
from src.core.settings import settings

# Structured Output Schemas for Gemini Enforced JSON Engine
class LLMBoundingBox(BaseModel):
    x_min: float
    y_min: float
    width: float
    height: float

class LLMPathologyPrediction(BaseModel):
    class_id: int
    name: str
    confidence: float
    box: LLMBoundingBox

class GeminiDentalDiagnosticResponse(BaseModel):
    predictions: List[LLMPathologyPrediction]
    clinical_notes: str

class LLMTextOnlyNotesResponse(BaseModel):
    clinical_notes: str


class DiagnosticImagingAgent:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL or "gemini-2.5-flash"
        self.model = genai.GenerativeModel(self.model_name)

    async def analyze_image_multimodal(self, image_bytes: bytes) -> tuple[GeminiDentalDiagnosticResponse, str, str]:
        """
        Executes a complete multimodal analysis fallback pass over raw radiograph bytes.
        """
        prompt = """
        You are an expert maxillofacial radiologist and clinical decision support AI asset. 
        Analyze this dental radiograph/intraoral image file with high diagnostic specificity.
        
        Locate all visible pathological anomalies and map them strictly to this Class ID matrix layout:
        - 0: Caries
        - 1: Filling
        - 2: Impacted
        - 3: Crown
        - 4: Calculus
        - 13: Bone Loss
        
        For every localized finding, track its position and output a clean bounding box using NORMALIZED FLOAT COORDINATES between 0.0 and 1.0:
        - x_min: distance from left edge to the box's left boundary, divided by total image width.
        - y_min: distance from top edge to the box's top boundary, divided by total image height.
        - width: box width divided by total image width.
        - height: box height divided by total image height.
        
        Also compile a detailed, expert-level string for 'clinical_notes' providing actionable preventative or restorative path guidelines.
        Return a JSON object matching the requested schema exactly.
        """
        
        # Format raw image binary data stream for the Google GenAI SDK inline payload
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }

        response = await self.model.generate_content_async(
            [prompt, image_part],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=GeminiDentalDiagnosticResponse,
                temperature=0.1
            )
        )
        
        raw_text = response.text
        parsed_result = GeminiDentalDiagnosticResponse.model_validate_json(raw_text)
        return parsed_result, prompt, raw_text

    async def generate_notes_from_findings(self, findings_summary: list) -> str:
        """
        Fast text-only helper that creates professional clinical recommendation notes
        based on the bounding boxes detected by your local custom ONNX model.
        """
        prompt = f"""
        You are an expert dentist writing clinical notes for a patient chart.
        Based on these findings discovered by our diagnostic imaging software, write a brief, professional, 
        and actionable clinical note describing recommendations or next steps for the dental clinic staff.
        
        FINDINGS:
        {json.dumps(findings_summary, indent=2)}
        
        Return a JSON object containing a single 'clinical_notes' string key matching the schema exactly.
        """
        
        response = await self.model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=LLMTextOnlyNotesResponse,
                temperature=0.2
            )
        )
        
        parsed = json.loads(response.text)
        return parsed.get("clinical_notes", "Routine evaluation advised.")