import json
import google.generativeai as genai
from src.schemas.schema import LLMNoShowAssessment
from src.core.settings import settings
from pydantic import BaseModel, Field



class NoShowPredictionAgent:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL or "gemini-2.5-flash"
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    async def predict_no_show(self, appointment_context: dict) -> tuple[LLMNoShowAssessment, str, str]:
        prompt = f"""
        You are an operational clinic analytics engine operating as a Behavioral Random Forest Classification model.
        
        TASK:
        Evaluate the missing/no-show risk probability for the following scheduled appointment.
        1. Evaluate historical behavioral metrics (missed ratios, baseline reliability).
        2. Adjust risk weights using situational vectors (travel distance, age, booking lead days, and weather risk factors).
        3. Quantify final `no_show_probability` (0.0 to 100.0).
        
        CRITICAL FORMATTING CONSTRAINT FOR 'reasoning':
        Provide exactly 1 or 2 concise, comma-separated clinical/operational reasons matching this style: 
        "Missed last 2 appts, High travel distance" or "Long commute, Inclement weather warning". Keep it under 12 words total.

        APPOINTMENT & PATIENT CONTEXT:
        {json.dumps(appointment_context, indent=2)}
        
        Return a JSON object matching the requested schema exactly.
        """
        
        response = await self.model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=LLMNoShowAssessment
            )
        )
        
        raw_text = response.text
        parsed_result = LLMNoShowAssessment.model_validate_json(raw_text)
        return parsed_result, prompt, raw_text
    