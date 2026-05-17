import json
import google.generativeai as genai
from src.core.settings import settings
from src.schemas.schema import LLMProcedureOutcome 

class TreatmentPredictionAgent:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL or "gemini-2.5-flash"
        
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    async def predict_outcome(self, patient_context: dict) -> tuple[LLMProcedureOutcome, str, str]:
        """
        Calls Gemini to diagnose the single most urgent required procedure 
        and simulate a Weighted Logistic Regression calculation for it.
        """
        prompt = f"""
        You are an advanced clinical decision support engine. 
        
        TASK:
        1. Analyze the provided Patient Health Risk Profile.
        2. Identify the SINGLE most urgent or necessary clinical dental procedure this patient requires.
        3. Operate as a Weighted Logistic Regression model for this chosen procedure:
           - Start with the global clinical baseline success rate for the procedure.
           - Apply positive or negative logistic weights based on the patient's analytics metrics.
           - Calculate the final compound `success_probability` (0.0 to 100.0).
        4. Isolate the primary clinical vectors responsible for the variance in `key_factors`.
        
        CONSTRAINTS:
        - `confidence_level`: Must be exactly one of: "Low", "Medium", "High".
        - `recommendation`: Provide ONLY immediate procedural, surgical, or chairside guidance (e.g., "Use antibiotic prophylaxis due to diabetic history", "Extend integration healing period"). 
          DO NOT provide standard preventive advice (e.g., no brushing, flossing, or general diet suggestions). Keep it highly technical.
        
        PATIENT HEALTH RISK PROFILE:
        {json.dumps(patient_context, indent=2)}
        
        Return a JSON object matching the requested schema exactly.
        """
        
        response = await self.model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=LLMProcedureOutcome
            )
        )
        
        raw_text = response.text
        parsed_result = LLMProcedureOutcome.model_validate_json(raw_text)
        
        return parsed_result, prompt, raw_text
    