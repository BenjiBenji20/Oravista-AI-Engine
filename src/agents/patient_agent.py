import json
import google.generativeai as genai
from src.schemas.schema import LLMCheckupAssessment
from src.core.settings import settings

class PatientCheckupAgent:
    # TO UPDATE: transform into async code
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL or "gemini-2.5-flash"
        
        # We enforce JSON output matching our Pydantic schema
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    async def analyze_patient_risk(self, patient_context: dict) -> tuple[LLMCheckupAssessment, str, str]:
        """
        Calls Gemini to assess oral health risk asynchronously.
        Returns: (Parsed Pydantic Object, The Prompt Sent, The Raw JSON String)
        """
        prompt = f"""
        You are an expert AI Dental Diagnostic Assistant.
        Analyze the following patient analytics data and provide a clinical assessment.
        Focus on risks of periodontitis, caries, and enamel demineralization.
        
        Patient Context:
        {json.dumps(patient_context, indent=2)}
        
        Return a JSON object that strictly adheres to the requested schema.
        """
        
        # Generate content with structured output asynchronously
        response = await self.model.generate_content_async(
            prompt,
            # In Gemini SDK, you can pass the Pydantic schema type directly
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=LLMCheckupAssessment
            )
        )
        
        raw_text = response.text
        
        # Parse the guaranteed JSON string into our Pydantic model
        parsed_result = LLMCheckupAssessment.model_validate_json(raw_text)
        
        return parsed_result, prompt, raw_text