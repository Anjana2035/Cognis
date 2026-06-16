import os
import logging
import requests

logger = logging.getLogger(__name__)


class Explainer:
    """
    LLM-based narrator for Cognis.
    Uses Gemini REST API with fallback support.
    FIX: Removed wrong 'from openai import api_key' import.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        self.api_url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.0-flash:generateContent"
        )

    # =========================
    # PUBLIC METHOD
    # =========================

    def generate(self, model_name, before_monitoring, diagnosis, fix, after_monitoring=None):
        prompt = self._build_prompt(
            model_name,
            before_monitoring,
            diagnosis,
            fix,
            after_monitoring
        )

        if self.api_key:
            response = self._call_llm(prompt)
            if response:
                return response

        logger.warning("No API key or LLM call failed. Using fallback explainer.")
        return self._fallback(
            model_name,
            before_monitoring,
            diagnosis,
            fix,
            after_monitoring
        )

    # =========================
    # GEMINI REST CALL
    # =========================

    def _call_llm(self, prompt):
        try:
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key
            }

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return None

            data = response.json()

            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse Gemini response: {e}")
                return None

        except Exception as e:
            logger.error(f"LLM call exception: {e}")
            return None

    # =========================
    # PROMPT BUILDER
    # =========================

    def _build_prompt(self, model_name, before, diagnosis, fix, after):
        before_acc = before["current_metrics"]["accuracy"]

        after_acc = None
        if after:
            after_acc = after["current_metrics"]["accuracy"]

        return f"""
You are Cognis, an intelligent self-healing AI system.

Explain the model behavior step-by-step in a clear and slightly conversational tone.

Model: {model_name}

Before Fix:
Accuracy: {round(before_acc, 3)}

Diagnosis:
Issue: {diagnosis.get('issue')}
Reason: {diagnosis.get('reason')}

Fix Applied:
{fix.get('action')} ({fix.get('status')})

After Fix:
Accuracy: {round(after_acc, 3) if after_acc is not None else 'N/A'}

Instructions:
- Explain what happened
- Explain why the issue occurred
- Explain what fix was applied
- Compare before vs after
- If no improvement, say retry is needed
"""

    # =========================
    # FALLBACK
    # =========================

    def _fallback(self, model_name, before, diagnosis, fix, after):

        before_acc = before["current_metrics"]["accuracy"]
        after_acc = None

        if after:
            after_acc = after["current_metrics"]["accuracy"]

        if after_acc is not None:
            if after_acc > before_acc:
                improvement_text = "Model performance improved."
            elif after_acc == before_acc:
                improvement_text = "No noticeable improvement."
            else:
                improvement_text = "Model performance decreased."
        else:
            improvement_text = "Improvement unknown."

        return (
            f"Observing model '{model_name}'. \n\n "
            f"Initial accuracy: {round(before_acc, 3)}. \n\n"
            f"Detected issue: {diagnosis.get('issue')}. \n\n"
            f"Reason: {diagnosis.get('reason')}. \n\n"
            f"Applying fix: {fix.get('action')}.\n\n"
            f"Re-evaluating after applying fix... "
            f"New accuracy: {round(after_acc, 3) if after_acc is not None else 'N/A'}. "
            f"{improvement_text} "
            f"{'Retrying further optimization...' if improvement_text != 'Model performance improved.' else 'System stabilized.'}"
        )
