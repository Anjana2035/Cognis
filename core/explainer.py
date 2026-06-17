import os
import logging
import requests

logger = logging.getLogger(__name__)


class Explainer:
    """
    LLM-based narrator for Cognis.
    Uses Gemini REST API with fallback support.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.0-flash:generateContent"
        )

    def generate(self, model_name, before_monitoring, diagnosis, fix,
                 after_monitoring=None, step=0, **kwargs):
        prompt = self._build_prompt(model_name, before_monitoring, diagnosis, fix, after_monitoring)

        if self.api_key:
            response = self._call_llm(prompt)
            if response:
                return response

        logger.warning("Using fallback explainer.")
        return self._fallback(model_name, before_monitoring, diagnosis, fix, after_monitoring, step)

    def _call_llm(self, prompt):
        try:
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key
            }
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)

            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return None

            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:
            logger.error(f"LLM call exception: {e}")
            return None

    def _build_prompt(self, model_name, before, diagnosis, fix, after):
        before_acc = before["current_metrics"]["accuracy"]
        after_acc  = after["current_metrics"]["accuracy"] if after else None

        return f"""
You are Cognis, an intelligent self-healing AI system.
Explain what happened to this model in a clear, professional tone. No bullet points. Max 4 sentences.

Model: {model_name}
Before — Accuracy: {round(before_acc, 3)}
Diagnosis: {diagnosis.get('issue')} — {diagnosis.get('reason')}
Fix Applied: {fix.get('action')} ({fix.get('status')})
After — Accuracy: {round(after_acc, 3) if after_acc is not None else 'N/A'}

Explain what went wrong, what fix was applied, and whether it helped.
If accuracy did not improve, say the system will retry with a different strategy.
"""

    def _fallback(self, model_name, before, diagnosis, fix, after, step=0):
        before_acc = before["current_metrics"]["accuracy"]
        after_acc  = after["current_metrics"]["accuracy"] if after else None

        issue  = diagnosis.get("issue", "unknown").replace("_", " ").title()
        action = fix.get("action", "none").replace("_", " ").title()
        status = fix.get("status", "unknown")
        reason = diagnosis.get("reason", "")

        # Healthy — no fix needed
        if fix.get("action") == "none" and fix.get("status") == "skipped":
            return (
                f"Attempt {step + 1}: {model_name} passed the health check. "
                f"No degradation detected — the model is performing as expected."
            )

        if after_acc is not None:
            improved = after_acc > before_acc
            delta    = round(abs(after_acc - before_acc), 3)

            if improved:
                result_line = (
                    f"After applying {action}, accuracy moved from "
                    f"{round(before_acc, 3)} to {round(after_acc, 3)} (+{delta}). "
                    f"The fix was successful and the model has been promoted."
                )
            else:
                result_line = (
                    f"After applying {action}, accuracy remained at "
                    f"{round(after_acc, 3)} — no measurable improvement. "
                    f"The candidate was rejected and the previous model state restored. "
                    f"The system will retry with a different strategy."
                )
        else:
            result_line = f"Evaluating the impact of {action}."

        return (
            f"Attempt {step + 1}: {issue} detected in {model_name}. "
            f"{reason} "
            f"Applying {action} ({status}). "
            f"{result_line}"
        )
