import os
import logging
import requests

logger = logging.getLogger(__name__)


class Explainer:
    """
    LLM-based narrator for Cognis.
    Uses Gemini REST API with fallback support.

    FIX: generate() previously had no visibility into what the Validator
    actually decided (promote / rollback). It guessed "promoted" purely by
    comparing before/after accuracy in the fallback text, and never told
    the LLM prompt what the real decision was either. That guess can
    contradict Fixer's own status label — e.g. a strategy the Fixer marks
    "applied_no_gain" (net_improvement <= 0) could still be narrated as
    "the fix was successful and the model has been promoted" if the
    accuracy numbers happened to look favourable in isolation.

    validation_output (the dict returned by Validator.validate(), containing
    at minimum a "decision" key of "promote" / "rollback") is now threaded
    through generate() -> _build_prompt() / _fallback(), and is the single
    source of truth for whether the narration says promoted or rolled back.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.0-flash:generateContent"
        )

    def generate(self, model_name, before_monitoring, diagnosis, fix,
                 after_monitoring=None, step=0, validation_output=None, **kwargs):
        prompt = self._build_prompt(
            model_name, before_monitoring, diagnosis, fix, after_monitoring, validation_output
        )

        if self.api_key:
            response = self._call_llm(prompt)
            if response:
                return response

        logger.warning("Using fallback explainer.")
        return self._fallback(
            model_name, before_monitoring, diagnosis, fix, after_monitoring, step, validation_output
        )

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

    def _build_prompt(self, model_name, before, diagnosis, fix, after, validation_output=None):
        before_acc = before["current_metrics"]["accuracy"]
        after_acc  = after["current_metrics"]["accuracy"] if after else None

        # Tell the model the REAL outcome instead of letting it infer one
        # from raw numbers, which is what produced the earlier contradiction.
        if validation_output is not None:
            decision_line = f"Validator Decision: {validation_output.get('decision')} (this is the ground truth outcome — do not contradict it)"
        else:
            decision_line = "Validator Decision: n/a (no healing attempted)"

        return f"""
You are Cognis, an intelligent self-healing AI system.
Explain what happened to this model in a clear, professional tone. No bullet points. Max 4 sentences.

Model: {model_name}
Before — Accuracy: {round(before_acc, 3)}
Diagnosis: {diagnosis.get('issue')} — {diagnosis.get('reason')}
Fix Applied: {fix.get('action')} ({fix.get('status')})
{decision_line}
After — Accuracy: {round(after_acc, 3) if after_acc is not None else 'N/A'}

Explain what went wrong, what fix was applied, and whether it was promoted or rolled back,
strictly matching the Validator Decision above. If it was rolled back, say the system will
retry with a different strategy. Do not describe a rolled-back fix as successful.
"""

    def _fallback(self, model_name, before, diagnosis, fix, after, step=0, validation_output=None):
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

        # Ground truth: what did the Validator actually decide?
        decision = validation_output.get("decision") if validation_output else None

        if after_acc is not None:
            delta = round(abs(after_acc - before_acc), 3)

            if decision == "promote":
                result_line = (
                    f"After applying {action}, accuracy moved from "
                    f"{round(before_acc, 3)} to {round(after_acc, 3)} "
                    f"(delta {delta}). The fix was accepted and the model has been promoted."
                )
            elif decision == "rollback":
                result_line = (
                    f"After applying {action}, accuracy moved from "
                    f"{round(before_acc, 3)} to {round(after_acc, 3)}, but the change "
                    f"was rejected by validation and the previous model state was restored. "
                    f"The system will retry with a different strategy."
                )
            else:
                # No validator decision available — fall back to a neutral,
                # non-committal description rather than guessing.
                result_line = (
                    f"After applying {action}, accuracy is now {round(after_acc, 3)} "
                    f"(previously {round(before_acc, 3)}). Evaluating whether to keep this change."
                )
        else:
            result_line = f"Evaluating the impact of {action}."

        return (
            f"Attempt {step + 1}: {issue} detected in {model_name}. "
            f"{reason} "
            f"Applying {action} ({status}). "
            f"{result_line}"
        )
