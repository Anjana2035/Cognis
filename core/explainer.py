import requests


class Explainer:
    """
    LLM-based narrator for Cognis.
    Produces step-wise explanations with fallback support.
    """

    def __init__(self, api_url=None, api_key=None):
        self.api_url = api_url
        self.api_key = api_key

    # =========================
    # PUBLIC METHOD
    # =========================

    def generate(self, model_name, before_monitoring, diagnosis, fix, after_monitoring=None):
        """
        Generates explanation for one iteration.
        """

        prompt = self._build_prompt(
            model_name,
            before_monitoring,
            diagnosis,
            fix,
            after_monitoring
        )

        if self.api_url:
            response = self._call_llm(prompt)
            if response:
                return response

        return self._fallback(
            model_name,
            before_monitoring,
            diagnosis,
            fix,
            after_monitoring
        )

    # =========================
    # LLM CALL
    # =========================

    def _call_llm(self, prompt):
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "prompt": prompt,
                    "max_tokens": 300
                },
                timeout=5
            )

            if response.status_code != 200:
                return None

            data = response.json()
            return data.get("text") or data.get("response")

        except Exception:
            return None

    # =========================
    # PROMPT BUILDER
    # =========================

    def _build_prompt(self, model_name, before, diagnosis, fix, after):
        return f"""
You are an intelligent AI system explaining model health step-by-step.

Model: {model_name}

Before Fix:
{before}

Diagnosis:
{diagnosis}

Fix Applied:
{fix}

After Fix:
{after}

Explain clearly:
1. What was observed initially
2. What problem was detected
3. Why it occurred
4. What fix was applied
5. What changed after the fix
6. Whether performance improved or not
7. If not improved, mention retry
"""

    # =========================
    # FALLBACK
    # =========================

    def _fallback(self, model_name, before, diagnosis, fix, after):

        before_acc = before["current_metrics"]["accuracy"]
        after_acc = None

        if after:
            after_acc = after["current_metrics"]["accuracy"]

        improvement_text = "unknown"

        if after_acc is not None:
            if after_acc > before_acc:
                improvement_text = "Model performance improved."
            elif after_acc == before_acc:
                improvement_text = "No noticeable improvement."
            else:
                improvement_text = "Model performance decreased."

        return (
            f"Observing model '{model_name}'. "
            f"Initial accuracy: {round(before_acc, 3)}. "
            f"Detected issue: {diagnosis.get('issue')}. "
            f"Reason: {diagnosis.get('reason')}. "
            f"Applying fix: {fix.get('action')}.\n\n"

            f"Re-evaluating after applying fix... "

            f"New accuracy: {round(after_acc, 3) if after_acc is not None else 'N/A'}. "
            f"{improvement_text} "

            f"{'Retrying further optimization...' if improvement_text != 'Model performance improved.' else 'System stabilized.'}"
        )