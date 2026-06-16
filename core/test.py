import os
import requests
import numpy as np
import unittest
from unittest.mock import MagicMock, patch


# =========================
# GEMINI API TEST
# =========================

def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ API key not found")
        return

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }

    payload = {
        "contents": [{"parts": [{"text": "Explain AI in one sentence."}]}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print("Status Code:", response.status_code)

        if response.status_code != 200:
            print("❌ API call failed")
            return

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        print("\n✅ LLM OUTPUT:\n", text)

    except Exception as e:
        print("❌ Exception occurred:", e)


# =========================
# UNIT TESTS
# =========================

class TestDiagnosisEngine(unittest.TestCase):

    def setUp(self):
        from core.diagnosis import DiagnosisEngine
        self.engine = DiagnosisEngine()

    def test_stable_model(self):
        monitoring = {"degraded": False, "triggered_signals": []}
        result = self.engine.diagnose(monitoring)
        self.assertEqual(result["issue"], "no_issue")

    def test_concept_drift_detection(self):
        monitoring = {
            "degraded": True,
            "triggered_signals": [
                {"name": "confidence_drift"},
                {"name": "accuracy_drop"},
                {"name": "entropy_shift"}
            ]
        }
        result = self.engine.diagnose(monitoring)
        self.assertEqual(result["issue"], "concept_drift")

    def test_unknown_signals(self):
        monitoring = {
            "degraded": True,
            "triggered_signals": [{"name": "unknown_signal_xyz"}]
        }
        result = self.engine.diagnose(monitoring)
        self.assertEqual(result["issue"], "unknown")


class TestValidator(unittest.TestCase):

    def setUp(self):
        from core.validator import Validator
        self.validator = Validator()

    def _make_monitoring(self, accuracy):
        return {"current_metrics": {"accuracy": accuracy}}

    def test_promote_on_improvement(self):
        result = self.validator.validate(
            self._make_monitoring(0.70),
            self._make_monitoring(0.85)
        )
        self.assertEqual(result["decision"], "promote")

    def test_rollback_on_no_improvement(self):
        result = self.validator.validate(
            self._make_monitoring(0.80),
            self._make_monitoring(0.75)
        )
        self.assertEqual(result["decision"], "rollback")


class TestModelInterface(unittest.TestCase):

    def setUp(self):
        from core.model_interface import ModelInterface
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        self.interface = ModelInterface(mock_model)

    def test_predict_returns_correct_shape(self):
        X = np.zeros((2, 4))
        y_pred, probs = self.interface.predict(X)
        self.assertEqual(len(y_pred), 2)
        self.assertEqual(probs.shape, (2, 2))

    def test_evaluate_returns_keys(self):
        X = np.zeros((2, 4))
        y = np.array([0, 1])
        result = self.interface.evaluate(X, y)
        self.assertIn("y_true", result)
        self.assertIn("y_pred", result)
        self.assertIn("probabilities", result)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("=== Testing Gemini API ===")
    test_gemini()

    print("\n=== Running Unit Tests ===")
    unittest.main(argv=[""], exit=False, verbosity=2)
