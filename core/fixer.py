import logging
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)


class Fixer:
    """
    Advanced Fixer with validation-driven strategy selection.
    FIX: temperature_scaling now stores scaled probs on model and actually affects predictions.
    FIX: temperature is now configurable via constructor.
    """

    def __init__(self, temperature=1.5):
        self.temperature = temperature  # FIX: configurable, not hard-coded

        self.strategy_map = {
            "concept_drift": [self._fine_tune],
            "class_imbalance": [self._reweight_classes],
            "label_noise": [self._noise_weighting, self._reweight_classes],
            "calibration_error": [self._temperature_scaling]
        }

    def apply_fix(self, model_interface, diagnosis_output, X, y):
        issue = diagnosis_output.get("issue")

        if issue not in self.strategy_map:
            logger.warning(f"No fix strategy for issue: {issue}")
            return {
                "action": "none",
                "status": "skipped",
                "reason": "No applicable fix"
            }

        strategies = self.strategy_map[issue]

        baseline_preds, baseline_probs = model_interface.predict(X)
        baseline_acc = np.mean(baseline_preds == y)

        best_acc = baseline_acc
        best_result = None
        best_model = None

        for strategy in strategies:
            try:
                candidate = deepcopy(model_interface)
                result = strategy(candidate, X, y)

                preds, _ = candidate.predict(X)
                acc = np.mean(preds == y)

                logger.info(f"Strategy '{result['action']}' accuracy: {round(acc, 4)}")

                if acc > best_acc:
                    best_acc = acc
                    best_result = result
                    best_model = candidate

            except Exception as e:
                logger.error(f"Strategy failed: {e}")
                continue

        if best_model is not None:
            model_interface.model = best_model.model
            # FIX: also carry over temperature_probs if set
            if hasattr(best_model, "_temperature_probs"):
                model_interface._temperature_probs = best_model._temperature_probs
                model_interface._temperature = best_model._temperature

            logger.info(f"Applied fix: {best_result['action']} | improvement: {round(best_acc - baseline_acc, 4)}")

            return {
                "action": best_result["action"],
                "status": "applied",
                "details": best_result.get("details", ""),
                "improvement": round(best_acc - baseline_acc, 4)
            }

        logger.warning("No strategy improved performance.")
        return {
            "action": "none",
            "status": "failed",
            "reason": "No strategy improved performance",
            "baseline_accuracy": round(baseline_acc, 4)
        }

    # =========================
    # STRATEGIES
    # =========================

    def _fine_tune(self, interface, X, y):
        interface.train(X, y)
        return {
            "action": "fine_tuning",
            "details": "Retrained model on available data"
        }

    def _reweight_classes(self, interface, X, y):
        y_int = y.astype(int)

        class_counts = np.bincount(y_int)
        class_counts[class_counts == 0] = 1

        weights = 1.0 / class_counts
        sample_weights = np.array([weights[int(label)] for label in y_int])

        interface.model.fit(X, y, sample_weight=sample_weights)

        return {
            "action": "class_reweighting",
            "details": "Applied inverse class weighting"
        }

    def _noise_weighting(self, interface, X, y):
        y_pred, probs = interface.predict(X)
        confidence = probs.max(axis=1)

        noisy_mask = (confidence > 0.8) & (y_pred != y)
        weights = np.ones(len(y))
        weights[noisy_mask] = 0.3

        interface.model.fit(X, y, sample_weight=weights)

        return {
            "action": "noise_weighting",
            "details": f"Down-weighted {np.sum(noisy_mask)} noisy samples"
        }

    def _temperature_scaling(self, interface, X, y):
        """
        FIX: Now stores scaled probabilities on the interface so predict() uses them.
        Temperature scaling adjusts confidence — stored and applied in ModelInterface.
        """
        _, probs = interface.predict(X)

        scaled_probs = probs ** (1 / self.temperature)
        scaled_probs /= scaled_probs.sum(axis=1, keepdims=True)

        # FIX: Store temperature config on interface so predict() uses it
        interface._temperature_probs = scaled_probs
        interface._temperature = self.temperature

        return {
            "action": "temperature_scaling",
            "details": f"Applied temperature scaling (T={self.temperature})"
        }
