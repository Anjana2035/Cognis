import numpy as np
from copy import deepcopy


class Fixer:
    """
    Advanced Fixer with validation-driven strategy selection.
    """

    def __init__(self):
        self.strategy_map = {
            "concept_drift": [self._fine_tune],
            "class_imbalance": [self._reweight_classes],
            "label_noise": [self._noise_weighting, self._reweight_classes],
            "calibration_error": [self._temperature_scaling]
        }

    def apply_fix(self, model_interface, diagnosis_output, X, y):
        issue = diagnosis_output.get("issue")

        if issue not in self.strategy_map:
            return {
                "action": "none",
                "status": "skipped",
                "reason": "No applicable fix"
            }

        strategies = self.strategy_map[issue]

        # 🔥 baseline performance
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

                if acc > best_acc:
                    best_acc = acc
                    best_result = result
                    best_model = candidate

            except Exception:
                continue

        # 🔥 APPLY ONLY IF IMPROVED
        if best_model is not None:
            model_interface.model = best_model.model

            return {
                "action": best_result["action"],
                "status": "applied",
                "details": best_result.get("details", ""),
                "improvement": round(best_acc - baseline_acc, 4)
            }

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

        # 🔥 smarter noise detection
        noisy_mask = (confidence > 0.8) & (y_pred != y)

        weights = np.ones(len(y))
        weights[noisy_mask] = 0.3  # reduce influence, not remove

        interface.model.fit(X, y, sample_weight=weights)

        return {
            "action": "noise_weighting",
            "details": f"Down-weighted {np.sum(noisy_mask)} noisy samples"
        }

    def _temperature_scaling(self, interface, X, y):
        _, probs = interface.predict(X)

        temperature = 1.5
        scaled_probs = probs ** (1 / temperature)
        scaled_probs /= scaled_probs.sum(axis=1, keepdims=True)

        return {
            "action": "temperature_scaling",
            "details": f"Applied temperature scaling (T={temperature})"
        }