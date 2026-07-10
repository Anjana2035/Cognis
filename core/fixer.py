import logging
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class Fixer:

    def __init__(self, temperature: float = 1.5):
        self.temperature = temperature
        self._memory = None  

        self.strategy_map = {
            "concept_drift": [
                self._fine_tune,
                self._reweight_classes,
                self._noise_weighting,
            ],
            "class_imbalance": [
                self._reweight_classes,
                self._fine_tune,
                self._noise_weighting,
            ],
            "label_noise": [
                self._noise_weighting,
                self._reweight_classes,
                self._fine_tune,
            ],
            "calibration_error": [
                self._temperature_scaling,
                self._fine_tune,
            ],
        }

    def set_memory(self, memory) -> None:
        """Wire in a StrategyMemory instance (called by Cognis)."""
        self._memory = memory
        logger.info("Fixer: StrategyMemory attached.")

    def apply_fix(self, model_interface, diagnosis_output, X, y):
        issue = diagnosis_output.get("issue")

        if issue not in self.strategy_map:
            logger.warning(f"No fix strategy for issue: {issue}")
            return {
                "action": "none",
                "status": "skipped",
                "reason": "No applicable fix",
            }

        strategies = list(self.strategy_map[issue])

        if self._memory is not None:
            strategies = self._memory.rank_strategies(issue, strategies)

        if len(X) >= 20:
            try:
                strat = y if len(np.unique(y)) > 1 and np.min(
                    np.bincount(y.astype(int))
                ) >= 2 else None
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=strat
                )
            except ValueError as e:
                logger.warning(f"Stratified split failed ({e}); retrying without stratify.")
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
            logger.warning("Dataset too small to split — evaluating on full set (overfitting risk).")

        baseline_preds, _ = model_interface.predict(X_val)
        baseline_acc = np.mean(baseline_preds == y_val)
        best_acc = -np.inf
        best_result = None
        best_model = None
        best_strategy_fn = None

        for strategy in strategies:
            try:
                candidate = deepcopy(model_interface)
                result = strategy(candidate, X_train, y_train)

                preds, _ = candidate.predict(X_val)
                acc = np.mean(preds == y_val)
                improvement = acc - baseline_acc

                logger.info(
                    f"Strategy '{result['action']}' acc={round(acc, 4)} "
                    f"Δ={round(improvement, 4)}"
                )

                if acc > best_acc:
                    best_acc = acc
                    best_result = result
                    best_model = candidate
                    best_strategy_fn = strategy

            except Exception as e:
                logger.error(f"Strategy '{strategy.__name__}' failed: {e}")
                continue

        if best_model is not None:
            model_interface.model = best_model.model
            if hasattr(best_model, "_temperature_probs"):
                model_interface._temperature_probs = best_model._temperature_probs
                model_interface._temperature = best_model._temperature

            net_improvement = round(best_acc - baseline_acc, 4)
            status = "applied" if net_improvement > 0 else "applied_no_gain"

            logger.info(
                f"Applied fix: {best_result['action']} | "
                f"improvement: {net_improvement}"
            )
            return {
                "action": best_result["action"],
                "status": status,
                "details": best_result.get("details", ""),
                "improvement": net_improvement,
                "strategy_name": best_strategy_fn.__name__,  
            }

        logger.warning("All strategies raised exceptions — nothing applied.")
        return {
            "action": "none",
            "status": "failed",
            "reason": "All strategies raised exceptions",
            "baseline_accuracy": round(baseline_acc, 4),
        }

    def _fine_tune(self, interface, X, y):
        model = interface.model
        regularized = False
        try:
            import sklearn.base as skbase
            if skbase.is_classifier(model) or skbase.is_regressor(model):
                params = model.get_params()

                if "C" in params:
                    new_C = max(params["C"] * 0.5, 1e-4)
                    model.set_params(C=new_C)
                    regularized = True

                if "max_depth" in params and params["max_depth"] is not None:
                    new_depth = max(params["max_depth"] - 1, 1)
                    model.set_params(max_depth=new_depth)
                    regularized = True

                if "n_estimators" in params:
                    new_n = max(int(params["n_estimators"] * 0.8), 10)
                    model.set_params(n_estimators=new_n)
                    regularized = True

                if "gamma" in params and isinstance(params["gamma"], float):
                    new_gamma = params["gamma"] * 0.5
                    model.set_params(gamma=new_gamma)
                    regularized = True

        except Exception as e:
            logger.warning(f"fine_tune regularization probe failed: {e}")

        interface.train(X, y)
        detail = "Retrained with tightened regularization" if regularized else "Retrained on available data"
        return {"action": "fine_tuning", "details": detail}

    def _reweight_classes(self, interface, X, y):
        y_int = y.astype(int)
        class_counts = np.bincount(y_int)
        class_counts[class_counts == 0] = 1
        weights = 1.0 / class_counts
        sample_weights = np.array([weights[int(label)] for label in y_int])
        interface.train(X, y, sample_weight=sample_weights)
        return {"action": "class_reweighting", "details": "Applied inverse class weighting"}

    def _noise_weighting(self, interface, X, y):
        y_pred, probs = interface.predict(X)
        confidence = probs.max(axis=1)
        noisy_mask = (confidence > 0.8) & (y_pred != y)
        weights = np.ones(len(y))
        weights[noisy_mask] = 0.3
        interface.train(X, y, sample_weight=weights)
        return {
            "action": "noise_weighting",
            "details": f"Down-weighted {int(np.sum(noisy_mask))} noisy samples",
        }

    def _temperature_scaling(self, interface, X, y):
        _, probs = interface.predict(X)
        scaled_probs = probs ** (1 / self.temperature)
        scaled_probs /= scaled_probs.sum(axis=1, keepdims=True)
        interface._temperature_probs = scaled_probs
        interface._temperature = self.temperature
        return {
            "action": "temperature_scaling",
            "details": f"Applied temperature scaling (T={self.temperature})",
        }
