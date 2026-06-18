import logging
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)


class Fixer:
    """
    Adaptive Self-Healing Engine.

    Selects and applies targeted repair strategies based on the diagnosed
    degradation type.  When a StrategyMemory instance is supplied (via
    `set_memory`), strategies are ranked by their historical win-rate
    before each attempt so the engine prefers what has worked before.

    After every attempt the outcome is recorded back into memory,
    enabling continuous experience-based learning across iterations.

    Changes from original:
    - `set_memory(memory)` wires in the StrategyMemory instance.
    - `apply_fix` ranks strategies via memory before trying them.
    - Every attempted strategy is recorded in memory with its improvement.
    - temperature is configurable (unchanged).
    - temperature_scaling fix stores scaled probs on interface (unchanged).
    """

    def __init__(self, temperature: float = 1.5):
        self.temperature = temperature
        self._memory = None  # injected by Cognis after construction

        self.strategy_map = {
            "concept_drift":    [self._fine_tune],
            "class_imbalance":  [self._reweight_classes],
            "label_noise":      [self._noise_weighting, self._reweight_classes],
            "calibration_error":[self._temperature_scaling]
        }

    # ------------------------------------------------------------------
    # MEMORY INJECTION
    # ------------------------------------------------------------------

    def set_memory(self, memory) -> None:
        """Wire in a StrategyMemory instance (called by Cognis)."""
        self._memory = memory
        logger.info("Fixer: StrategyMemory attached.")

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------

    def apply_fix(self, model_interface, diagnosis_output, X, y):
        issue = diagnosis_output.get("issue")

        if issue not in self.strategy_map:
            logger.warning(f"No fix strategy for issue: {issue}")
            return {
                "action": "none",
                "status": "skipped",
                "reason": "No applicable fix"
            }

        strategies = list(self.strategy_map[issue])

        # Rank by historical win-rate when memory is available
        if self._memory is not None:
            strategies = self._memory.rank_strategies(issue, strategies)

        baseline_preds, _ = model_interface.predict(X)
        baseline_acc = np.mean(baseline_preds == y)

        best_acc    = baseline_acc
        best_result = None
        best_model  = None

        for strategy in strategies:
            try:
                candidate = deepcopy(model_interface)
                result    = strategy(candidate, X, y)

                preds, _ = candidate.predict(X)
                acc      = np.mean(preds == y)
                improvement = acc - baseline_acc

                logger.info(
                    f"Strategy '{result['action']}' acc={round(acc, 4)} "
                    f"Δ={round(improvement, 4)}"
                )

                # Record outcome in memory regardless of whether it won
                if self._memory is not None:
                    self._memory.record(issue, strategy.__name__, improvement)

                if acc > best_acc:
                    best_acc    = acc
                    best_result = result
                    best_model  = candidate

            except Exception as e:
                logger.error(f"Strategy '{strategy.__name__}' failed: {e}")
                if self._memory is not None:
                    # Count a hard failure as zero improvement
                    self._memory.record(issue, strategy.__name__, 0.0)
                continue

        if best_model is not None:
            model_interface.model = best_model.model
            if hasattr(best_model, "_temperature_probs"):
                model_interface._temperature_probs = best_model._temperature_probs
                model_interface._temperature       = best_model._temperature

            logger.info(
                f"Applied fix: {best_result['action']} | "
                f"improvement: {round(best_acc - baseline_acc, 4)}"
            )
            return {
                "action":      best_result["action"],
                "status":      "applied",
                "details":     best_result.get("details", ""),
                "improvement": round(best_acc - baseline_acc, 4)
            }

        logger.warning("No strategy improved performance.")
        return {
            "action":            "none",
            "status":            "failed",
            "reason":            "No strategy improved performance",
            "baseline_accuracy": round(baseline_acc, 4)
        }

    # ------------------------------------------------------------------
    # STRATEGIES
    # ------------------------------------------------------------------

    def _fine_tune(self, interface, X, y):
        interface.train(X, y)
        return {
            "action":  "fine_tuning",
            "details": "Retrained model on available data"
        }

    def _reweight_classes(self, interface, X, y):
        y_int        = y.astype(int)
        class_counts = np.bincount(y_int)
        class_counts[class_counts == 0] = 1
        weights      = 1.0 / class_counts
        sample_weights = np.array([weights[int(label)] for label in y_int])
        interface.train(X, y, sample_weight=sample_weights)
        return {
            "action":  "class_reweighting",
            "details": "Applied inverse class weighting"
        }

    def _noise_weighting(self, interface, X, y):
        y_pred, probs = interface.predict(X)
        confidence    = probs.max(axis=1)
        noisy_mask    = (confidence > 0.8) & (y_pred != y)
        weights       = np.ones(len(y))
        weights[noisy_mask] = 0.3
        interface.train(X, y, sample_weight=weights)
        return {
            "action":  "noise_weighting",
            "details": f"Down-weighted {int(np.sum(noisy_mask))} noisy samples"
        }

    def _temperature_scaling(self, interface, X, y):
        _, probs     = interface.predict(X)
        scaled_probs = probs ** (1 / self.temperature)
        scaled_probs /= scaled_probs.sum(axis=1, keepdims=True)
        interface._temperature_probs = scaled_probs
        interface._temperature       = self.temperature
        return {
            "action":  "temperature_scaling",
            "details": f"Applied temperature scaling (T={self.temperature})"
        }
