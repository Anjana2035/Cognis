import logging
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class Fixer:
    """
    Adaptive Self-Healing Engine.

    Selects and applies targeted repair strategies based on the diagnosed
    degradation type.  When a StrategyMemory instance is supplied (via
    `set_memory`), strategies are ranked by their historical win-rate
    before each attempt so the engine prefers what has worked before.

    -----------------------------------------------------------------------
    FIX — memory recording moved out of candidate selection:
    -----------------------------------------------------------------------
    apply_fix() internally splits (X, y) into a 75/25 X_train/X_val purely
    to decide WHICH candidate strategy looks best before committing to one.
    Previously, every candidate's X_val-based improvement was written to
    StrategyMemory as it was tried — but that's a different measurement
    than the one that actually decides promote/rollback: Cognis's
    Validator.validate() compares full-dataset accuracy (or ECE, for
    calibration fixes) before vs. after the chosen fix is applied to the
    live model. A strategy could easily look like a loss on the internal
    25% shard while being the one that visibly improved the full dataset
    and got promoted — which is exactly what produced the "0/1 wins" /
    "0%" strategy memory panel for a strategy that had just been promoted
    in the chat log above it.

    apply_fix() no longer calls memory.record() for the internal candidate
    comparison. Instead it returns "strategy_name" (the __name__ of
    whichever strategy was actually applied to the live model) inside the
    result dict. Cognis now calls memory.record() exactly once per healing
    attempt, AFTER Validator.validate() has run, using the real
    full-dataset outcome. This makes "attempts" mean "times this strategy
    was actually used on the live model" and "wins" mean "times that use
    was actually promoted" — matching what the UI shows elsewhere.

    Exceptions raised by a candidate strategy during selection are still
    recorded immediately (as a 0.0-improvement loss) — that's independent,
    genuinely useful signal ("this strategy is broken") that doesn't
    conflict with the promote/rollback contradiction above, since a
    crashed candidate is never a candidate for promotion anyway.
    """

    def __init__(self, temperature: float = 1.5):
        self.temperature = temperature
        self._memory = None  # injected by Cognis after construction

        # Each issue now has multiple ordered strategies.
        # concept_drift gets three distinct approaches so that successive
        # iterations each try something meaningfully different.
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
                "reason": "No applicable fix",
                "strategy_name": None,
            }

        strategies = list(self.strategy_map[issue])

        # Rank by historical win-rate when memory is available
        if self._memory is not None:
            strategies = self._memory.rank_strategies(issue, strategies)

        # Split into train / held-out val to prevent overfitting.
        # We train on X_train, but measure quality on X_val only.
        # If too few samples to split, fall back to full set (small datasets).
        #
        # NOTE: this split is used ONLY to pick which candidate strategy to
        # apply. It is a different, smaller measurement than the full-dataset
        # before/after comparison Cognis's Validator uses to actually decide
        # promote/rollback — so its numbers are no longer written to
        # StrategyMemory (see class docstring). It remains a reasonable
        # heuristic for candidate selection itself.
        if len(X) >= 20:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
                if len(np.unique(y)) > 1 else None
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
                # Train on X_train only
                result = strategy(candidate, X_train, y_train)

                # Evaluate on held-out X_val — used only to pick a candidate
                preds, _ = candidate.predict(X_val)
                acc = np.mean(preds == y_val)
                improvement = acc - baseline_acc

                logger.info(
                    f"Strategy '{result['action']}' acc={round(acc, 4)} "
                    f"Δ={round(improvement, 4)} (internal selection split)"
                )

                # KEY FIX: always track the best candidate, even if it
                # didn't beat baseline.  We want to apply *something*.
                if acc > best_acc:
                    best_acc = acc
                    best_result = result
                    best_model = candidate
                    best_strategy_fn = strategy

            except Exception as e:
                logger.error(f"Strategy '{strategy.__name__}' failed: {e}")
                # Genuinely independent signal: this strategy is broken,
                # regardless of which candidate ends up chosen/applied.
                if self._memory is not None:
                    self._memory.record(issue, strategy.__name__, 0.0)
                continue

        if best_model is not None:
            # Always write back the best candidate found
            model_interface.model = best_model.model
            if hasattr(best_model, "_temperature_probs"):
                model_interface._temperature_probs = best_model._temperature_probs
                model_interface._temperature = best_model._temperature

            net_improvement = round(best_acc - baseline_acc, 4)
            status = "applied" if net_improvement > 0 else "applied_no_gain"

            logger.info(
                f"Applied fix: {best_result['action']} | "
                f"selection-split improvement: {net_improvement}"
            )
            return {
                "action": best_result["action"],
                "status": status,
                "details": best_result.get("details", ""),
                "improvement": net_improvement,
                # Real name Cognis needs to record the REAL outcome against,
                # once Validator has scored the full dataset.
                "strategy_name": best_strategy_fn.__name__,
            }

        logger.warning("All strategies raised exceptions — nothing applied.")
        return {
            "action": "none",
            "status": "failed",
            "reason": "All strategies raised exceptions",
            "baseline_accuracy": round(baseline_acc, 4),
            "strategy_name": None,
        }

    # ------------------------------------------------------------------
    # STRATEGIES
    # ------------------------------------------------------------------

    def _fine_tune(self, interface, X, y):
        """
        Retrain with stronger regularization if the underlying model supports it.
        For sklearn estimators this means tightening C (logistic/SVM) or
        reducing max_depth / n_estimators (tree models) to reduce overfitting.
        Falls back to plain refit for other model types.
        """
        model = interface.model
        regularized = False

        try:
            import sklearn.base as skbase
            if skbase.is_classifier(model) or skbase.is_regressor(model):
                params = model.get_params()

                # Logistic Regression / LinearSVC — lower C = more regularization
                if "C" in params:
                    new_C = max(params["C"] * 0.5, 1e-4)
                    model.set_params(C=new_C)
                    logger.info(f"fine_tune: tightened C → {new_C:.5f}")
                    regularized = True

                # Decision Tree / Random Forest — cap depth and estimators
                if "max_depth" in params and params["max_depth"] is not None:
                    new_depth = max(params["max_depth"] - 1, 1)
                    model.set_params(max_depth=new_depth)
                    logger.info(f"fine_tune: reduced max_depth → {new_depth}")
                    regularized = True

                if "n_estimators" in params:
                    new_n = max(int(params["n_estimators"] * 0.8), 10)
                    model.set_params(n_estimators=new_n)
                    logger.info(f"fine_tune: reduced n_estimators → {new_n}")
                    regularized = True

                # SVM with kernel — gamma regularization
                if "gamma" in params and isinstance(params["gamma"], float):
                    new_gamma = params["gamma"] * 0.5
                    model.set_params(gamma=new_gamma)
                    logger.info(f"fine_tune: reduced gamma → {new_gamma:.5f}")
                    regularized = True

        except Exception as e:
            logger.warning(f"fine_tune regularization probe failed: {e}")

        interface.train(X, y)
        detail = "Retrained with tightened regularization" if regularized else "Retrained on available data"
        return {
            "action": "fine_tuning",
            "details": detail,
        }

    def _reweight_classes(self, interface, X, y):
        y_int = y.astype(int)
        class_counts = np.bincount(y_int)
        class_counts[class_counts == 0] = 1
        weights = 1.0 / class_counts
        sample_weights = np.array([weights[int(label)] for label in y_int])
        interface.train(X, y, sample_weight=sample_weights)
        return {
            "action": "class_reweighting",
            "details": "Applied inverse class weighting",
        }

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
