import logging
import numpy as np

from core.model_interface import ModelInterface
from core.monitoring import HealthMonitor
from utils.metrics import compute_performance_metrics
from core.diagnosis import DiagnosisEngine
from core.fixer import Fixer
from core.explainer import Explainer
from core.validator import Validator
from core.strategy_memory import StrategyMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Cognis:
    """
    Self-Healing AI orchestrator.

    Objectives addressed:
    1. Continuous Health Monitoring  — HealthMonitor (monitoring.py)
    2. Autonomous Root-Cause Diagnosis — DiagnosisEngine (diagnosis.py)
    3. Safe & Adaptive Self-Healing   — Fixer + Validator (fixer.py / validator.py)
    4. Experience-Based Learning      — StrategyMemory (strategy_memory.py)
                                        Fixer ranks strategies by win-rate;
                                        every outcome is recorded so future
                                        iterations prefer proven strategies.

    -----------------------------------------------------------------------
    FIX #1 (kept from previous round): best_accuracy / best_model_snapshot
    must be seeded from THIS RUN's data (X, y) — evaluated once before the
    loop — not from self.baseline_metrics["accuracy"], which is computed
    in __init__ against a different dataset (X_baseline / y_baseline).
    Those two numbers were never comparable; if the baseline dataset scored
    higher than anything achievable on the live data, "current_acc >
    best_accuracy" could fail on every single iteration, and the final
    restore() would silently hand back the original unhealed model even
    after healing clearly helped.

    FIX #2 (new): the result dict now carries explicit "initial_accuracy"
    and "final_accuracy" fields on every return path. Previously app.py had
    to reach into result["baseline_metrics"]["accuracy"] (wrong dataset,
    same bug as #1 but on the UI side) and
    result["history"][-1]["monitoring_after"]["current_metrics"]["accuracy"]
    for the "after" number — but the last history entry can be a step that
    got rolled back, so that number does not necessarily describe the
    model actually being returned as final_model. initial_accuracy and
    final_accuracy are now computed directly from the same values Cognis
    itself uses to decide promote/rollback/best-so-far, so they always
    describe the model that is actually handed back.

    FIX #3 (new): validation_output is now passed into explainer.generate().
    Previously the explainer had no access to the Validator's actual
    promote/rollback decision and had to *guess* "promoted" vs "rejected"
    purely by comparing before/after accuracy on its own. That guess could
    contradict the real decision (e.g. a strategy recorded as
    "applied_no_gain" but narrated as "promoted"). The explainer now uses
    the real decision as source of truth.
    """

    def __init__(self, model, X_baseline, y_baseline, thresholds,
                 api_key=None, max_iters=3, temperature=1.5):

        self.interface  = ModelInterface(model)
        self.thresholds = thresholds
        self.max_iters  = max_iters

        # Core components
        self.diagnoser = DiagnosisEngine()
        self.fixer     = Fixer(temperature=temperature)
        self.validator = Validator()
        self.explainer = Explainer(api_key=api_key)

        # Objective 4: experience-based learning
        self.memory = StrategyMemory()
        self.fixer.set_memory(self.memory)

        logger.info("Computing baseline metrics...")
        baseline_eval = self.interface.evaluate(X_baseline, y_baseline)

        self.baseline_metrics = compute_performance_metrics(
            baseline_eval["y_true"],
            baseline_eval["y_pred"],
            baseline_eval["probabilities"]
        )
        self.baseline_probs = baseline_eval["probabilities"]

        self.monitor = HealthMonitor(
            baseline_metrics=self.baseline_metrics,
            baseline_probs=self.baseline_probs,
            thresholds=self.thresholds,
            baseline_y=y_baseline
        )

        logger.info(f"Baseline accuracy: {round(self.baseline_metrics['accuracy'], 4)}")

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def start_diagnosis(self, X, y, model_name="Model"):

        history = []

        # See FIX #1 above. This is the live-data reference point —
        # everything in this run is compared against it, never against
        # self.baseline_metrics.
        initial_eval = self.interface.evaluate(X, y)
        initial_eval_accuracy = np.mean(initial_eval["y_pred"] == initial_eval["y_true"])
        best_accuracy = initial_eval_accuracy
        best_model_snapshot = self.validator.backup_model(self.interface)

        for step in range(self.max_iters):
            logger.info(f"=== Attempt {step + 1}/{self.max_iters} ===")

            # 1. Evaluate current state (reuse the eval above on step 0
            #    instead of scoring the model against X/y twice)
            evaluation    = initial_eval if step == 0 else self.interface.evaluate(X, y)
            y_true        = evaluation["y_true"]
            y_pred        = evaluation["y_pred"]
            probabilities = evaluation["probabilities"]

            # 2. Monitor — Objective 1
            monitoring_output = self.monitor.detect_degradation(y_true, y_pred, probabilities)

            # 3. Diagnose — Objective 2
            diagnosis_output = self.diagnoser.diagnose(monitoring_output)

            # Healthy — stop early
            if not monitoring_output["degraded"]:
                logger.info("Model is healthy. Stopping.")
                current_accuracy_this_eval = np.mean(y_pred == y_true)
                explanation = self.explainer.generate(
                    model_name, monitoring_output, diagnosis_output,
                    {"action": "none", "status": "skipped"},
                    monitoring_output, step=step,
                    validation_output=None
                )
                history.append({
                    "step":              step,
                    "monitoring_before": monitoring_output,
                    "monitoring_after":  monitoring_output,
                    "diagnosis":         diagnosis_output,
                    "healing":           None,
                    "validation":        None,
                    "explanation":       explanation
                })
                return {
                    "baseline_metrics":  self.baseline_metrics,
                    "history":           history,
                    "final_status":      "stable",
                    "improved":          False,
                    "final_model":       self.interface,
                    "strategy_memory":   self.memory.summary(),  # Obj 4
                    "initial_accuracy":  initial_eval_accuracy,
                    "final_accuracy":    current_accuracy_this_eval,
                }

            # Backup before fix
            backup_interface = self.validator.backup_model(self.interface)

            # 4. Apply fix — Objectives 3 & 4 (Fixer uses memory to rank)
            healing_output = self.fixer.apply_fix(self.interface, diagnosis_output, X, y)

            # 5. Evaluate after fix
            new_eval       = self.interface.evaluate(X, y)
            new_monitoring = self.monitor.detect_degradation(
                new_eval["y_true"], new_eval["y_pred"], new_eval["probabilities"]
            )

            # 6. Validate — Objective 3 (safe rollback)
            validation_output = self.validator.validate(
                monitoring_output, new_monitoring, healing_output=healing_output
            )

            # Track best model across all attempts
            current_acc = new_monitoring["current_metrics"]["accuracy"]
            if current_acc > best_accuracy:
                best_accuracy       = current_acc
                best_model_snapshot = self.validator.backup_model(self.interface)
                logger.info(f"New best accuracy: {round(best_accuracy, 4)}")

            # Rollback if this step made things worse
            if validation_output["decision"] == "rollback":
                self.validator.restore_model(self.interface, backup_interface)
                logger.warning("Step rolled back.")

            # 7. Explain — now passes the REAL validator decision instead of
            # letting the explainer infer promoted/rejected from raw numbers.
            explanation = self.explainer.generate(
                model_name, monitoring_output, diagnosis_output,
                healing_output, new_monitoring, step=step,
                validation_output=validation_output
            )

            history.append({
                "step":              step,
                "monitoring_before": monitoring_output,
                "monitoring_after":  new_monitoring,
                "diagnosis":         diagnosis_output,
                "healing":           healing_output,
                "validation":        validation_output,
                "explanation":       explanation
            })

            # Stop early if promoted and now healthy
            if validation_output["decision"] == "promote" and not new_monitoring["degraded"]:
                logger.info("Model stabilised.")
                return {
                    "baseline_metrics":  self.baseline_metrics,
                    "history":           history,
                    "final_status":      "stable",
                    "improved":          best_accuracy > initial_eval_accuracy,
                    "final_model":       best_model_snapshot,
                    "strategy_memory":   self.memory.summary(),  # Obj 4
                    "initial_accuracy":  initial_eval_accuracy,
                    "final_accuracy":    best_accuracy,
                }

        # Max iterations — restore best found
        logger.warning(f"Max iterations ({self.max_iters}) reached.")
        self.validator.restore_model(self.interface, best_model_snapshot)
        improved = best_accuracy > initial_eval_accuracy

        return {
            "baseline_metrics":  self.baseline_metrics,
            "history":           history,
            "final_status":      "max_iters_reached",
            "improved":          improved,
            "final_model":       best_model_snapshot,
            "strategy_memory":   self.memory.summary(),  # Obj 4
            "initial_accuracy":  initial_eval_accuracy,
            "final_accuracy":    best_accuracy,
        }
