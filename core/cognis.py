import logging
import numpy as np

from core.model_interface import ModelInterface
from core.monitoring import HealthMonitor
from utils.metrics import compute_performance_metrics
from core.diagnosis import DiagnosisEngine
from core.fixer import Fixer
from core.explainer import Explainer
from core.validator import Validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


class Cognis:

    def __init__(self, model, X_baseline, y_baseline, thresholds,
                 api_key=None, max_iters=10, temperature=1.5):

        self.interface = ModelInterface(model)
        self.thresholds = thresholds
        self.max_iters = max_iters  # max 10 attempts to fix

        self.diagnoser = DiagnosisEngine()
        self.fixer = Fixer(temperature=temperature)
        self.validator = Validator()
        self.explainer = Explainer(api_key=api_key)

        # === Baseline ===
        logger.info("Computing baseline metrics...")
        baseline_eval = self.interface.evaluate(X_baseline, y_baseline)

        self.baseline_metrics = compute_performance_metrics(
            baseline_eval["y_true"],
            baseline_eval["y_pred"],
            baseline_eval["probabilities"]
        )

        self.baseline_probs = baseline_eval["probabilities"]

        # FIX B: pass baseline_y so class distribution is real, not uniform
        self.monitor = HealthMonitor(
            baseline_metrics=self.baseline_metrics,
            baseline_probs=self.baseline_probs,
            thresholds=self.thresholds,
            baseline_y=y_baseline
        )

        logger.info(f"Baseline accuracy: {round(self.baseline_metrics['accuracy'], 4)}")

    def start_diagnosis(self, X, y, model_name="Model"):

        history = []
        best_accuracy = self.baseline_metrics["accuracy"]
        best_model_snapshot = self.validator.backup_model(self.interface)

        for step in range(self.max_iters):
            logger.info(f"=== Attempt {step + 1}/{self.max_iters} ===")

            # Step 1: Evaluate current model
            evaluation = self.interface.evaluate(X, y)
            y_true = evaluation["y_true"]
            y_pred = evaluation["y_pred"]
            probabilities = evaluation["probabilities"]

            # Step 2: Monitor
            monitoring_output = self.monitor.detect_degradation(
                y_true, y_pred, probabilities
            )

            # Step 3: Diagnose
            diagnosis_output = self.diagnoser.diagnose(monitoring_output)

            # Step 4: If healthy — stop
            if not monitoring_output["degraded"]:
                logger.info("Model is healthy. Stopping.")
                explanation = self.explainer.generate(
                    model_name, monitoring_output, diagnosis_output,
                    {"action": "none", "status": "skipped"},
                    monitoring_output, step=step
                )
                history.append({
                    "step": step,
                    "monitoring_before": monitoring_output,
                    "monitoring_after": monitoring_output,
                    "diagnosis": diagnosis_output,
                    "healing": None,
                    "validation": None,
                    "explanation": explanation
                })
                return {
                    "baseline_metrics": self.baseline_metrics,
                    "history": history,
                    "final_status": "stable",
                    "improved": False,
                    "final_model": self.interface
                }

            # Step 5: Backup before applying fix
            backup_interface = self.validator.backup_model(self.interface)

            # Step 6: Apply fix
            healing_output = self.fixer.apply_fix(
                self.interface, diagnosis_output, X, y
            )

            # Step 7: Evaluate after fix
            new_eval = self.interface.evaluate(X, y)
            new_monitoring = self.monitor.detect_degradation(
                new_eval["y_true"], new_eval["y_pred"], new_eval["probabilities"]
            )

            # Step 8: Validate — FIX C: pass healing_output so calibration uses ECE
            validation_output = self.validator.validate(
                monitoring_output, new_monitoring, healing_output=healing_output
            )

            # Step 9: Track best model seen so far
            current_acc = new_monitoring["current_metrics"]["accuracy"]
            if current_acc > best_accuracy:
                best_accuracy = current_acc
                best_model_snapshot = self.validator.backup_model(self.interface)
                logger.info(f"New best accuracy: {round(best_accuracy, 4)} — snapshot saved.")

            # Step 10: Rollback if this step made things worse
            if validation_output["decision"] == "rollback":
                self.validator.restore_model(self.interface, backup_interface)
                logger.warning("Step rolled back — restoring pre-fix state.")

            # Step 11: Generate explanation
            explanation = self.explainer.generate(
                model_name, monitoring_output, diagnosis_output,
                healing_output, new_monitoring, step=step
            )

            history.append({
                "step": step,
                "monitoring_before": monitoring_output,
                "monitoring_after": new_monitoring,
                "diagnosis": diagnosis_output,
                "healing": healing_output,
                "validation": validation_output,
                "explanation": explanation
            })

            # Step 12: If promoted AND now healthy — stop early
            if validation_output["decision"] == "promote" and not new_monitoring["degraded"]:
                logger.info("Model stabilised. Stopping early.")
                return {
                    "baseline_metrics": self.baseline_metrics,
                    "history": history,
                    "final_status": "stable",
                    "improved": best_accuracy > self.baseline_metrics["accuracy"],
                    "final_model": best_model_snapshot
                }

        # Max iterations reached — restore best model found across all attempts
        logger.warning(f"Max iterations ({self.max_iters}) reached.")
        self.validator.restore_model(self.interface, best_model_snapshot)

        improved = best_accuracy > self.baseline_metrics["accuracy"]
        logger.info(f"Best accuracy achieved: {round(best_accuracy, 4)} | Improved: {improved}")

        return {
            "baseline_metrics": self.baseline_metrics,
            "history": history,
            "final_status": "max_iters_reached",
            "improved": improved,
            "final_model": best_model_snapshot
        }
