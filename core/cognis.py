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
        self.max_iters = max_iters

        self.diagnoser = DiagnosisEngine()
        self.fixer = Fixer(temperature=temperature)  # FIX: configurable temperature
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

        self.monitor = HealthMonitor(
            baseline_metrics=self.baseline_metrics,
            baseline_probs=self.baseline_probs,
            thresholds=self.thresholds
        )

        logger.info(f"Baseline accuracy: {round(self.baseline_metrics['accuracy'], 4)}")

    def start_diagnosis(self, X, y, X_baseline=None, y_baseline=None, model_name="Model"):
        """
        FIX: Accepts optional separate baseline data (X_baseline, y_baseline).
        If not provided, falls back to using X, y (original behavior).
        """

        history = []

        for step in range(self.max_iters):
            logger.info(f"=== Diagnosis Step {step + 1} ===")

            # === Step 1: Evaluate ===
            evaluation = self.interface.evaluate(X, y)
            y_true = evaluation["y_true"]
            y_pred = evaluation["y_pred"]
            probabilities = evaluation["probabilities"]

            # === Step 2: Monitor ===
            monitoring_output = self.monitor.detect_degradation(
                y_true, y_pred, probabilities
            )

            # === Step 3: Diagnosis ===
            diagnosis_output = self.diagnoser.diagnose(monitoring_output)

            # === If Already Healthy → STOP ===
            if not monitoring_output["degraded"]:
                logger.info("Model is healthy. Stopping diagnosis.")
                explanation = self.explainer.generate(
                    model_name,
                    monitoring_output,
                    diagnosis_output,
                    {"action": "none", "status": "skipped"},
                    monitoring_output
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
                    "final_status": "stable"
                }

            # === Step 4: Backup + Apply Fix ===
            backup_interface = self.validator.backup_model(self.interface)

            healing_output = self.fixer.apply_fix(
                self.interface,
                diagnosis_output,
                X,
                y
            )

            # === Step 5: Evaluate AGAIN (AFTER FIX) ===
            new_eval = self.interface.evaluate(X, y)

            new_monitoring = self.monitor.detect_degradation(
                new_eval["y_true"],
                new_eval["y_pred"],
                new_eval["probabilities"]
            )

            # === Step 6: Validate Improvement ===
            validation_output = self.validator.validate(
                monitoring_output,
                new_monitoring
            )

            # === Step 7: Rollback if needed ===
            if validation_output["decision"] == "rollback":
                self.validator.restore_model(self.interface, backup_interface)
                logger.warning("Fix rolled back — no improvement detected.")

            # === Step 8: Generate Explanation ===
            explanation = self.explainer.generate(
                model_name,
                monitoring_output,
                diagnosis_output,
                healing_output,
                new_monitoring
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

            # === Step 9: STOP if validated improvement ===
            if validation_output["decision"] == "promote":
                logger.info("Fix promoted. System stabilized.")
                return {
                    "baseline_metrics": self.baseline_metrics,
                    "history": history,
                    "final_status": "stable"
                }

        logger.warning("Max iterations reached without full stabilization.")
        return {
            "baseline_metrics": self.baseline_metrics,
            "history": history,
            "final_status": "max_iters_reached"
        }
