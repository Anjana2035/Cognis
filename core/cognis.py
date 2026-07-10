import logging
import os
import numpy as np

from core.model_interface import ModelInterface
from core.monitoring import HealthMonitor
from utils.metrics import compute_performance_metrics
from core.diagnosis import DiagnosisEngine
from core.fixer import Fixer
from core.explainer import Explainer
from core.validator import Validator
from core.strategy_memory import StrategyMemory

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cognis.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class Cognis:

    def __init__(self, model, X_baseline, y_baseline, thresholds,
                 api_key=None, max_iters=3, temperature=1.5, memory=None):

        self.interface  = ModelInterface(model)  # raises on non-classifiers
        self.thresholds = thresholds
        self.max_iters  = max_iters

        self.diagnoser = DiagnosisEngine()
        self.fixer     = Fixer(temperature=temperature)
        self.validator = Validator()
        self.explainer = Explainer(api_key=api_key)

        self.memory = memory if memory is not None else StrategyMemory()
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


    def start_diagnosis(self, X, y, model_name="Model"):

        history = []
        initial_eval = self.interface.evaluate(X, y)
        initial_eval_accuracy = np.mean(initial_eval["y_pred"] == initial_eval["y_true"])
        best_accuracy = initial_eval_accuracy
        best_model_snapshot = self.validator.backup_model(self.interface)

        for step in range(self.max_iters):
            logger.info(f"=== Attempt {step + 1}/{self.max_iters} ===")

            evaluation    = initial_eval if step == 0 else self.interface.evaluate(X, y)
            y_true        = evaluation["y_true"]
            y_pred        = evaluation["y_pred"]
            probabilities = evaluation["probabilities"]

            monitoring_output = self.monitor.detect_degradation(y_true, y_pred, probabilities)
            diagnosis_output = self.diagnoser.diagnose(monitoring_output)

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
                    "strategy_memory":   self.memory.summary(),
                    "initial_accuracy":  initial_eval_accuracy,
                    "final_accuracy":    current_accuracy_this_eval,
                }

            backup_interface = self.validator.backup_model(self.interface)

            healing_output = self.fixer.apply_fix(self.interface, diagnosis_output, X, y)

            new_eval       = self.interface.evaluate(X, y)
            new_monitoring = self.monitor.detect_degradation(
                new_eval["y_true"], new_eval["y_pred"], new_eval["probabilities"]
            )

            validation_output = self.validator.validate(
                monitoring_output, new_monitoring, healing_output=healing_output
            )

            strategy_name = healing_output.get("strategy_name")
            if self.memory is not None and strategy_name:
                self.memory.record(
                    diagnosis_output["issue"],
                    strategy_name,
                    validation_output["improvement"]
                )

            current_acc = new_monitoring["current_metrics"]["accuracy"]
            if current_acc > best_accuracy:
                best_accuracy       = current_acc
                best_model_snapshot = self.validator.backup_model(self.interface)
                logger.info(f"New best accuracy: {round(best_accuracy, 4)}")

            if validation_output["decision"] == "rollback":
                self.validator.restore_model(self.interface, backup_interface)
                logger.warning("Step rolled back.")

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

            if validation_output["decision"] == "promote" and not new_monitoring["degraded"]:
                logger.info("Model stabilised.")
                return {
                    "baseline_metrics":  self.baseline_metrics,
                    "history":           history,
                    "final_status":      "stable",
                    "improved":          best_accuracy > initial_eval_accuracy,
                    "final_model":       best_model_snapshot,
                    "strategy_memory":   self.memory.summary(),
                    "initial_accuracy":  initial_eval_accuracy,
                    "final_accuracy":    best_accuracy,
                }

        logger.warning(f"Max iterations ({self.max_iters}) reached.")
        self.validator.restore_model(self.interface, best_model_snapshot)
        improved = best_accuracy > initial_eval_accuracy

        return {
            "baseline_metrics":  self.baseline_metrics,
            "history":           history,
            "final_status":      "max_iters_reached",
            "improved":          improved,
            "final_model":       best_model_snapshot,
            "strategy_memory":   self.memory.summary(),
            "initial_accuracy":  initial_eval_accuracy,
            "final_accuracy":    best_accuracy,
        }
