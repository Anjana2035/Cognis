import numpy as np

from core.model_interface import ModelInterface
from core.monitoring import HealthMonitor
from utils.metrics import compute_performance_metrics
from core.diagnosis import DiagnosisEngine
from core.fixer import Fixer
from core.explainer import Explainer
from core.validator import Validator


class Cognis:

    def __init__(self, model, X_baseline, y_baseline, thresholds,
                 api_key=None, max_iters=10):

        self.interface = ModelInterface(model)
        self.thresholds = thresholds
        self.max_iters = max_iters

        self.diagnoser = DiagnosisEngine()
        self.fixer = Fixer()
        self.validator = Validator()

        # ✅ Updated Explainer (no api_url anymore)
        self.explainer = Explainer(api_key=api_key)

        # === Baseline ===
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

    def start_diagnosis(self, X, y, model_name="Model"):

        history = []

        for step in range(self.max_iters):

            # === Step 1: Evaluate (BEFORE FIX) ===
            evaluation = self.interface.evaluate(X, y)

            y_true = evaluation["y_true"]
            y_pred = evaluation["y_pred"]
            probabilities = evaluation["probabilities"]

            # === Step 2: Monitor (BEFORE FIX) ===
            monitoring_output = self.monitor.detect_degradation(
                y_true,
                y_pred,
                probabilities
            )

            # === Step 3: Diagnosis ===
            diagnosis_output = self.diagnoser.diagnose(monitoring_output)

            # === If Already Healthy → STOP ===
            if not monitoring_output["degraded"]:
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
                return {
                    "baseline_metrics": self.baseline_metrics,
                    "history": history,
                    "final_status": "stable"
                }

        # === Max Iterations Reached ===
        return {
            "baseline_metrics": self.baseline_metrics,
            "history": history,
            "final_status": "max_iters_reached"
        }