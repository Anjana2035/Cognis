import copy
import logging

logger = logging.getLogger(__name__)


class Validator:
    """
    Validates model improvements and handles rollback/promotion.

    FIX C: calibration_error fix (temperature scaling) is validated using
    calibration_error signal improvement, NOT accuracy. Accuracy won't change
    after temperature scaling — that's expected and correct.
    """

    def __init__(self, min_improvement=0.01):
        self.min_improvement = min_improvement

    def validate(self, before_monitoring, after_monitoring, healing_output=None):
        before_metrics = before_monitoring["current_metrics"]
        after_metrics = after_monitoring["current_metrics"]

        # FIX C: if the fix was temperature_scaling, validate on calibration
        # improvement (ECE reduction), not accuracy
        fix_action = (healing_output or {}).get("action", "")

        if fix_action == "temperature_scaling":
            return self._validate_calibration(before_monitoring, after_monitoring)

        # Default: validate on accuracy
        before_acc = before_metrics["accuracy"]
        after_acc = after_metrics["accuracy"]
        improvement = after_acc - before_acc

        if improvement > self.min_improvement:
            decision, reason = "promote", "Significant accuracy improvement"
        elif improvement > 0:
            decision, reason = "promote", "Minor accuracy improvement"
        else:
            decision, reason = "rollback", "No accuracy improvement or degradation"

        logger.info(f"Validation (accuracy): {decision} | Δ={round(improvement, 4)}")

        return {
            "decision": decision,
            "improvement": round(improvement, 4),
            "metric": "accuracy",
            "before": round(before_acc, 4),
            "after": round(after_acc, 4),
            "reason": reason
        }

    def _validate_calibration(self, before_monitoring, after_monitoring):
        """
        FIX C: For temperature scaling, promote if ECE went DOWN (calibration improved).
        ECE is in the calibration_error signal.
        """
        def get_ece(monitoring):
            for s in monitoring.get("signals", []):
                if s["name"] == "calibration_error":
                    return s["value"]
            return None

        before_ece = get_ece(before_monitoring)
        after_ece = get_ece(after_monitoring)

        if before_ece is None or after_ece is None:
            # Can't compare — fall back to accuracy
            logger.warning("ECE not found in signals — falling back to accuracy validation.")
            before_acc = before_monitoring["current_metrics"]["accuracy"]
            after_acc = after_monitoring["current_metrics"]["accuracy"]
            improvement = after_acc - before_acc
            decision = "promote" if improvement >= 0 else "rollback"
            return {
                "decision": decision,
                "improvement": round(improvement, 4),
                "metric": "accuracy_fallback",
                "before": round(before_acc, 4),
                "after": round(after_acc, 4),
                "reason": "ECE unavailable — used accuracy"
            }

        ece_improvement = before_ece - after_ece  # positive = ECE went down = better

        if ece_improvement > 0.01:
            decision, reason = "promote", "Calibration improved (ECE reduced)"
        elif ece_improvement >= 0:
            decision, reason = "promote", "Minor calibration improvement"
        else:
            decision, reason = "rollback", "Calibration worsened after scaling"

        logger.info(f"Validation (ECE): {decision} | ΔECE={round(ece_improvement, 4)}")

        return {
            "decision": decision,
            "improvement": round(ece_improvement, 4),
            "metric": "ece",
            "before": round(before_ece, 4),
            "after": round(after_ece, 4),
            "reason": reason
        }

    def backup_model(self, model_interface):
        logger.info("Backing up model state.")
        return copy.deepcopy(model_interface)

    def restore_model(self, model_interface, backup_interface):
        logger.info("Rolling back to previous model state.")
        model_interface.model = backup_interface.model
        model_interface._temperature_probs = backup_interface._temperature_probs
        model_interface._temperature = backup_interface._temperature
