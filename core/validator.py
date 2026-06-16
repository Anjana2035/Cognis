import copy
import logging

logger = logging.getLogger(__name__)


class Validator:
    """
    Validates model improvements and handles rollback/promotion.
    """

    def __init__(self, min_improvement=0.01):
        self.min_improvement = min_improvement

    def validate(self, before_monitoring, after_monitoring):
        """
        Compare metrics and decide whether to promote or rollback.
        """

        before_acc = before_monitoring["current_metrics"]["accuracy"]
        after_acc = after_monitoring["current_metrics"]["accuracy"]

        improvement = after_acc - before_acc

        if improvement > self.min_improvement:
            decision = "promote"
            reason = "Significant improvement"
        elif improvement > 0:
            decision = "promote"
            reason = "Minor improvement"
        else:
            decision = "rollback"
            reason = "No improvement or degradation"

        logger.info(f"Validation: {decision} | improvement={round(improvement, 4)} | reason={reason}")

        return {
            "decision": decision,
            "improvement": round(improvement, 4),
            "before_accuracy": round(before_acc, 4),
            "after_accuracy": round(after_acc, 4),
            "reason": reason
        }

    def backup_model(self, model_interface):
        """
        Create a deep copy of current model state.
        """
        logger.info("Backing up model state.")
        return copy.deepcopy(model_interface)

    def restore_model(self, model_interface, backup_interface):
        """
        Restore previous model state.
        """
        logger.info("Rolling back to previous model state.")
        model_interface.model = backup_interface.model
        model_interface._temperature_probs = backup_interface._temperature_probs
        model_interface._temperature = backup_interface._temperature
