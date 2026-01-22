import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils.metrics import compute_performance_metrics
from utils.drift import confidence_drift
from utils.config import THRESHOLDS


class HealthMonitor:
    def __init__(self, baseline_metrics, baseline_probs, thresholds):
        self.baseline_metrics = baseline_metrics
        self.baseline_probs = baseline_probs
        self.thresholds = thresholds

    def detect_degradation(self, y_true, y_pred, probabilities):
        """
        Detects performance degradation using multi-signal analysis.
        """
        current_metrics = compute_performance_metrics(
            y_true, y_pred, probabilities
        )

        alerts = {}

        acc_drop = (
            self.baseline_metrics["accuracy"] -
            current_metrics["accuracy"]
        )
        if acc_drop > self.thresholds["accuracy_drop"]:
            alerts["accuracy_drop"] = acc_drop

        entropy_shift = (
            current_metrics["entropy"] -
            self.baseline_metrics["entropy"]
        )
        if entropy_shift > self.thresholds["entropy_increase"]:
            alerts["entropy_shift"] = entropy_shift

        conf_drift = confidence_drift(
            self.baseline_probs,
            probabilities,
            alpha=self.thresholds["confidence_alpha"]
        )
        if conf_drift["drift_detected"]:
            alerts["confidence_drift"] = conf_drift

        degraded_classes = []
        for cls, acc in current_metrics["classwise_accuracy"].items():
            base_acc = self.baseline_metrics["classwise_accuracy"].get(cls, 1.0)
            if base_acc - acc > self.thresholds["class_accuracy_drop"]:
                degraded_classes.append(cls)

        if degraded_classes:
            alerts["classwise_degradation"] = degraded_classes

        return {
            "current_metrics": current_metrics,
            "alerts": alerts,
            "degraded": len(alerts) >= self.thresholds["min_alerts"]
        }
