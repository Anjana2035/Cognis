import numpy as np
from utils.metrics import compute_performance_metrics
from utils.drift import confidence_drift
from utils.config import THRESHOLDS


class HealthMonitor:
    """
    Signal-based monitoring system.
    Computes signals → aggregates → decides degradation.
    """

    def __init__(self, baseline_metrics, baseline_probs, thresholds=THRESHOLDS):
        self.baseline_metrics = baseline_metrics
        self.baseline_probs = baseline_probs
        self.thresholds = thresholds

    # =========================
    # SIGNAL DEFINITIONS
    # =========================

    def _accuracy_drop_signal(self, current_metrics):
        drop = self.baseline_metrics["accuracy"] - current_metrics["accuracy"]

        return {
            "name": "accuracy_drop",
            "value": float(drop),
            "triggered": drop > self.thresholds["accuracy_drop"],
            "weight": 2
        }

    def _entropy_shift_signal(self, current_metrics):
        shift = current_metrics["entropy"] - self.baseline_metrics["entropy"]

        return {
            "name": "entropy_shift",
            "value": float(shift),
            "triggered": shift > self.thresholds["entropy_increase"],
            "weight": 1
        }

    def _confidence_drift_signal(self, probabilities):
        drift = confidence_drift(
            self.baseline_probs,
            probabilities,
            alpha=self.thresholds["confidence_alpha"]
        )

        return {
            "name": "confidence_drift",
            "value": drift,
            "triggered": drift["drift_detected"],
            "weight": 2
        }

    def _class_imbalance_signal(self, y_pred):
        current_dist = np.bincount(y_pred) / len(y_pred)

        baseline_classes = list(self.baseline_metrics["classwise_accuracy"].keys())
        baseline_dist = np.ones(len(baseline_classes)) / len(baseline_classes)

        # Align lengths
        min_len = min(len(current_dist), len(baseline_dist))
        shift = np.abs(current_dist[:min_len] - baseline_dist[:min_len]).mean()

        return {
            "name": "class_imbalance",
            "value": float(shift),
            "triggered": shift > 0.1,
            "weight": 2
        }

    def _label_noise_signal(self, y_true, y_pred, probabilities):
        confidence = probabilities.max(axis=1)
        wrong = (y_pred != y_true)

        high_conf_wrong = (confidence > 0.8) & wrong
        ratio = np.mean(high_conf_wrong)

        return {
            "name": "label_noise",
            "value": float(ratio),
            "triggered": ratio > 0.1,
            "weight": 2
        }

    def _calibration_signal(self, y_true, y_pred, probabilities):
        confidence = probabilities.max(axis=1)
        correctness = (y_pred == y_true).astype(float)

        gap = np.abs(confidence - correctness).mean()

        return {
            "name": "calibration_error",
            "value": float(gap),
            "triggered": gap > 0.2,
            "weight": 1
        }

    # =========================
    # MAIN MONITORING FUNCTION
    # =========================

    def detect_degradation(self, y_true, y_pred, probabilities):
        """
        Runs full signal-based monitoring.
        """

        current_metrics = compute_performance_metrics(
            y_true, y_pred, probabilities
        )

        # === Generate Signals ===
        signals = [
            self._accuracy_drop_signal(current_metrics),
            self._entropy_shift_signal(current_metrics),
            self._confidence_drift_signal(probabilities),
            self._class_imbalance_signal(y_pred),
            self._label_noise_signal(y_true, y_pred, probabilities),
            self._calibration_signal(y_true, y_pred, probabilities)
        ]

        triggered_signals = [s for s in signals if s["triggered"]]

        # === Severity Score ===
        severity_score = sum(s["weight"] for s in triggered_signals)

        # 🔥 smarter degradation logic
        accuracy = current_metrics["accuracy"]
        drop = self.baseline_metrics["accuracy"] - accuracy

        # 🔥 Allow degradation from signals OR accuracy drop

        degraded = (
            severity_score >= 3 or
            drop > self.thresholds["accuracy_drop"]
        ) # simple threshold

        return {
            "current_metrics": current_metrics,
            "signals": signals,
            "triggered_signals": triggered_signals,
            "severity_score": severity_score,
            "degraded": degraded
        }