import logging
import numpy as np
from utils.metrics import compute_performance_metrics
from utils.drift import confidence_drift
from utils.config import THRESHOLDS

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Signal-based monitoring system.
    Computes signals → aggregates → decides degradation.

    FIX B: class_imbalance now uses actual baseline class distribution,
            not a fake uniform distribution.
    """

    def __init__(self, baseline_metrics, baseline_probs, thresholds=THRESHOLDS,
                 baseline_y=None):
        self.baseline_metrics = baseline_metrics
        self.baseline_probs = baseline_probs
        self.thresholds = thresholds

        # FIX B: store actual baseline class distribution
        if baseline_y is not None:
            counts = np.bincount(baseline_y.astype(int))
            self.baseline_class_dist = counts / counts.sum()
        else:
            # fall back to deriving from classwise_accuracy keys (equal weight)
            n = len(baseline_metrics.get("classwise_accuracy", {1: 1}))
            self.baseline_class_dist = np.ones(n) / n

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
        """
        FIX B: Compare current prediction distribution against ACTUAL
        baseline class distribution, not a uniform assumption.
        """
        current_dist = np.bincount(y_pred.astype(int)) / len(y_pred)
        baseline_dist = self.baseline_class_dist

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
        ratio = float(np.mean(high_conf_wrong))
        return {
            "name": "label_noise",
            "value": ratio,
            "triggered": ratio > 0.1,
            "weight": 2
        }

    def _calibration_signal(self, y_true, y_pred, probabilities):
        """
        FIX C (ChatGPT B): calibration uses ECE (Expected Calibration Error),
        not a simple confidence-correctness gap, so it's meaningful independent
        of accuracy. This means calibration_error triggers correctly even when
        accuracy is unchanged after temperature scaling.
        """
        confidence = probabilities.max(axis=1)
        correctness = (y_pred == y_true).astype(float)

        # Bin into 10 buckets and compute weighted ECE
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
            if mask.sum() > 0:
                bin_conf = confidence[mask].mean()
                bin_acc = correctness[mask].mean()
                ece += (mask.sum() / len(y_true)) * abs(bin_conf - bin_acc)

        return {
            "name": "calibration_error",
            "value": float(ece),
            "triggered": ece > 0.1,   # ECE > 10% is meaningful miscalibration
            "weight": 1
        }

    # =========================
    # MAIN MONITORING FUNCTION
    # =========================

    def detect_degradation(self, y_true, y_pred, probabilities):
        current_metrics = compute_performance_metrics(
            y_true, y_pred, probabilities
        )

        signals = [
            self._accuracy_drop_signal(current_metrics),
            self._entropy_shift_signal(current_metrics),
            self._confidence_drift_signal(probabilities),
            self._class_imbalance_signal(y_pred),
            self._label_noise_signal(y_true, y_pred, probabilities),
            self._calibration_signal(y_true, y_pred, probabilities)
        ]

        triggered_signals = [s for s in signals if s["triggered"]]
        severity_score = sum(s["weight"] for s in triggered_signals)

        accuracy = current_metrics["accuracy"]
        drop = self.baseline_metrics["accuracy"] - accuracy

        degraded = (
            severity_score >= 3 or
            drop > self.thresholds["accuracy_drop"]
        )

        logger.info(
            f"Monitoring: accuracy={round(accuracy, 4)}, "
            f"drop={round(drop, 4)}, severity={severity_score}, degraded={degraded}"
        )

        return {
            "current_metrics": current_metrics,
            "signals": signals,
            "triggered_signals": triggered_signals,
            "severity_score": severity_score,
            "degraded": degraded
        }
