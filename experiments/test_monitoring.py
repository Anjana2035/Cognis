import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utils.metrics import compute_performance_metrics
from core.monitoring import HealthMonitor
from utils.config import THRESHOLDS

X, y = load_digits(return_X_y=True)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

base_probs = model.predict_proba(X_val)
base_preds = np.argmax(base_probs, axis=1)

baseline_metrics = compute_performance_metrics(
    y_val, base_preds, base_probs
)

monitor = HealthMonitor(
    baseline_metrics=baseline_metrics,
    baseline_probs=base_probs,
    thresholds=THRESHOLDS
)

print("\n================ BASELINE METRICS ================")
print(baseline_metrics)

healthy_report = monitor.detect_degradation(
    y_true=y_val,
    y_pred=base_preds,
    probabilities=base_probs
)

print("\n================ HEALTHY CHECK ==================")
print(healthy_report)

np.random.seed(42)

X_noisy = X_val + np.random.normal(0, 3.0, X_val.shape)

noisy_probs = model.predict_proba(X_noisy)
noisy_preds = np.argmax(noisy_probs, axis=1)

degraded_report = monitor.detect_degradation(
    y_true=y_val,
    y_pred=noisy_preds,
    probabilities=noisy_probs
)

def pretty_report(report, title):
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)

    if not report["degraded"]:
        print("Status        : HEALTHY")
        print("System Verdict: Model behavior is stable. No action required.")
        print("Cognis Note   : I checked twice. Everything looks fine.")
    else:
        print("Status        : DEGRADED")
        print("System Verdict: Performance degradation detected.")
        print("Cognis Note   : Something changed. I wonder why? ")

        print("\nDetected Signals:")
        for k, v in report["alerts"].items():
            if isinstance(v, dict):
                print(f"  - {k}: statistically significant drift detected")
            else:
                print(f"  - {k}: deviation = {round(float(v), 4)}")

    print("\nSummary Metrics:")
    for k, v in report["current_metrics"].items():
        if isinstance(v, dict):
            continue
        print(f"  {k:<20}: {round(float(v), 4)}")

    print("=" * 70 + "\n")

pretty_report(
    healthy_report,
    title="COGNIS HEALTH CHECK REPORT (STABLE SCENARIO)"
)