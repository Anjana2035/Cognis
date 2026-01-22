import numpy as np
from sklearn.metrics import accuracy_score, log_loss

def compute_entropy(probabilities):
    probs = np.clip(probabilities, 1e-9, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return np.mean(entropy)


def compute_confidence_stats(probabilities):
    max_conf = probabilities.max(axis=1)
    return {
        "mean_confidence": float(np.mean(max_conf)),
        "std_confidence": float(np.std(max_conf))
    }


def compute_classwise_accuracy(y_true, y_pred):
    class_acc = {}
    for c in np.unique(y_true):
        idx = y_true == c
        class_acc[int(c)] = float(accuracy_score(y_true[idx], y_pred[idx]))
    return class_acc


def compute_performance_metrics(y_true, y_pred, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "entropy": compute_entropy(probabilities),
        **compute_confidence_stats(probabilities),
        "classwise_accuracy": compute_classwise_accuracy(y_true, y_pred)
    }
