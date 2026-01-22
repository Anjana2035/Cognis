import numpy as np
from scipy.stats import ks_2samp

def ks_drift_test(reference, current, alpha=0.05):
    statistic, p_value = ks_2samp(reference, current)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": p_value < alpha
    }


def confidence_drift(ref_probs, cur_probs, alpha=0.05):
    ref_conf = ref_probs.max(axis=1)
    cur_conf = cur_probs.max(axis=1)
    return ks_drift_test(ref_conf, cur_conf, alpha)
