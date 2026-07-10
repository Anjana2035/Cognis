import logging

logger = logging.getLogger(__name__)


class DiagnosisEngine:

    def __init__(self):
        self.issue_rules = {
            "concept_drift": {
                "signals": {
                    "confidence_drift": 2,
                    "entropy_shift":    1,
                    "accuracy_drop":    2
                },
                "penalise_if": {"class_imbalance", "label_noise", "calibration_error"},
                "description": "Data distribution has shifted, causing model confusion."
            },
            "class_imbalance": {
                "signals": {
                    "class_imbalance": 4,   
                    "accuracy_drop":   1
                },
                "penalise_if": set(),
                "description": "Class distribution has changed, affecting performance."
            },
            "label_noise": {
                "signals": {
                    "label_noise": 4       
                },
                "penalise_if": set(),
                "description": "High-confidence incorrect predictions indicate noisy labels."
            },
            "calibration_error": {
                "signals": {
                    "calibration_error": 3  
                },
                "penalise_if": set(),
                "description": "Model confidence does not match prediction accuracy."
            }
        }

    def diagnose(self, monitoring_output):
       
        if not monitoring_output.get("degraded", False):
            logger.info("Model is stable. No degradation detected.")
            return {
                "issue":       "no_issue",
                "confidence":  1.0,
                "reason":      "Model is stable. No degradation detected.",
                "signals_used": []
            }

        triggered = monitoring_output.get("triggered_signals", [])

        if not triggered:
            logger.warning("Degradation detected but no clear signal pattern found.")
            return {
                "issue":       "unknown",
                "confidence":  0.0,
                "reason":      "Degradation detected but no clear signal pattern found.",
                "signals_used": []
            }

        signal_map     = {s["name"]: s for s in triggered}
        triggered_names = set(signal_map.keys())
        issue_scores   = {}

        for issue, rule in self.issue_rules.items():
            score       = 0
            used_signals = []

            for signal_name, weight in rule["signals"].items():
                if signal_name in signal_map:
                    score += weight
                    used_signals.append(signal_name)

            if score > 0:
                penalise = rule.get("penalise_if", set())
                if penalise & triggered_names:
                    score = score * 0.5
                    logger.info(
                        f"DiagnosisEngine: penalising '{issue}' score "
                        f"(specific signal present)"
                    )

                issue_scores[issue] = {
                    "score":        score,
                    "signals_used": used_signals
                }

        if not issue_scores:
            logger.warning("Signals detected but do not match known patterns.")
            return {
                "issue":       "unknown",
                "confidence":  0.0,
                "reason":      "Signals detected but do not match known patterns.",
                "signals_used": []
            }

        best_issue   = max(issue_scores, key=lambda x: issue_scores[x]["score"])
        best_score   = issue_scores[best_issue]["score"]
        signals_used = issue_scores[best_issue]["signals_used"]

        max_possible_score = sum(self.issue_rules[best_issue]["signals"].values())
        confidence = min(best_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0

        logger.info(f"Diagnosed issue: {best_issue} (confidence={round(confidence, 3)})")

        return {
            "issue":        best_issue,
            "confidence":   round(confidence, 3),
            "reason":       self.issue_rules[best_issue]["description"],
            "signals_used": signals_used
        }
