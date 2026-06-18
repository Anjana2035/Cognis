import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class StrategyMemory:
    """
    Experience-based learning layer for Cognis.

    Tracks the historical effectiveness of each healing strategy per
    diagnosed issue type. On each healing attempt the Fixer consults
    this memory to order strategies by their win-rate, so the system
    prefers what has worked before and avoids repeating failures.

    Record format (per entry in self._history):
        {
            "issue":       str,   # e.g. "concept_drift"
            "strategy":    str,   # e.g. "fine_tuning"
            "improvement": float, # accuracy delta (or ECE delta)
            "success":     bool   # True if improvement > threshold
        }

    Win-rate is defined as:
        success_count / total_attempts  (for that issue × strategy pair)

    Strategies with no history default to a neutral score of 0.5 so
    they are tried before known-bad ones but after proven ones.
    """

    SUCCESS_THRESHOLD = 0.005  # minimum improvement to count as a win

    def __init__(self):
        # { issue -> { strategy -> {"wins": int, "total": int} } }
        self._stats: dict[str, dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"wins": 0, "total": 0})
        )
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # WRITE
    # ------------------------------------------------------------------

    def record(self, issue: str, strategy: str, improvement: float) -> None:
        """
        Called after each fix attempt with the measured improvement.

        Args:
            issue:       Diagnosed issue key (e.g. "concept_drift").
            strategy:    Strategy function name used (e.g. "fine_tuning").
            improvement: Numeric delta (positive = better).
        """
        success = improvement > self.SUCCESS_THRESHOLD
        self._stats[issue][strategy]["total"] += 1
        if success:
            self._stats[issue][strategy]["wins"] += 1

        entry = {
            "issue": issue,
            "strategy": strategy,
            "improvement": round(improvement, 6),
            "success": success
        }
        self._history.append(entry)
        logger.info(
            f"StrategyMemory: recorded {strategy} for {issue} | "
            f"improvement={round(improvement, 6)} success={success}"
        )

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def win_rate(self, issue: str, strategy: str) -> float:
        """
        Returns the historical win-rate for this issue × strategy pair.
        Falls back to 0.5 (neutral) when no history exists.
        """
        stats = self._stats[issue][strategy]
        if stats["total"] == 0:
            return 0.5  # neutral prior
        return stats["wins"] / stats["total"]

    def rank_strategies(self, issue: str, strategies: list) -> list:
        """
        Sorts a list of strategy callables by descending win-rate for
        the given issue.  Strategies with equal win-rates preserve their
        original order (stable sort).

        Args:
            issue:      Diagnosed issue key.
            strategies: List of strategy callables (from Fixer).

        Returns:
            The same list, reordered highest-win-rate first.
        """
        ranked = sorted(
            strategies,
            key=lambda fn: self.win_rate(issue, fn.__name__),
            reverse=True
        )

        names = [f.__name__ for f in ranked]
        logger.info(f"StrategyMemory: ranked strategies for '{issue}': {names}")
        return ranked

    # ------------------------------------------------------------------
    # INTROSPECTION
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Returns a human-readable dict summarising win-rates by issue.
        Exposed in Cognis results for UI transparency.
        """
        out = {}
        for issue, strategies in self._stats.items():
            out[issue] = {}
            for strategy, counts in strategies.items():
                total = counts["total"]
                wins  = counts["wins"]
                rate  = wins / total if total > 0 else None
                out[issue][strategy] = {
                    "attempts": total,
                    "wins":     wins,
                    "win_rate": round(rate, 3) if rate is not None else "no data"
                }
        return out

    def history(self) -> list:
        return list(self._history)

    def to_json(self) -> str:
        """Serialise memory for persistence / display."""
        return json.dumps({
            "stats":   {k: dict(v) for k, v in self._stats.items()},
            "history": self._history
        }, indent=2)
