import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class StrategyMemory:
    
    SUCCESS_THRESHOLD = 0.005  

    def __init__(self):
        self._stats: dict[str, dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"wins": 0, "total": 0})
        )
        self._history: list[dict] = []


    def record(self, issue: str, strategy: str, improvement: float) -> None:
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


    def win_rate(self, issue: str, strategy: str) -> float:
        stats = self._stats[issue][strategy]
        if stats["total"] == 0:
            return 0.5  # neutral prior
        return stats["wins"] / stats["total"]

    def rank_strategies(self, issue: str, strategies: list) -> list:
        ranked = sorted(
            strategies,
            key=lambda fn: self.win_rate(issue, fn.__name__),
            reverse=True
        )
        names = [f.__name__ for f in ranked]
        logger.info(f"StrategyMemory: ranked strategies for '{issue}': {names}")
        return ranked


    def summary(self) -> dict:
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

    @classmethod
    def from_json(cls, json_str: str) -> "StrategyMemory":
       
        mem = cls()
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning("StrategyMemory.from_json: invalid JSON, starting fresh.")
            return mem

        for issue, strategies in data.get("stats", {}).items():
            for strategy, counts in strategies.items():
                mem._stats[issue][strategy]["wins"] = int(counts.get("wins", 0))
                mem._stats[issue][strategy]["total"] = int(counts.get("total", 0))
        mem._history = data.get("history", [])
        return mem
