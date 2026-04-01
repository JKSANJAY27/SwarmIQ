"""
SwarmIQ — Confidence Engine
Runs multiple branches of a simulation to determine prediction variance.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .engine import SimulationEngine

logger = logging.getLogger("swarmiq.simulation.confidence")


@dataclass
class EnsembleResult:
    """Result of running multiple simulation branches."""
    goal: str
    num_branches: int
    mean_opinions: dict[str, float]
    variance: dict[str, float]
    confidence_score: float  # [0.0, 1.0]

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "num_branches": self.num_branches,
            "mean_opinions": self.mean_opinions,
            "variance": self.variance,
            "confidence_score": round(self.confidence_score, 4),
        }


class ConfidenceEngine:
    """
    Runs an ensemble of simulations to generate confidence bounds for a prediction.
    """

    def __init__(self, base_engine_factory):
        """factory() should return a fresh SimulationEngine."""
        self.factory = base_engine_factory

    async def evaluate(self, goal: str, branches: int = 3, ticks: int = 50) -> EnsembleResult:
        """
        Run N identical simulations to measure outcome variance.
        """
        logger.info("Evaluating confidence with %d branches for %d ticks", branches, ticks)
        
        # Create N parallel engines
        engines = [self.factory() for _ in range(branches)]
        
        # Run them concurrently
        await asyncio.gather(*[e.run(ticks) for e in engines])
        
        # Aggregate final states
        topics = engines[0].state.active_topics
        
        means = {}
        variances = {}
        total_variance = 0.0
        
        from numpy import var, mean
        
        for topic in topics:
            # 1 float per branch representing the final mean opinion of that branch
            branch_outcomes = [e.state.opinion_summary().get(topic, 0.0) for e in engines]
            
            m = float(mean(branch_outcomes))
            v = float(var(branch_outcomes))
            
            means[topic] = round(m, 3)
            variances[topic] = round(v, 4)
            total_variance += v

        # Confidence is inverse to variance (heuristic)
        avg_var = total_variance / len(topics) if topics else 0.0
        # If variance is 0, confidence is 1. If variance is 0.5 (very high for [-1,1]), confidence is 0.
        confidence = max(0.0, 1.0 - (avg_var * 2.0))
        
        return EnsembleResult(
            goal=goal,
            num_branches=branches,
            mean_opinions=means,
            variance=variances,
            confidence_score=confidence,
        )
