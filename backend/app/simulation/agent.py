"""
SwarmIQ — Agent and Personality definitions.
BigFivePersonality (OCEAN model) + Agent dataclass.
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BigFivePersonality:
    """
    OCEAN model personality traits — each a float in [0.0, 1.0].

    openness:          curiosity, creativity, willingness to consider new ideas
    conscientiousness: reliability, self-discipline, goal-directedness
    extraversion:      sociability, assertiveness, talkativeness
    agreeableness:     cooperativeness, empathy, conflict-avoidance
    neuroticism:       emotional instability, anxiety, stress reactivity
    """

    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def __post_init__(self) -> None:
        # Clamp all traits to [0.0, 1.0]
        for attr in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))

    @classmethod
    def random(cls, seed: int | None = None) -> "BigFivePersonality":
        """Generate random personality with Gaussian distribution centred at 0.5."""
        rng = random.Random(seed)
        return cls(
            openness=max(0.0, min(1.0, rng.gauss(0.5, 0.18))),
            conscientiousness=max(0.0, min(1.0, rng.gauss(0.5, 0.18))),
            extraversion=max(0.0, min(1.0, rng.gauss(0.5, 0.18))),
            agreeableness=max(0.0, min(1.0, rng.gauss(0.5, 0.18))),
            neuroticism=max(0.0, min(1.0, rng.gauss(0.5, 0.18))),
        )

    @classmethod
    def for_occupation(cls, occupation: str) -> "BigFivePersonality":
        """Bias personality slightly based on occupation archetype."""
        base = cls.random()
        occ_lower = occupation.lower()
        if any(k in occ_lower for k in ("journalist", "artist", "writer", "researcher")):
            base.openness = min(1.0, base.openness + 0.15)
        if any(k in occ_lower for k in ("bureaucrat", "accountant", "manager", "officer")):
            base.conscientiousness = min(1.0, base.conscientiousness + 0.15)
        if any(k in occ_lower for k in ("salesperson", "politician", "activist", "teacher")):
            base.extraversion = min(1.0, base.extraversion + 0.15)
        if any(k in occ_lower for k in ("doctor", "nurse", "social worker", "counselor")):
            base.agreeableness = min(1.0, base.agreeableness + 0.15)
        return base

    def to_prompt_description(self) -> str:
        """Convert OCEAN traits into natural language for the agent system prompt."""
        def _level(v: float) -> str:
            if v >= 0.75:
                return "very high"
            if v >= 0.55:
                return "moderately high"
            if v >= 0.45:
                return "average"
            if v >= 0.25:
                return "moderately low"
            return "very low"

        parts = [
            f"openness to new ideas ({_level(self.openness)})",
            f"conscientiousness ({_level(self.conscientiousness)})",
            f"extraversion ({_level(self.extraversion)})",
            f"agreeableness ({_level(self.agreeableness)})",
            f"tendency toward anxiety/neuroticism ({_level(self.neuroticism)})",
        ]
        return f"Your personality: {'; '.join(parts)}."

    def to_dict(self) -> dict:
        return {
            "openness": round(self.openness, 3),
            "conscientiousness": round(self.conscientiousness, 3),
            "extraversion": round(self.extraversion, 3),
            "agreeableness": round(self.agreeableness, 3),
            "neuroticism": round(self.neuroticism, 3),
        }


@dataclass
class Agent:
    """A simulation participant with personality, opinions, and memory."""

    id: str
    name: str
    age: int
    occupation: str
    background: str                        # 2-3 sentence backstory
    personality: BigFivePersonality

    # Current state
    opinions: dict[str, float]             # topic -> sentiment [-1.0, 1.0]
    influence_score: float = 0.5           # how much others listen to this agent
    trust_network: dict[str, float] = field(default_factory=dict)  # agent_id -> trust [0,1]

    # Tracking
    tick_born: int = 0
    interactions_this_tick: list[str] = field(default_factory=list)
    opinion_history: list[dict] = field(default_factory=list)    # [{tick, opinions}]

    def opinion_vector(self) -> list[float]:
        """Return opinions as a flat vector for clustering/drift analysis."""
        return list(self.opinions.values())

    def system_prompt(self) -> str:
        """
        Build the agent's system prompt from their profile.
        Passed to Ollama on every cognitive call.
        """
        opinions_text = "; ".join(
            f"{topic}: {'positive' if score > 0.2 else 'negative' if score < -0.2 else 'neutral'} ({score:.2f})"
            for topic, score in list(self.opinions.items())[:8]
        )
        return (
            f"You are {self.name}, a {self.age}-year-old {self.occupation}.\n"
            f"Background: {self.background}\n"
            f"{self.personality.to_prompt_description()}\n"
            f"Your current opinions on active topics: {opinions_text}\n\n"
            "You respond as this real person would, not as an AI. "
            "Be authentic to your background and personality. "
            "Always respond with valid JSON as instructed."
        )

    def record_opinion_snapshot(self, tick: int) -> None:
        """Save current opinions to history for drift analysis."""
        self.opinion_history.append({"tick": tick, "opinions": dict(self.opinions)})

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "occupation": self.occupation,
            "background": self.background,
            "personality": self.personality.to_dict(),
            "opinions": self.opinions,
            "influence_score": round(self.influence_score, 4),
            "trust_network": self.trust_network,
            "tick_born": self.tick_born,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        personality = BigFivePersonality(**data["personality"])
        return cls(
            id=data["id"],
            name=data["name"],
            age=data["age"],
            occupation=data["occupation"],
            background=data["background"],
            personality=personality,
            opinions=data["opinions"],
            influence_score=data.get("influence_score", 0.5),
            trust_network=data.get("trust_network", {}),
            tick_born=data.get("tick_born", 0),
        )
