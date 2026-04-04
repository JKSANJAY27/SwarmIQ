"""
SwarmIQ — Gemini LLM Client
Used for expensive, low-frequency global tasks only.
"""

import asyncio
import logging
from typing import Any

from ..config import Config

logger = logging.getLogger("swarmiq.llm.gemini")


class GeminiClient:
    """
    Wraps Gemini for expensive, low-frequency tasks:
      - Final report synthesis
      - Deep seed document analysis
      - Cross-scenario comparison
      - Complex event synthesis (god mode)

    Only instantiated if GEMINI_API_KEY is set.
    If key is absent, these tasks fall back to Ollama with a warning.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai

        self._api_key = api_key or Config.GEMINI_API_KEY
        self._model_name = model or Config.GEMINI_MODEL
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        self._client = genai.Client(api_key=self._api_key)
        logger.info("GeminiClient initialised with model %s", self._model_name)

    async def _generate(self, prompt: str) -> str:
        """Run a Gemini generation call in a thread (SDK is synchronous)."""
        def _call() -> str:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt
            )
            return response.text

        try:
            return await asyncio.to_thread(_call)
        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc)
            raise exc

    async def analyze_seed(self, document_text: str, prediction_goal: str) -> dict:
        """Deep analysis of seed document. Returns structured world context."""
        prompt = f"""You are an expert analyst. Deeply analyse the following document and extract structured world context for a social simulation.

Prediction goal: {prediction_goal}

Document (first 8000 chars):
{document_text[:8000]}

Return a JSON object with:
{{
  "summary": "3-5 sentence world description",
  "active_topics": ["topic1", "topic2", ...],  // 5-10 topics agents will discuss
  "initial_sentiments": {{"topic": 0.0}},       // -1.0 to 1.0
  "agent_archetypes": ["archetype1", ...],      // suggested agent types
  "key_entities": ["entity1", ...]              // important named entities
}}

Respond with JSON only."""
        import json, re
        raw = await self._generate(prompt)
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\n?```\s*$", "", raw).strip()
        try:
            return json.loads(raw)
        except Exception:
            return {"summary": document_text[:200], "active_topics": [], "initial_sentiments": {}, "agent_archetypes": [], "key_entities": []}

    async def synthesize_report(
        self, simulation_summary: dict, agent_samples: list, goal: str
    ) -> str:
        """Generate the final prediction report from simulation data."""
        import json
        prompt = f"""You are a strategic analysis expert. Generate a comprehensive prediction report based on simulation results.

Prediction goal: {goal}

Simulation summary:
{json.dumps(simulation_summary, indent=2)[:4000]}

Sample agent outcomes (representative):
{json.dumps(agent_samples[:10], indent=2)[:2000]}

Write a detailed prediction report that includes:
1. Executive Summary
2. Key Findings
3. Opinion Dynamics
4. Risk Factors & Tail Risks
5. Confidence Assessment
6. Strategic Recommendations

Use markdown formatting."""
        return await self._generate(prompt)

    async def synthesize_event(
        self, world_state: dict, event_description: str
    ) -> dict:
        """Turn a natural language event description into a structured simulation event."""
        import json, re
        prompt = f"""Convert this natural language event into a structured simulation event.

Current world state topics: {world_state.get('active_topics', [])}
Event description: {event_description}

Return JSON:
{{
  "event_name": "short name",
  "description": "detailed description",
  "severity": "low|medium|high|catastrophic",
  "affected_topics": ["topic1"],
  "opinion_shifts": {{"topic": delta}},
  "affected_agent_types": ["all" or specific types]
}}"""
        raw = await self._generate(prompt)
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\n?```\s*$", "", raw).strip()
        try:
            return json.loads(raw)
        except Exception:
            return {"event_name": event_description[:50], "description": event_description, "severity": "medium", "affected_topics": [], "opinion_shifts": {}, "affected_agent_types": ["all"]}

    async def compare_branches(self, branch_summaries: list[dict]) -> str:
        """Compare outcomes across branched simulations."""
        import json
        prompt = f"""Compare these parallel simulation branch outcomes and identify key divergences.

Branch summaries:
{json.dumps(branch_summaries, indent=2)[:5000]}

Provide a comparative analysis covering:
1. How outcomes diverged between branches
2. Key turning points
3. Which branch shows the most favorable outcome and why
4. Probability-weighted synthesis

Use markdown formatting."""
        return await self._generate(prompt)

    async def generic_chat(
        self, system: str = "", message: str = "", history: list | None = None
    ) -> str:
        """Generic Q&A chat — used for interactive report analysis."""
        history = history or []
        history_text = "\n".join(
            f"{turn.get('role','user').upper()}: {turn.get('content','')}"
            for turn in history[-6:]  # last 6 turns for context
        )
        prompt = f"{system}\n\n{history_text}\n\nUSER: {message}\n\nASSISTANT:"
        return await self._generate(prompt)

