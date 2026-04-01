"""
SwarmIQ — LLM Router
Routes tasks to Ollama (per-agent, high-frequency) or Gemini (global, low-frequency).
"""

import logging
from typing import Any

from .ollama_client import OllamaClient
from .gemini_client import GeminiClient

logger = logging.getLogger("swarmiq.llm.router")

# Tasks handled by Ollama (high-frequency, per-agent)
OLLAMA_TASKS = {
    "agent_react",
    "agent_interact",
    "agent_update_belief",
    "agent_generate",
    "entity_extract",
    "sentiment_classify",
}

# Tasks handled by Gemini (low-frequency, once per simulation)
GEMINI_TASKS = {
    "seed_analyze",
    "report_synthesize",
    "event_synthesize",
    "branch_compare",
}


class LLMRouter:
    """
    Routes LLM calls to the right model based on task type.

    Gemini tasks fall back to Ollama if GEMINI_API_KEY is not set.
    """

    def __init__(self, ollama: OllamaClient, gemini: GeminiClient | None):
        self.ollama = ollama
        self.gemini = gemini

    async def call(self, task: str, **kwargs: Any) -> Any:
        """Dispatch a task to the appropriate LLM backend."""
        if task in GEMINI_TASKS:
            if self.gemini:
                try:
                    method_name = task  # e.g. "seed_analyze" -> gemini.analyze_seed (mapped below)
                    return await self._dispatch_gemini(task, **kwargs)
                except Exception as exc:
                    logger.warning(
                        "Gemini task '%s' failed (%s), falling back to Ollama", task, exc
                    )
            else:
                logger.warning(
                    "Gemini not configured, falling back to Ollama for task '%s'", task
                )
            return await self._dispatch_ollama_fallback(task, **kwargs)

        if task in OLLAMA_TASKS:
            return await self._dispatch_ollama(task, **kwargs)

        raise ValueError(f"Unknown task: {task}")

    async def _dispatch_gemini(self, task: str, **kwargs: Any) -> Any:
        mapping = {
            "seed_analyze": self.gemini.analyze_seed,
            "report_synthesize": self.gemini.synthesize_report,
            "event_synthesize": self.gemini.synthesize_event,
            "branch_compare": self.gemini.compare_branches,
        }
        fn = mapping.get(task)
        if fn is None:
            raise ValueError(f"No Gemini handler for task: {task}")
        return await fn(**kwargs)

    async def _dispatch_ollama(self, task: str, **kwargs: Any) -> Any:
        """Route Ollama tasks — all use complete_json with appropriate prompts."""
        system = kwargs.get("system", "You are a helpful assistant. Respond with JSON only.")
        user = kwargs.get("user", "")
        return await self.ollama.complete_json(system=system, user=user)

    async def _dispatch_ollama_fallback(self, task: str, **kwargs: Any) -> Any:
        """Fallback for Gemini tasks if Gemini is unavailable."""
        if task == "seed_analyze":
            doc = kwargs.get("document_text", "")[:3000]
            goal = kwargs.get("prediction_goal", "")
            system = "You are a world-building analyst. Return JSON only."
            user = f"Analyse this text for a simulation about: {goal}\n\nText:\n{doc}\n\nReturn JSON with keys: summary, active_topics (list), initial_sentiments (dict), agent_archetypes (list), key_entities (list)."
            return await self.ollama.complete_json(system=system, user=user)

        if task == "report_synthesize":
            summary = kwargs.get("simulation_summary", {})
            goal = kwargs.get("goal", "")
            system = "You are a strategic analyst. Write a prediction report."
            user = f"Goal: {goal}\nSummary: {str(summary)[:2000]}\n\nWrite a prediction report."
            return await self.ollama.complete(system=system, user=user)

        if task == "event_synthesize":
            desc = kwargs.get("event_description", "")
            system = "You are a simulation event designer. Return JSON only."
            user = f"Convert this event to JSON: {desc}\n\nReturn: {{event_name, description, severity, affected_topics, opinion_shifts, affected_agent_types}}"
            return await self.ollama.complete_json(system=system, user=user)

        if task == "branch_compare":
            summaries = kwargs.get("branch_summaries", [])
            system = "You are a comparative analyst."
            user = f"Compare these simulation branches and summarise differences:\n{str(summaries)[:3000]}"
            return await self.ollama.complete(system=system, user=user)

        logger.error("No fallback handler for task: %s", task)
        return {}
