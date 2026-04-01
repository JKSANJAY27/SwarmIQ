"""
SwarmIQ — Ollama LLM Client
Handles all per-agent cognitive calls using local models.
"""

import asyncio
import json
import logging
import re
from typing import AsyncGenerator

import ollama

from ..config import Config

logger = logging.getLogger("swarmiq.llm.ollama")


class OllamaClient:
    """
    Wraps Ollama for agent cognition calls.
    Default model: llama3.2:3b — fast, low memory, good for structured agent responses.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
    ):
        self.model = model or Config.OLLAMA_AGENT_MODEL
        self.embed_model = embed_model or Config.OLLAMA_EMBED_MODEL
        self.client = ollama.AsyncClient(host=base_url or Config.OLLAMA_BASE_URL)

    async def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
    ) -> str:
        """Single completion. Used for agent cognitive steps."""
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": temperature},
            )
            content: str = response["message"]["content"]
            # Strip any <think>...</think> tags from reasoning models
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            return content
        except Exception as exc:
            logger.warning("Ollama complete failed: %s", exc)
            return ""

    async def complete_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
    ) -> dict:
        """
        Completion that enforces JSON output.
        Appends JSON instruction to system prompt and parses result.
        Falls back gracefully on parse failure.
        """
        json_system = (
            system
            + "\n\nIMPORTANT: Respond with valid JSON only. No markdown fences, no extra text."
        )
        raw = await self.complete(json_system, user, temperature=temperature)
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract first JSON object
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning("JSON parse failed for Ollama response, returning empty dict")
            return {}

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using nomic-embed-text via Ollama."""
        try:
            response = await self.client.embeddings(
                model=self.embed_model,
                prompt=text,
            )
            return response["embedding"]
        except Exception as exc:
            logger.warning("Ollama embed failed: %s", exc)
            return []

    async def batch_complete(
        self,
        prompts: list[dict],
        max_concurrent: int | None = None,
    ) -> list[str]:
        """
        Run multiple completions concurrently with a semaphore.
        Used for ticking all agents simultaneously.
        Critical for simulation performance.
        """
        limit = max_concurrent or Config.SIM_PARALLEL_WORKERS
        sem = asyncio.Semaphore(limit)

        async def _single(p: dict) -> str:
            async with sem:
                return await self.complete(p["system"], p["user"])

        return list(await asyncio.gather(*[_single(p) for p in prompts]))
