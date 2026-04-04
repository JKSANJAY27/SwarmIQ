"""
SwarmIQ — World Builder
Takes a NetworkX graph (from GraphRAG) and generates the initial World Context.
Falls back to heuristic generation when all LLMs are unavailable.
"""

import json
import logging
import random
from typing import Any

import networkx as nx

from ..llm.llm_router import LLMRouter
from ..simulation.prompts import WORLD_CONTEXT_PROMPT

logger = logging.getLogger("swarmiq.graphrag.world")


def _heuristic_world_context(graph: nx.DiGraph, goal: str) -> dict:
    """Build a minimal world context from graph structure when no LLM is available."""
    # Top entities by degree become active topics
    top_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)[:8]
    topic_candidates = [n[0] for n in top_nodes if n[0]]

    # Fill up to 6 topics; add generic fallbacks if graph is sparse
    generic_topics = [
        "Public Opinion", "Policy Change", "Economic Impact",
        "Social Cohesion", "Media Narrative", "Community Response"
    ]
    active_topics = topic_candidates[:6]
    while len(active_topics) < 4:
        t = generic_topics.pop(0)
        if t not in active_topics:
            active_topics.append(t)

    initial_sentiments = {t: round(random.gauss(0.0, 0.2), 3) for t in active_topics}

    summary_text = graph.graph.get("summary", "") or goal or "A social simulation world."
    summary = (
        f"This world is shaped by tensions around: {', '.join(active_topics[:3])}. "
        f"The scenario under analysis is: {goal or 'emerging societal dynamics'}. "
        f"Agents hold diverse opinions and are influenced by events and peer interactions."
    )

    archetypes = [
        "General Public", "Policy Maker", "Activist",
        "Journalist", "Business Person", "Academic Researcher"
    ]

    key_entities = [n[0] for n in top_nodes[:10]]

    return {
        "summary": summary,
        "active_topics": active_topics,
        "initial_sentiments": initial_sentiments,
        "agent_archetypes": archetypes,
        "key_entities": key_entities,
    }


class WorldBuilder:
    def __init__(self, llm: LLMRouter):
        self.llm = llm

    async def build_context(self, graph: nx.DiGraph, goal: str) -> dict:
        """Analyze the knowledge graph and prediction goal to set up the world."""
        # Get top entities by degree
        top_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)[:15]
        entities_text = ", ".join([f"{n[0]} ({graph.nodes[n[0]].get('type', 'Unknown')})" for n in top_nodes])

        # Get sample relationships
        rels = list(graph.edges(data=True))[:10]
        rels_text = "; ".join([f"{u} -> {v} ({d.get('relation', '')})" for u, v, d in rels])

        logger.info("Building world context for goal: %s", goal)

        context: dict = {}

        # --- Attempt LLM-based seed analysis (Gemini first, then Ollama fallback) ---
        try:
            context = await self.llm.call(
                "seed_analyze",
                document_text=graph.graph.get("summary", ""),
                prediction_goal=goal
            )
        except Exception as exc:
            logger.warning("seed_analyze LLM call failed: %s", exc)
            context = {}

        # If LLM produced no active_topics, try Ollama prompt directly
        if not context.get("active_topics"):
            sys = "You are a world-building simulation architect."
            usr = WORLD_CONTEXT_PROMPT.format(
                entities=entities_text, relationships=rels_text, goal=goal
            )
            try:
                ollama_ctx = await self.llm.ollama.complete_json(system=sys, user=usr)
                if ollama_ctx.get("active_topics"):
                    context = ollama_ctx
            except Exception as exc2:
                logger.warning("Ollama world context also failed: %s", exc2)

        # Final fallback: heuristic from graph structure
        if not context.get("active_topics"):
            logger.warning("Both LLMs unavailable — using heuristic world context generator.")
            context = _heuristic_world_context(graph, goal)

        logger.info("World context ready. %d active topics.", len(context.get("active_topics", [])))
        return context
