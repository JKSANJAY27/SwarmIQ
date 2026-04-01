"""
SwarmIQ — World Builder
Takes a NetworkX graph (from GraphRAG) and generates the initial World Context.
"""

import json
import logging
from typing import Any

import networkx as nx

from ..llm.llm_router import LLMRouter
from ..simulation.prompts import WORLD_CONTEXT_PROMPT

logger = logging.getLogger("swarmiq.graphrag.world")

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
        
        sys = "You are a world-building simulation architect."
        usr = WORLD_CONTEXT_PROMPT.format(entities=entities_text, relationships=rels_text, goal=goal)
        
        logger.info("Building world context for goal: %s", goal)
        context = await self.llm.call("seed_analyze", document_text=graph.graph.get("summary", ""), prediction_goal=goal)
        
        # Merge if missing
        if "active_topics" not in context:
            context = await self.llm.ollama.complete_json(system=sys, user=usr)
            
        logger.info("World context generated. %d active topics found.", len(context.get("active_topics", [])))
        return context
