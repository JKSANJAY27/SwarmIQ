"""
SwarmIQ — Local GraphRAG implementation.
Uses Ollama to extract entities and build a local NetworkX graph.
Replaces Zep Standalone Graph.
"""

import json
import logging
from typing import Any

import networkx as nx

from ..llm.ollama_client import OllamaClient
from ..simulation.prompts import ENTITY_EXTRACT_PROMPT

logger = logging.getLogger("swarmiq.graphrag")


class EntityExtractor:
    """
    Extracts entities and relationships from text using Ollama.
    Builds a complete NetworkX graph.
    """

    def __init__(self, ollama: OllamaClient):
        self.ollama = ollama

    async def extract_from_text(self, text: str, chunk_size: int = 4000) -> nx.DiGraph:
        """Split text, extract entities concurrently, and merge into graph."""
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        system = "You are a precise data extraction algorithm. Always return valid JSON only."
        prompts = [
            {"system": system, "user": ENTITY_EXTRACT_PROMPT.format(text_chunk=chunk)}
            for chunk in chunks
        ]

        logger.info("Extracting graph from %d text chunks", len(chunks))
        results = await self.ollama.batch_complete(prompts, max_concurrent=4)
        
        G = nx.DiGraph(summary=text[:1000])  # store first bit as graph summary
        
        for raw_json in results:
            try:
                data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
                
                # Add nodes
                for ent in data.get("entities", []):
                    name = ent.get("name")
                    if name:
                        if G.has_node(name):
                            G.nodes[name]["mentions"] = G.nodes[name].get("mentions", 1) + 1
                        else:
                            G.add_node(name, type=ent.get("type", "Unknown"), desc=ent.get("description", ""))
                
                # Add edges
                for rel in data.get("relationships", []):
                    src, tgt = rel.get("source"), rel.get("target")
                    if src and tgt:
                        G.add_edge(src, tgt, relation=rel.get("relation", ""), strength=rel.get("strength", 0.5))
                        
            except Exception as e:
                logger.warning("Failed to parse extraction result: %s", e)
                
        logger.info("Graph extraction complete: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G
