"""
SwarmIQ — Local GraphRAG implementation.
Uses Ollama to extract entities and build a local NetworkX graph.
Falls back to keyword-based extraction when Ollama is unavailable.
"""

import json
import logging
import re
from typing import Any

import networkx as nx

from ..llm.ollama_client import OllamaClient
from ..simulation.prompts import ENTITY_EXTRACT_PROMPT

logger = logging.getLogger("swarmiq.graphrag")


def _keyword_fallback_extraction(text: str) -> dict:
    """
    Minimal entity extraction without LLM.
    Pulls capitalized noun phrases as entities and creates co-occurrence edges.
    Good enough to bootstrap a world context when Ollama is offline.
    """
    # Find capitalized phrases (naive NER proxy)
    raw_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    # Deduplicate and keep most frequent (top 20)
    freq: dict[str, int] = {}
    for e in raw_entities:
        if len(e) > 3:
            freq[e] = freq.get(e, 0) + 1
    top_entities = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]

    entities = [
        {"name": name, "type": "Concept", "description": f"Mentioned {cnt} times in the document."}
        for name, cnt in top_entities
    ]

    # Co-occurrence edges: pair adjacent high-freq entities
    names = [e["name"] for e in entities]
    relationships = [
        {"source": names[i], "target": names[i + 1], "relation": "related_to", "strength": 0.5}
        for i in range(min(len(names) - 1, 10))
    ]
    return {"entities": entities, "relationships": relationships}


class EntityExtractor:
    """
    Extracts entities and relationships from text using Ollama.
    Falls back to keyword-based extraction when Ollama is unavailable.
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
        try:
            results = await self.ollama.batch_complete(prompts, max_concurrent=4)
        except Exception as exc:
            logger.warning("Ollama unavailable for entity extraction (%s). Using keyword fallback.", exc)
            results = [""] * len(chunks)
        
        G = nx.DiGraph(summary=text[:1000])  # store first bit as graph summary
        
        any_llm_success = False
        for chunk_idx, raw_json in enumerate(results):
            try:
                data = None
                if raw_json and isinstance(raw_json, str):
                    try:
                        data = json.loads(raw_json)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(raw_json, dict) and raw_json:
                    data = raw_json

                # Fallback for this chunk if LLM returned nothing
                if not data or not isinstance(data, dict):
                    data = _keyword_fallback_extraction(chunks[chunk_idx])
                else:
                    any_llm_success = True

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
                logger.warning("Failed to parse extraction result for chunk %d: %s", chunk_idx, e)

        if not any_llm_success:
            logger.warning("All chunks used keyword fallback — Ollama appears offline.")
                
        logger.info("Graph extraction complete: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G
