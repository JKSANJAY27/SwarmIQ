"""
SwarmIQ — Per-agent ChromaDB memory layer.
Three memory types per agent: episodic, semantic, social.
All embeddings generated locally via Ollama nomic-embed-text.
"""

import asyncio
import logging
import time
from typing import Any

import chromadb

from ..config import Config

logger = logging.getLogger("swarmiq.memory")

PREFIX = Config.CHROMA_COLLECTION_PREFIX
MEMORY_TYPES = ("episodic", "semantic", "social")


class AgentMemory:
    """
    Per-agent persistent vector memory backed by ChromaDB.

    Three memory types per agent:
      - episodic:  things the agent experienced (events, interactions)
      - semantic:  beliefs, opinions, world knowledge
      - social:    relationships with other agents (trust, influence scores)

    All embeddings generated locally via Ollama nomic-embed-text.
    """

    def __init__(self, agent_id: str, chroma_client: chromadb.Client):
        self.agent_id = agent_id
        self._client = chroma_client
        self._collections: dict[str, chromadb.Collection] = {}
        for mtype in MEMORY_TYPES:
            col_name = f"{PREFIX}{agent_id}_{mtype}"
            self._collections[mtype] = self._client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"},
            )

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding via Ollama (wrapped in thread)."""
        import ollama
        def _call():
            client = ollama.Client(host=Config.OLLAMA_BASE_URL)
            resp = client.embeddings(model=Config.OLLAMA_EMBED_MODEL, prompt=text)
            return resp["embedding"]
        try:
            return await asyncio.to_thread(_call)
        except Exception as exc:
            logger.warning("Embedding failed for agent %s: %s", self.agent_id, exc)
            return []

    async def remember(
        self,
        memory_type: str,
        content: str,
        metadata: dict | None = None,
        tick: int = 0,
    ) -> None:
        """Embed content and store in the appropriate collection."""
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unknown memory type: {memory_type}")
        embedding = await self._embed(content)
        if not embedding:
            return
        meta = (metadata or {}).copy()
        meta.update({
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "tick": tick,
            "memory_type": memory_type,
        })
        doc_id = f"{self.agent_id}_{memory_type}_{int(time.time()*1000)}"
        collection = self._collections[memory_type]
        await asyncio.to_thread(
            collection.add,
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta],
            ids=[doc_id],
        )

    async def recall(
        self,
        memory_type: str,
        query: str,
        n_results: int = 5,
    ) -> list[str]:
        """Semantic search over a memory type. Returns list of relevant memories."""
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unknown memory type: {memory_type}")
        embedding = await self._embed(query)
        if not embedding:
            return []
        collection = self._collections[memory_type]
        try:
            count = await asyncio.to_thread(lambda: collection.count())
            if count == 0:
                return []
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[embedding],
                n_results=min(n_results, count),
            )
            return results.get("documents", [[]])[0]
        except Exception as exc:
            logger.warning("Recall failed (agent=%s, type=%s): %s", self.agent_id, memory_type, exc)
            return []

    async def recall_all(
        self, query: str, n_results: int = 3
    ) -> dict[str, list[str]]:
        """Search across all three memory types. Returns dict keyed by type."""
        results = {}
        for mtype in MEMORY_TYPES:
            results[mtype] = await self.recall(mtype, query, n_results=n_results)
        return results

    async def forget_old(self, memory_type: str, keep_last_n: int = 200) -> None:
        """Prune oldest memories to keep collection size bounded."""
        if memory_type not in MEMORY_TYPES:
            return
        collection = self._collections[memory_type]
        try:
            count = await asyncio.to_thread(lambda: collection.count())
            if count <= keep_last_n:
                return
            all_items = await asyncio.to_thread(
                collection.get, include=["metadatas"]
            )
            ids = all_items["ids"]
            metadatas = all_items["metadatas"]
            # Sort by timestamp ascending, delete oldest
            paired = sorted(zip(ids, metadatas), key=lambda x: x[1].get("timestamp", 0))
            to_delete = [pair[0] for pair in paired[: count - keep_last_n]]
            if to_delete:
                await asyncio.to_thread(collection.delete, ids=to_delete)
        except Exception as exc:
            logger.warning("forget_old failed (agent=%s): %s", self.agent_id, exc)
