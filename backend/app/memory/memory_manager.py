"""
SwarmIQ — Memory Manager
Manages AgentMemory instances for all agents in a simulation.
"""

import asyncio
import logging

import chromadb

from .chroma_memory import AgentMemory
from ..config import Config

logger = logging.getLogger("swarmiq.memory.manager")


class MemoryManager:
    """
    Manages AgentMemory instances for all agents in a simulation.
    Handles creation, retrieval, and bulk operations.
    """

    def __init__(self, chroma_client: chromadb.Client):
        self._memories: dict[str, AgentMemory] = {}
        self._client = chroma_client

    def get_or_create(self, agent_id: str) -> AgentMemory:
        """Return existing or create new AgentMemory for agent_id."""
        if agent_id not in self._memories:
            self._memories[agent_id] = AgentMemory(agent_id, self._client)
        return self._memories[agent_id]

    async def clear_simulation(self, sim_id: str) -> None:
        """Delete all agent memory collections for a given simulation."""
        prefix = Config.CHROMA_COLLECTION_PREFIX
        to_remove = []
        for agent_id, mem in self._memories.items():
            if agent_id.startswith(sim_id):
                for col in mem._collections.values():
                    try:
                        await asyncio.to_thread(
                            self._client.delete_collection, col.name
                        )
                    except Exception as exc:
                        logger.warning("Failed to delete collection %s: %s", col.name, exc)
                to_remove.append(agent_id)
        for aid in to_remove:
            del self._memories[aid]
        logger.info("Cleared %d agent memories for sim %s", len(to_remove), sim_id)

    async def export_snapshot(self, sim_id: str, tick: int) -> dict:
        """Export memory state for all agents at a given tick (for replay)."""
        snapshot: dict = {"sim_id": sim_id, "tick": tick, "agents": {}}
        for agent_id, mem in self._memories.items():
            agent_snap: dict = {}
            for mtype in ("episodic", "semantic", "social"):
                collection = mem._collections[mtype]
                try:
                    data = await asyncio.to_thread(
                        collection.get, include=["documents", "metadatas"]
                    )
                    agent_snap[mtype] = {
                        "documents": data.get("documents", []),
                        "metadatas": data.get("metadatas", []),
                    }
                except Exception:
                    agent_snap[mtype] = {"documents": [], "metadatas": []}
            snapshot["agents"][agent_id] = agent_snap
        return snapshot
