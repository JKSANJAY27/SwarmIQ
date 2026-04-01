"""
SwarmIQ — Branch Manager
Manages forking and parallel timelines using SQLite snapshots.
"""

import copy
import logging

from .world import WorldState
from ..memory.memory_manager import MemoryManager

logger = logging.getLogger("swarmiq.simulation.branch")


class BranchManager:
    """
    Manages branching/forking the simulation timeline.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

    async def fork_state(self, original_state: WorldState, new_sim_id: str) -> WorldState:
        """
        Deep copy an entire WorldState. 
        Agent memories must be duplicated in ChromaDB.
        """
        logger.info("Forking sim %s -> %s at tick %d", original_state.sim_id, new_sim_id, original_state.tick)
        
        # Deep copy state in memory
        new_state = copy.deepcopy(original_state)
        new_state.sim_id = new_sim_id
        
        # Update agent IDs and copy ChromaDB collections
        # (For SwarmIQ MVP we re-use the memory snapshot export/import mechanism)
        snapshot = await self.memory.export_snapshot(original_state.sim_id, original_state.tick)
        
        new_agents = {}
        for old_aid, agent in new_state.agents.items():
            # Rewrite agent ID
            new_aid = old_aid.replace(original_state.sim_id, new_sim_id)
            agent.id = new_aid
            new_agents[new_aid] = agent
            
            # Copy memory to new ID
            new_mem = self.memory.get_or_create(new_aid)
            old_mem_data = snapshot["agents"].get(old_aid, {})
            
            import asyncio
            for mtype in ("episodic", "semantic", "social"):
                mdata = old_mem_data.get(mtype, {})
                docs = mdata.get("documents", [])
                metas = mdata.get("metadatas", [])
                
                # Write to new collection
                if docs:
                    import time
                    col = new_mem._collections[mtype]
                    # Generate new unique IDs for entries
                    ids = [f"{new_aid}_{mtype}_{int(time.time()*1000)}_{i}" for i in range(len(docs))]
                    # Update metadata agent_id
                    for meta in metas:
                        if "agent_id" in meta:
                            meta["agent_id"] = new_aid
                            
                    await asyncio.to_thread(col.add, documents=docs, metadatas=metas, ids=ids)
                    
        new_state.agents = new_agents
        
        return new_state
