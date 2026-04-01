"""
SwarmIQ — Simulation API Router
Drives the async SimulationEngine, hooks into WebSockets, and controls ticks.
"""

import json
import logging
import os
import uuid
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .dependencies import (
    get_llm_router,
    get_memory_manager,
    get_snapshot_store,
    get_active_simulations,
    LLMRouter,
)
from ..memory.memory_manager import MemoryManager
from ..simulation.engine import SimulationEngine
from ..simulation.analytics import SimulationAnalytics
from ..config import Config

logger = logging.getLogger("swarmiq.api.simulation")
router = APIRouter()

class SimulationStartRequest(BaseModel):
    sim_id: str
    num_agents: int = 50
    num_ticks: int = 100

class InjectEventRequest(BaseModel):
    event_description: str


@router.post("/start")
async def start_simulation(
    req: SimulationStartRequest,
    llm: LLMRouter = Depends(get_llm_router),
    memory: MemoryManager = Depends(get_memory_manager)
):
    """
    Initialize the Simulation Engine, spawn N agents, and prepare for run.
    """
    seed_file = os.path.join(Config.EXPORTS_DIR, f"{req.sim_id}_seed.json")
    if not os.path.exists(seed_file):
        raise HTTPException(status_code=404, detail="Seed context not found. Build graph first.")
        
    with open(seed_file, "r", encoding="utf-8") as f:
        world_context = json.load(f)

    engine = SimulationEngine(
        sim_id=req.sim_id,
        llm=llm,
        memory_manager=memory,
        world_context=world_context
    )
    
    # Initialize agents via LLM
    await engine.initialize_agents(count=min(req.num_agents, Config.SIM_MAX_AGENTS))
    
    # Register active engine
    sims = get_active_simulations()
    sims[req.sim_id] = engine

    # Run in the background (we start it immediately to a background task)
    # The client connects via websocket to view progress
    
    async def run_loop():
        try:
            for _ in range(req.num_ticks):
                await engine.tick()
                
                # Check snapshot
                if engine.state.tick % Config.SIM_SNAPSHOT_INTERVAL == 0:
                    store = await get_snapshot_store()
                    await store.save_snapshot(engine.state)
                    
            # Analytics on finished run
            SimulationAnalytics.detect_echo_chambers(engine.state)
            SimulationAnalytics.calculate_influence(engine.state)
            
            store = await get_snapshot_store()
            await store.save_snapshot(engine.state)
            logger.info("Simulation %s finished.", req.sim_id)
            
        except Exception as e:
            logger.error("Simulation run failed: %s", e)

    asyncio.create_task(run_loop())
    
    return {"success": True, "sim_id": req.sim_id, "message": f"Simulation running ({req.num_agents} agents)"}


@router.post("/{sim_id}/inject")
async def inject_event(
    sim_id: str,
    req: InjectEventRequest,
    llm: LLMRouter = Depends(get_llm_router)
):
    """God mode: Inject a structural event mid-simulation."""
    sims = get_active_simulations()
    if sim_id not in sims:
        raise HTTPException(status_code=404, detail="Simulation not physically active in memory")
        
    engine: SimulationEngine = sims[sim_id]
    
    # Synthesize structured event from natural language
    event_dict = await llm.call(
        "event_synthesize", 
        world_state={"active_topics": engine.state.active_topics}, 
        event_description=req.event_description
    )
    
    await engine.inject_event(event_dict)
    return {"success": True, "event": event_dict}


@router.websocket("/{sim_id}/ws")
async def simulation_websocket(websocket: WebSocket, sim_id: str):
    """Realtime stream of simulation events via WebSocket."""
    await websocket.accept()
    
    sims = get_active_simulations()
    if sim_id not in sims:
        await websocket.close(code=1008, reason="Sim not running")
        return
        
    engine: SimulationEngine = sims[sim_id]
    
    # Simple pubsub queue per connection
    queue = asyncio.Queue()
    
    def on_tick(state):
        queue.put_nowait({"type": "tick", "tick": state.tick})
        
    def on_event(event):
        queue.put_nowait({"type": "public_statement", "data": event})
        
    engine.on_tick(on_tick)
    engine.on_event(on_event)
    
    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for %s", sim_id)
        # We should ideally remove the hooks here to avoid memory leaks
        # In a production system we'd use id-based weakrefs 
        try:
            engine._on_tick.remove(on_tick)
            engine._on_event.remove(on_event)
        except ValueError:
            pass
