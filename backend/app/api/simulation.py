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
    num_agents: int = 5
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
    
    # Register active engine immediately
    sims = get_active_simulations()
    sims[req.sim_id] = engine

    # Run in the background (we start it immediately to a background task)
    # The client connects via websocket to view progress
    
    async def run_loop():
        try:
            # Initialize agents via LLM in the background
            await engine.initialize_agents(count=min(req.num_agents, Config.SIM_MAX_AGENTS))
            
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


@router.post("/stop")
async def stop_simulation(req: dict):
    """Stop a running simulation gracefully."""
    sim_id = req.get("simulation_id") or req.get("sim_id")
    if not sim_id:
        raise HTTPException(status_code=400, detail="simulation_id is required")
    sims = get_active_simulations()
    if sim_id in sims:
        del sims[sim_id]
        logger.info("Simulation %s stopped by user request.", sim_id)
    return {"success": True, "sim_id": sim_id, "message": "Simulation stopped."}


@router.get("/{sim_id}/status")
async def get_simulation_status(sim_id: str):
    """Return the current tick/status of a running or completed simulation."""
    sims = get_active_simulations()
    if sim_id in sims:
        engine: SimulationEngine = sims[sim_id]
        return {
            "success": True,
            "sim_id": sim_id,
            "status": "running",
            "tick": engine.state.tick,
            "agent_count": len(engine.state.agents),
            "active_topics": engine.state.active_topics,
            "twitter_current_round": engine.state.tick,
            "reddit_current_round": engine.state.tick,
            "twitter_running": True,
            "reddit_running": True,
            "twitter_actions_count": len(engine.state.agents) * engine.state.tick,
            "total_rounds": 100,
        }
    # Not active — check snapshot store for a completed simulation
    store = await get_snapshot_store()
    latest_tick = await store.get_latest_tick(sim_id)
    if latest_tick > 0:
        return {
            "success": True,
            "sim_id": sim_id,
            "status": "completed",
            "tick": latest_tick,
            "twitter_current_round": latest_tick,
            "reddit_current_round": latest_tick,
            "twitter_running": False,
            "reddit_running": False,
            "twitter_completed": True,
            "reddit_completed": True,
            "total_rounds": 100,
        }
    return {"success": True, "sim_id": sim_id, "status": "unknown", "tick": 0}
    
@router.get("/{sim_id}/graph")
async def get_simulation_graph(sim_id: str):
    """Retrieve the real-time or snapshot graph data for the simulation."""
    sims = get_active_simulations()
    if sim_id in sims:
        engine: SimulationEngine = sims[sim_id]
        if engine.state.graph_data:
            return {"success": True, "data": engine.state.graph_data}
    
    # Try looking in the latest snapshot
    store = await get_snapshot_store()
    latest_tick = await store.get_latest_tick(sim_id)
    if latest_tick > 0:
        state = await store.load_snapshot(sim_id, latest_tick)
        if state and state.graph_data:
            return {"success": True, "data": state.graph_data}
            
    # Try the initial seed
    seed_file = os.path.join(Config.EXPORTS_DIR, f"{sim_id}_seed.json")
    if os.path.exists(seed_file):
        with open(seed_file, "r", encoding="utf-8") as f:
            world_context = json.load(f)
            if "graph_data" in world_context:
                return {"success": True, "data": world_context["graph_data"]}

    # Fallback to empty
    return {"success": True, "data": {"nodes": [], "edges": []}}


@router.get("/{sim_id}/config")
async def get_simulation_config(sim_id: str):
    """Return the world context / config seed for this simulation."""
    seed_file = os.path.join(Config.EXPORTS_DIR, f"{sim_id}_seed.json")
    if not os.path.exists(seed_file):
        return {"success": True, "sim_id": sim_id, "config": {}, "status": "no_seed"}
    with open(seed_file, "r", encoding="utf-8") as f:
        import json as _json
        world_context = _json.load(f)
    return {"success": True, "sim_id": sim_id, "config": world_context}


@router.get("/{sim_id}/config/realtime")
async def get_simulation_config_realtime(sim_id: str):
    """Return real-time world context/config — same as /config, also checks active engine."""
    sims = get_active_simulations()
    config_data: dict = {}

    # Pull live state from running engine if available
    if sim_id in sims:
        engine: SimulationEngine = sims[sim_id]
        config_data = {
            "summary": engine.state.graph.graph.get("summary", ""),
            "active_topics": engine.state.active_topics,
            "agent_count": len(engine.state.agents),
            "tick": engine.state.tick,
            "status": "running",
        }
    else:
        seed_file = os.path.join(Config.EXPORTS_DIR, f"{sim_id}_seed.json")
        if os.path.exists(seed_file):
            with open(seed_file, "r", encoding="utf-8") as f:
                import json as _json
                config_data = _json.load(f)
            config_data["status"] = "ready"
        else:
            config_data = {"status": "no_seed"}

    return {"success": True, "sim_id": sim_id, "config": config_data}


@router.get("/{sim_id}/run-status")
async def get_run_status(sim_id: str):
    """Alias — delegates to /status."""
    return await get_simulation_status(sim_id)


@router.get("/{sim_id}/run-status/detail")
async def get_run_status_detail(sim_id: str):
    """Extended status including agent samples."""
    base = await get_simulation_status(sim_id)
    sims = get_active_simulations()
    if sim_id in sims:
        engine: SimulationEngine = sims[sim_id]
        agent_samples = [
            {
                "id": a.id,
                "name": a.name,
                "occupation": a.occupation,
                "opinions": a.opinions,
                "influence_score": a.influence_score,
            }
            for a in list(engine.state.agents.values())[:10]
        ]
        base["agent_samples"] = agent_samples
        base["active_topics"] = engine.state.active_topics
    return base


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
        try:
            engine._on_tick.remove(on_tick)
            engine._on_event.remove(on_event)
        except ValueError:
            pass
