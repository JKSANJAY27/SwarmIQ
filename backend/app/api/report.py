"""
SwarmIQ — Report API Routes
Generates the final predictive analysis using Gemini (or Ollama fallback).
"""

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .dependencies import get_llm_router, get_snapshot_store, LLMRouter
from ..db.snapshot_store import SnapshotStore
from ..simulation.world import WorldState
from ..config import Config

logger = logging.getLogger("swarmiq.api.report")
router = APIRouter()

class ReportRequest(BaseModel):
    sim_id: str
    goal: str


@router.post("/generate")
async def generate_report(
    req: ReportRequest,
    llm: LLMRouter = Depends(get_llm_router),
    store: SnapshotStore = Depends(get_snapshot_store)
):
    """
    Generate final prediction report based on the last simulation snapshot.
    """
    logger.info("Generating report for sim %s, goal: %s", req.sim_id, req.goal)

    # 1. Load latest state
    latest_tick = await store.get_latest_tick(req.sim_id)
    if latest_tick == 0:
        raise HTTPException(status_code=400, detail="No snapshots found for this simulation.")
        
    state: WorldState | None = await store.load_snapshot(req.sim_id, latest_tick)
    if not state:
        raise HTTPException(status_code=500, detail="Failed to load world state snapshot.")
        
    # 2. Extract interesting insights
    opinion_summary = state.opinion_summary()
    
    # 3. Grab agent samples
    agent_samples = []
    for aid, agent in list(state.agents.items())[:10]:
        agent_samples.append({
            "name": agent.name,
            "occupation": agent.occupation,
            "opinions": agent.opinions
        })
        
    # 4. Synthesize via LLM
    summary_payload = {
        "final_tick": latest_tick,
        "total_agents": len(state.agents),
        "mean_opinions": opinion_summary,
        "echo_chambers": state.echo_chambers,
    }

    report_md = await llm.call(
        "report_synthesize",
        simulation_summary=summary_payload,
        agent_samples=agent_samples,
        goal=req.goal
    )
    
    # Save to disk
    report_file = os.path.join(Config.EXPORTS_DIR, f"{req.sim_id}_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    return {
        "success": True, 
        "sim_id": req.sim_id, 
        "markdown": report_md
    }
