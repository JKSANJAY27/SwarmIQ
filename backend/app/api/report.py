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


class ChatRequest(BaseModel):
    simulation_id: str
    message: str
    chat_history: Optional[list] = None


def _build_fallback_report(sim_id: str, summary_payload: dict, agent_samples: list, goal: str) -> str:
    """Build a minimal markdown report without LLM when all APIs are unavailable."""
    opinions = summary_payload.get("mean_opinions", {})
    agent_count = summary_payload.get("total_agents", 0)
    final_tick = summary_payload.get("final_tick", 0)

    opinion_lines = "\n".join(
        f"- **{topic}**: {round(score, 3):+.3f} ({'positive' if score > 0 else 'negative' if score < 0 else 'neutral'} sentiment)"
        for topic, score in opinions.items()
    ) or "- No opinion data available."

    agent_lines = "\n".join(
        f"| {a.get('name','?')} | {a.get('occupation','?')} | "
        + ", ".join(f"{k}: {round(v,2):+.2f}" for k, v in (a.get('opinions') or {}).items())
        + " |"
        for a in agent_samples[:5]
    ) or "| No agent data |  |  |"

    return f"""# SwarmIQ Simulation Report
**Simulation ID:** `{sim_id}`
**Prediction Goal:** {goal}
**Final Tick:** {final_tick} | **Total Agents:** {agent_count}

---

## Opinion Summary (Mean across all agents)

{opinion_lines}

## Sample Agent Profiles

| Name | Occupation | Key Opinions |
|------|------------|--------------|
{agent_lines}

## Echo Chambers Detected

{len(summary_payload.get('echo_chambers', []))} echo chamber(s) identified during simulation.

---

> *Note: This report was generated using the heuristic fallback engine (LLM synthesis was unavailable).*
"""


@router.post("/generate")
async def generate_report(
    req: ReportRequest,
    llm: LLMRouter = Depends(get_llm_router),
    store: SnapshotStore = Depends(get_snapshot_store)
):
    """Generate final prediction report based on the last simulation snapshot."""
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
        
    # 4. Synthesize via LLM (with fallback)
    summary_payload = {
        "final_tick": latest_tick,
        "total_agents": len(state.agents),
        "mean_opinions": opinion_summary,
        "echo_chambers": state.echo_chambers,
    }

    report_md = ""
    try:
        report_md = await llm.call(
            "report_synthesize",
            simulation_summary=summary_payload,
            agent_samples=agent_samples,
            goal=req.goal
        )
    except Exception as exc:
        logger.warning("LLM report synthesis failed: %s — using fallback report.", exc)

    if not report_md or not isinstance(report_md, str):
        report_md = _build_fallback_report(req.sim_id, summary_payload, agent_samples, req.goal)
    
    # 5. Save to disk
    report_file = os.path.join(Config.EXPORTS_DIR, f"{req.sim_id}_report.md")
    os.makedirs(Config.EXPORTS_DIR, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    return {
        "success": True, 
        "sim_id": req.sim_id, 
        "markdown": report_md
    }


@router.get("/generate/status")
async def get_report_status(report_id: str):
    """Check if a report has been generated."""
    report_file = os.path.join(Config.EXPORTS_DIR, f"{report_id}_report.md")
    if os.path.exists(report_file):
        return {"success": True, "report_id": report_id, "status": "completed"}
    return {"success": True, "report_id": report_id, "status": "pending"}


@router.get("/{sim_id}")
async def get_report(sim_id: str):
    """Retrieve a previously generated report."""
    report_file = os.path.join(Config.EXPORTS_DIR, f"{sim_id}_report.md")
    if not os.path.exists(report_file):
        raise HTTPException(status_code=404, detail="Report not found. Run /generate first.")
    with open(report_file, "r", encoding="utf-8") as f:
        markdown = f.read()
    return {"success": True, "sim_id": sim_id, "markdown": markdown}


@router.post("/chat")
async def chat_with_report(
    req: ChatRequest,
    llm: LLMRouter = Depends(get_llm_router),
    store: SnapshotStore = Depends(get_snapshot_store)
):
    """Interactive Q&A against simulation results."""
    sim_id = req.simulation_id

    # Load report context
    report_file = os.path.join(Config.EXPORTS_DIR, f"{sim_id}_report.md")
    report_context = ""
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            report_context = f.read()[:3000]  # Trim for context window

    # Try LLM chat
    try:
        system = (
            "You are an expert analyst reviewing a social simulation report. "
            "Answer questions based on the provided report context. Be concise and factual.\n\n"
            f"REPORT CONTEXT:\n{report_context}"
        )
        response = await llm.call(
            "generic_chat",
            system=system,
            message=req.message,
            history=req.chat_history or []
        )
        if not response or not isinstance(response, str):
            raise ValueError("Empty LLM response")
    except Exception as exc:
        logger.warning("Chat LLM failed: %s", exc)
        response = (
            f"I couldn't process your question with the LLM right now. "
            f"Based on the report for simulation `{sim_id}`, "
            f"you asked: \"{req.message}\". "
            f"Please check the report markdown directly for details."
        )

    return {"success": True, "response": response, "sim_id": sim_id}


@router.get("/{sim_id}/agent-log")
async def get_agent_log(sim_id: str, from_line: int = 0):
    """Retrieve incremental agent logs."""
    return {"success": True, "data": {"logs": [], "from_line": from_line}}


@router.get("/{sim_id}/console-log")
async def get_console_log(sim_id: str, from_line: int = 0):
    """Retrieve incremental console logs."""
    return {"success": True, "data": {"logs": [], "from_line": from_line}}
