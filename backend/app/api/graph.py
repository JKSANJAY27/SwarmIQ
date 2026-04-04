"""
SwarmIQ — Graph API Routes
Replaces Zep Graph API. Fully Local GraphRAG.
"""

import os
import shutil
import uuid
import logging
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
import networkx as nx

from .dependencies import get_llm_router, get_snapshot_store, LLMRouter
from ..ingestion.document_processor import DocumentProcessor
from ..graphrag.entity_extractor import EntityExtractor
from ..graphrag.world_builder import WorldBuilder
from ..config import Config

logger = logging.getLogger("swarmiq.api.graph")
router = APIRouter()


class GraphBuildResponse(BaseModel):
    success: bool
    sim_id: str
    message: str


@router.post("/build", response_model=GraphBuildResponse)
async def build_graph(
    goal: str = Form(""),
    files: List[UploadFile] = File(...),
    llm: LLMRouter = Depends(get_llm_router)
):
    """
    1. Ingest files (PDF, Markdown, etc.)
    2. Extract local NetworkX Graph via Ollama
    3. Build Initial World Context
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    sim_id = f"sim_{uuid.uuid4().hex[:8]}"
    upload_dir = os.path.join(Config.UPLOADS_DIR, sim_id)
    os.makedirs(upload_dir, exist_ok=True)

    extracted_text = []

    # Save and Parse
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        content = DocumentProcessor.process_file(file_path)
        if content:
            extracted_text.append(f"--- Document: {file.filename} ---\n{content}\n")

    full_text = "\n".join(extracted_text)
    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Failed to extract any text from files.")

    logger.info("Building KG from %d chars for sim %s", len(full_text), sim_id)

    # Extract Graph using Local LLM
    extractor = EntityExtractor(ollama=llm.ollama)
    G = await extractor.extract_from_text(full_text)

    # Build World Context
    builder = WorldBuilder(llm=llm)
    world_context = await builder.build_context(G, goal)

    world_context["graph_data"] = {
        "nodes": [{"id": str(n), "name": str(n), "type": d.get("type", "Entity"), "attrs": dict(d)} for n, d in G.nodes(data=True)],
        "edges": [{"source": str(u), "target": str(v), "relation": d.get("relation", "RELATED"), "attrs": dict(d)} for u, v, d in G.edges(data=True)]
    }

    # Save initial Seed State
    import json
    seed_file = os.path.join(Config.EXPORTS_DIR, f"{sim_id}_seed.json")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(world_context, f, indent=2)

    return GraphBuildResponse(
        success=True,
        sim_id=sim_id,
        message="Knowledge Graph and World Context built successfully."
    )
