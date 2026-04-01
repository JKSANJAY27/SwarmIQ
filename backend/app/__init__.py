"""
SwarmIQ Backend — FastAPI Application Factory
"""

import os
import logging
import chromadb

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Config

logger = logging.getLogger("swarmiq")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    Config.ensure_dirs()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = FastAPI(
        title="SwarmIQ API",
        description="SwarmIQ — Local-first multi-agent swarm simulation and prediction engine",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialise shared ChromaDB client (singleton on app state)
    chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
    app.state.chroma_client = chroma_client
    logger.info("ChromaDB client initialised at %s", Config.CHROMA_PERSIST_DIR)

    # Register routers
    from .api.graph import router as graph_router
    from .api.simulation import router as simulation_router
    from .api.report import router as report_router

    app.include_router(graph_router, prefix="/api/graph", tags=["graph"])
    app.include_router(simulation_router, prefix="/api/simulations", tags=["simulation"])
    app.include_router(report_router, prefix="/api/report", tags=["report"])

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "SwarmIQ Backend"}

    logger.info("SwarmIQ Backend ready")
    return app
