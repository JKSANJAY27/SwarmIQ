"""
SwarmIQ — Shared API Dependencies
Provides FastAPi dependency injection for database, LLM routers, and simulation engine.
"""

from fastapi import Request
import chromadb

from ..db.snapshot_store import SnapshotStore
from ..llm.ollama_client import OllamaClient
from ..llm.gemini_client import GeminiClient
from ..llm.llm_router import LLMRouter
from ..memory.memory_manager import MemoryManager
from ..config import Config

_snapshot_store = None
_ollama_client = None
_gemini_client = None
_llm_router = None


async def get_snapshot_store() -> SnapshotStore:
    global _snapshot_store
    if _snapshot_store is None:
        _snapshot_store = SnapshotStore()
        await _snapshot_store.initialize()
    return _snapshot_store


def get_llm_router() -> LLMRouter:
    global _ollama_client, _gemini_client, _llm_router
    if _llm_router is None:
        _ollama_client = OllamaClient()
        if Config.GEMINI_API_KEY:
            _gemini_client = GeminiClient()
        _llm_router = LLMRouter(ollama=_ollama_client, gemini=_gemini_client)
    return _llm_router


def get_memory_manager(request: Request) -> MemoryManager:
    # Uses the shared persistent chroma client from app state
    chroma_client = getattr(request.app.state, "chroma_client", None)
    if not chroma_client:
        chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
    return MemoryManager(chroma_client)


# In-memory dict mapping sim_id -> SimulationEngine instance for running simulations
_active_simulations = {}

def get_active_simulations() -> dict:
    return _active_simulations
