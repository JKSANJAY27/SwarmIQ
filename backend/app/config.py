"""
SwarmIQ Backend Configuration
"""

import os
from dotenv import load_dotenv

# Load .env from project root
project_root_env = os.path.join(os.path.dirname(__file__), "../../.env")
if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    load_dotenv(override=True)


class Config:
    """Application configuration loaded from environment variables."""

    # --- LLM: Local (Ollama) ---
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_AGENT_MODEL: str = os.environ.get("OLLAMA_AGENT_MODEL", "llama3.2:3b")
    OLLAMA_EMBED_MODEL: str = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # --- LLM: Cloud (Gemini, optional) ---
    GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    # --- Memory: ChromaDB ---
    CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
    CHROMA_COLLECTION_PREFIX: str = os.environ.get(
        "CHROMA_COLLECTION_PREFIX", "swarmiq_agent_"
    )

    # --- Simulation ---
    SIM_MAX_AGENTS: int = int(os.environ.get("SIM_MAX_AGENTS", "500"))
    SIM_DEFAULT_TICKS: int = int(os.environ.get("SIM_DEFAULT_TICKS", "100"))
    SIM_SNAPSHOT_INTERVAL: int = int(os.environ.get("SIM_SNAPSHOT_INTERVAL", "10"))
    SIM_PARALLEL_WORKERS: int = int(os.environ.get("SIM_PARALLEL_WORKERS", "8"))

    # --- Storage ---
    SQLITE_DB_PATH: str = os.environ.get("SQLITE_DB_PATH", "./data/swarmiq.db")
    UPLOADS_DIR: str = os.environ.get("UPLOADS_DIR", "./data/uploads")
    EXPORTS_DIR: str = os.environ.get("EXPORTS_DIR", "./data/exports")

    # --- Server ---
    BACKEND_PORT: int = int(os.environ.get("BACKEND_PORT", "5001"))
    FRONTEND_PORT: int = int(os.environ.get("FRONTEND_PORT", "3000"))

    # --- File Upload ---
    MAX_CONTENT_LENGTH: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {"pdf", "md", "txt", "markdown", "docx", "html", "htm"}

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required data directories if they don't exist."""
        for path in [
            cls.CHROMA_PERSIST_DIR,
            cls.UPLOADS_DIR,
            cls.EXPORTS_DIR,
            os.path.dirname(cls.SQLITE_DB_PATH),
        ]:
            if path:
                os.makedirs(path, exist_ok=True)
