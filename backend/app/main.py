"""
SwarmIQ Backend — ASGI entrypoint
"""

import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from app import create_app  # noqa: E402

app = create_app()
