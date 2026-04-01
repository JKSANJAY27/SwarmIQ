"""
SwarmIQ — API Router Initialization
"""

from .graph import router as graph_router
from .simulation import router as simulation_router
from .report import router as report_router

__all__ = ["graph_router", "simulation_router", "report_router"]
