"""
SwarmIQ — SQLite Snapshot Store
Persists WorldState at specific ticks for replay, analytics, and branching.
"""

import json
import logging
from typing import Any

import aiosqlite

from ..config import Config
from ..simulation.world import WorldState

logger = logging.getLogger("swarmiq.db.snapshot_store")


class SnapshotStore:
    """
    Stores full simulation tick snapshots in SQLite.
    
    Tables:
      - simulations: metadata about runs
      - snapshots: WorldState JSON blob per tick
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or Config.SQLITE_DB_PATH
        self._init_task = None

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS simulations (
                    sim_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    max_ticks INTEGER,
                    status TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    sim_id TEXT,
                    tick INTEGER,
                    state_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (sim_id, tick)
                )
            """)
            await db.commit()
        logger.info("SnapshotStore initialized at %s", self.db_path)

    async def save_snapshot(self, state: WorldState) -> None:
        """Persist a WorldState to SQLite."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO snapshots (sim_id, tick, state_json) VALUES (?, ?, ?)",
                (state.sim_id, state.tick, json.dumps(state.to_snapshot()))
            )
            await db.commit()
        logger.debug("Saved snapshot for %s tick %d", state.sim_id, state.tick)

    async def load_snapshot(self, sim_id: str, tick: int) -> WorldState | None:
        """Retrieve a specific WorldState snapshot."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT state_json FROM snapshots WHERE sim_id = ? AND tick = ?",
                (sim_id, tick)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    return WorldState.from_snapshot(data)
        return None

    async def get_latest_tick(self, sim_id: str) -> int:
        """Get the highest tick number saved for a simulation."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT MAX(tick) FROM snapshots WHERE sim_id = ?",
                (sim_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row and row[0] is not None else 0

    async def get_all_ticks(self, sim_id: str) -> list[int]:
        """Get all available ticks for replay."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT tick FROM snapshots WHERE sim_id = ? ORDER BY tick ASC",
                (sim_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
