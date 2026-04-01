"""
SwarmIQ — WorldState definition.
Complete serializable state of one simulation tick.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .agent import Agent


@dataclass
class WorldState:
    """
    Complete state of a simulation at a given tick.
    Fully serializable to JSON for SQLite snapshots.
    """

    sim_id: str
    tick: int
    agents: dict[str, Agent]               # agent_id -> Agent
    active_topics: list[str]               # topics being discussed this tick
    global_events: list[dict]             # events injected this tick
    opinion_clusters: list[list[str]]     # agent_id groups by opinion similarity
    echo_chambers: list[dict]             # detected echo chambers
    graph_data: dict = field(default_factory=dict)  # serialized graph (nodes/edges)

    # Runtime-only (not persisted)
    _graph: nx.DiGraph | None = field(default=None, repr=False, compare=False)

    @property
    def graph(self) -> nx.DiGraph:
        if self._graph is None:
            self._graph = nx.DiGraph()
            for node in self.graph_data.get("nodes", []):
                self._graph.add_node(node["id"], **node.get("attrs", {}))
            for edge in self.graph_data.get("edges", []):
                self._graph.add_edge(edge["source"], edge["target"], **edge.get("attrs", {}))
        return self._graph

    @graph.setter
    def graph(self, g: nx.DiGraph) -> None:
        self._graph = g
        self.graph_data = {
            "nodes": [{"id": n, "attrs": dict(d)} for n, d in g.nodes(data=True)],
            "edges": [
                {"source": u, "target": v, "attrs": dict(d)}
                for u, v, d in g.edges(data=True)
            ],
        }

    def to_snapshot(self) -> dict:
        """Serialize to JSON-safe dict for SQLite storage."""
        return {
            "sim_id": self.sim_id,
            "tick": self.tick,
            "agents": {aid: a.to_dict() for aid, a in self.agents.items()},
            "active_topics": self.active_topics,
            "global_events": self.global_events,
            "opinion_clusters": self.opinion_clusters,
            "echo_chambers": self.echo_chambers,
            "graph_data": self.graph_data,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "WorldState":
        """Deserialize from SQLite snapshot."""
        agents = {aid: Agent.from_dict(ad) for aid, ad in data["agents"].items()}
        state = cls(
            sim_id=data["sim_id"],
            tick=data["tick"],
            agents=agents,
            active_topics=data.get("active_topics", []),
            global_events=data.get("global_events", []),
            opinion_clusters=data.get("opinion_clusters", []),
            echo_chambers=data.get("echo_chambers", []),
            graph_data=data.get("graph_data", {}),
        )
        return state

    def opinion_summary(self) -> dict[str, float]:
        """Mean opinion per topic across all agents."""
        if not self.agents or not self.active_topics:
            return {}
        summary = {}
        for topic in self.active_topics:
            values = [a.opinions.get(topic, 0.0) for a in self.agents.values()]
            summary[topic] = round(sum(values) / len(values), 4) if values else 0.0
        return summary
