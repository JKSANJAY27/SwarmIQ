"""
SwarmIQ — Core Simulation Engine
Asynchronous event loop for ticking the swarm simulation.
"""

import asyncio
import json
import logging
import random
import time
from typing import Callable

import networkx as nx

from .agent import Agent, BigFivePersonality
from .world import WorldState
from .prompts import AGENT_REACT_PROMPT, AGENT_INTERACT_PROMPT, AGENT_GENERATE_PROMPT
from ..config import Config
from ..llm.llm_router import LLMRouter
from ..memory.memory_manager import MemoryManager

logger = logging.getLogger("swarmiq.simulation.engine")

# ---------------------------------------------------------------------------
# Fallback agent generation constants (used when Ollama is unavailable)
# ---------------------------------------------------------------------------
_FALLBACK_FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Karen", "Leo", "Maya", "Nathan", "Olivia", "Peter",
    "Quinn", "Rachel", "Samuel", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe", "Alex", "Blake", "Casey", "Drew", "Eli", "Fiona",
    "George", "Holly", "Ivan", "Julia", "Kevin", "Laura", "Mike", "Nina",
    "Oscar", "Priya", "Ryan", "Sara", "Tom", "Ursula", "Vera", "Will",
]
_FALLBACK_LAST_NAMES = [
    "Smith", "Chen", "Martinez", "Kim", "Patel", "Williams", "Johnson",
    "Brown", "Davis", "Wilson", "Thompson", "Moore", "Taylor", "Anderson",
    "Jackson", "White", "Harris", "Martin", "Garcia", "Rodriguez", "Lewis",
    "Lee", "Walker", "Hall", "Allen", "Young", "Scott", "Hernandez",
    "Nguyen", "Singh", "Okafor", "Tanaka", "Ivanova", "Mueller", "Dupont",
]
_FALLBACK_OCCUPATIONS: dict[str, list[str]] = {
    "activist":      ["Community Organizer", "Campaigner", "Advocate", "Volunteer Coordinator"],
    "journalist":    ["Reporter", "Editor", "Blogger", "Correspondent", "Analyst"],
    "policy":        ["Government Official", "Policy Analyst", "Bureaucrat", "Adviser"],
    "business":      ["Business Owner", "Marketing Manager", "Accountant", "Executive"],
    "academic":      ["Professor", "Researcher", "PhD Student", "Scientist"],
    "worker":        ["Factory Worker", "Driver", "Warehouse Operative", "Technician"],
    "professional":  ["Lawyer", "Doctor", "Engineer", "Software Developer", "Nurse"],
    "student":       ["University Student", "Postgraduate Student", "Intern"],
    "media":         ["Influencer", "Content Creator", "Podcaster", "Streamer"],
    "default":       ["Teacher", "Retail Assistant", "Clerk", "Administrator", "Freelancer"],
}


class SimulationEngine:
    """
    Drives the simulation forward in discrete ticks.
    Fully asynchronous. Uses Ollama via LLMRouter for heavy lifting.
    """

    def __init__(
        self,
        sim_id: str,
        llm: LLMRouter,
        memory_manager: MemoryManager,
        world_context: dict,
    ):
        self.sim_id = sim_id
        self.llm = llm
        self.memory = memory_manager
        
        # Initialize an empty world state for tick 0
        self.state = WorldState(
            sim_id=sim_id,
            tick=0,
            agents={},
            active_topics=world_context.get("active_topics", []),
            global_events=[],
            opinion_clusters=[],
            echo_chambers=[],
            graph_data=world_context.get("graph_data", {"nodes": [], "edges": []})
        )
        self.state.graph = nx.DiGraph(seed=world_context)

        # Build agents from archetypes
        self.archetypes = world_context.get("agent_archetypes", [])
        if not self.archetypes:
            self.archetypes = ["General Public"]

        # Callbacks (e.g. for WebSocket broadcasting)
        self._on_tick: list[Callable[[WorldState], None]] = []
        self._on_event: list[Callable[[dict], None]] = []

    def on_tick(self, callback: Callable[[WorldState], None]) -> None:
        """Register a callback to run after every tick."""
        self._on_tick.append(callback)

    def on_event(self, callback: Callable[[dict], None]) -> None:
        """Register a callback to run when agents broadcast public statements."""
        self._on_event.append(callback)

    def _generate_fallback_agent(self, idx: int, archetype: str) -> dict:
        """Generate a procedural agent profile without LLM when Ollama is unavailable."""
        rng = random.Random(idx)
        first = _FALLBACK_FIRST_NAMES[idx % len(_FALLBACK_FIRST_NAMES)]
        last = _FALLBACK_LAST_NAMES[(idx // len(_FALLBACK_FIRST_NAMES)) % len(_FALLBACK_LAST_NAMES)]
        # Ensure uniqueness by appending number suffix when wrapping
        suffix = str(idx // (len(_FALLBACK_FIRST_NAMES) * len(_FALLBACK_LAST_NAMES))) if idx >= len(_FALLBACK_FIRST_NAMES) * len(_FALLBACK_LAST_NAMES) else ""
        name = f"{first} {last}{suffix}"

        arch_lower = archetype.lower()
        occupations = _FALLBACK_OCCUPATIONS["default"]
        for key, occs in _FALLBACK_OCCUPATIONS.items():
            if key in arch_lower:
                occupations = occs
                break
        occupation = rng.choice(occupations)
        age = rng.randint(22, 65)

        initial_sentiments = self.state.graph.graph.get("initial_sentiments", {})
        initial_opinions = {
            t: round(rng.gauss(initial_sentiments.get(t, 0.0), 0.25), 3)
            for t in self.state.active_topics
        }

        background = (
            f"{first} is a {age}-year-old {occupation} with strong personal views "
            f"on the issues unfolding in their community. "
            f"They follow current events closely and are not shy about sharing opinions."
        )
        return {
            "name": name,
            "age": age,
            "occupation": occupation,
            "background": background,
            "initial_opinions": initial_opinions,
        }

    async def initialize_agents(self, count: int) -> None:
        """Generate N agents using LLM (with procedural fallback) and create their memories."""
        logger.info("Initializing %d agents for sim %s", count, self.sim_id)

        archetypes_for_idx = [random.choice(self.archetypes) for _ in range(count)]

        # Build LLM prompts for all agents
        prompts = []
        for idx, arch in enumerate(archetypes_for_idx):
            sys = f"You are creating a character profile for the {arch} archetype."
            usr = AGENT_GENERATE_PROMPT.format(
                seed_summary=self.state.graph.graph.get("summary", ""),
                topics=", ".join(self.state.active_topics)
            )
            prompts.append({"system": sys, "user": usr})

        # Attempt concurrent Ollama calls; fall back gracefully on failure
        try:
            results = await self.llm.ollama.batch_complete(prompts)
        except Exception as exc:
            logger.warning("Ollama batch_complete raised %s — using procedural fallback for all agents.", exc)
            results = [""] * count

        llm_ok = 0
        fallback_ok = 0

        for idx, result_json in enumerate(results):
            try:
                data: dict | None = None

                # --- Try parsing LLM output ---
                if result_json and isinstance(result_json, str):
                    try:
                        data = json.loads(result_json)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(result_json, dict) and result_json:
                    data = result_json

                # --- Procedural fallback when LLM unavailable / returned garbage ---
                if not data or not isinstance(data, dict) or not data.get("name"):
                    data = self._generate_fallback_agent(idx, archetypes_for_idx[idx])
                    fallback_ok += 1
                else:
                    llm_ok += 1

                agent_id = f"agent_{self.sim_id}_{idx}"
                data["id"] = agent_id

                pers = BigFivePersonality.for_occupation(data.get("occupation", "Worker"))
                data["personality"] = pers.to_dict()

                opinions = data.get("initial_opinions", {})
                for t in self.state.active_topics:
                    if t not in opinions:
                        opinions[t] = round(
                            random.gauss(
                                self.state.graph.graph.get("initial_sentiments", {}).get(t, 0.0), 0.2
                            ), 3
                        )
                data["opinions"] = opinions

                agent = Agent.from_dict(data)
                self.state.agents[agent_id] = agent

                mem = self.memory.get_or_create(agent_id)
                await mem.remember(
                    "semantic",
                    f"I am {agent.name}, a {agent.age} year old {agent.occupation}. {agent.background}"
                )

                # Add agent to the visualization graph
                self.state.graph_data["nodes"].append({
                    "id": agent_id, 
                    "name": agent.name, 
                    "type": "Agent",
                    "attrs": {"occupation": agent.occupation, "age": agent.age}
                })
                # Add initial opinion edges
                for topic, op in agent.opinions.items():
                    if abs(op) > 0.3:
                        # Add topic node if it doesn't exist
                        if not any(n["id"] == topic for n in self.state.graph_data["nodes"]):
                            self.state.graph_data["nodes"].append({
                                "id": topic, "name": topic, "type": "Topic", "attrs": {}
                            })
                        self.state.graph_data["edges"].append({
                            "source": agent_id, "target": topic, "relation": "OPINION",
                            "attrs": {"weight": op}
                        })

            except Exception as e:
                logger.warning("Unexpected error for agent %d: %s", idx, e)

        logger.info(
            "Created %d valid agents (%d via LLM, %d procedural fallback)",
            len(self.state.agents), llm_ok, fallback_ok
        )

    async def inject_event(self, event: dict) -> None:
        """Inject a global event directly into the current tick."""
        logger.info("Injecting global event: %s", event.get("event_name"))
        self.state.global_events.append(event)
        
        # Distribute into episodic memory of all agents immediately
        desc = f"GLOBAL EVENT: {event.get('description')}"
        async def _remember(agent_id: str):
            mem = self.memory.get_or_create(agent_id)
            await mem.remember("episodic", desc, tick=self.state.tick)
            
        await asyncio.gather(*[_remember(aid) for aid in self.state.agents])

    async def _agent_step(self, agent: Agent) -> dict:
        """Cognitive loop for a single agent during a tick."""
        mem = self.memory.get_or_create(agent.id)
        
        # 1. Recall recent memories (events and interactions)
        episodes = await mem.recall("episodic", "What happened recently?", n_results=3)
        
        # 2. React to world
        sys = agent.system_prompt()
        usr = AGENT_REACT_PROMPT.format(
            events="\n".join([e.get("description", "") for e in self.state.global_events]),
            memories="\n".join(episodes)
        )
        
        reaction = await self.llm.call("agent_react", system=sys, user=usr)
        
        # 3. Apply updates internally
        if isinstance(reaction, dict):
            # Update opinions
            updates = reaction.get("opinion_updates", {})
            for topic, val in updates.items():
                if topic in agent.opinions:
                    try:
                        new_val = float(val)
                        # Smooth transition (alpha filter)
                        agent.opinions[topic] = (0.7 * agent.opinions[topic]) + (0.3 * max(-1.0, min(1.0, new_val)))
                    except (ValueError, TypeError):
                        pass

            # Update timeline tracker
            agent.record_opinion_snapshot(self.state.tick)

            # Record internal thought purely to memory
            internal = reaction.get("internal_reaction")
            if internal:
                await mem.remember("semantic", f"[Tick {self.state.tick}] Thought: {internal}")
                
        return reaction

    async def tick(self) -> None:
        """Advance the simulation by one time step."""
        self.state.tick += 1
        logger.info("--- Starting Tick %d ---", self.state.tick)
        
        start_t = time.time()
        
        public_statements = []

        async def _step_and_emit(agent: Agent):
            try:
                reaction = await self._agent_step(agent)
                stmt_data = None

                if isinstance(reaction, dict):
                    stmt = reaction.get("public_statement")
                    internal = reaction.get("internal_reaction")
                    
                    if stmt:
                        stmt_data = {
                            "agent_id": agent.id, "name": agent.name,
                            "statement": stmt, "action_type": "CREATE_POST", "tick": self.state.tick
                        }
                        
                        # Graph updates for the statement
                        topic = self.state.active_topics[0] if self.state.active_topics else "General public"
                        self.state.graph_data["edges"].append({
                            "source": agent.id,
                            "target": topic,
                            "relation": "POSTED",
                            "attrs": {"tick": self.state.tick, "fact": stmt[:30] + "..."}
                        })
                        # Keep edge count reasonable
                        if len(self.state.graph_data["edges"]) > 200:
                            self.state.graph_data["edges"] = self.state.graph_data["edges"][-150:]
                            
                    elif internal:
                        stmt_data = {
                            "agent_id": agent.id, "name": agent.name,
                            "statement": f"(Thought) {internal}", "action_type": "DO_NOTHING", "tick": self.state.tick
                        }
                
                # If neither statement nor dict response exists, emit fallback so UI isn't dead
                if not stmt_data:
                    stmt_data = {
                        "agent_id": agent.id, "name": agent.name,
                        "statement": "Action skipped locally.", "action_type": "DO_NOTHING", "tick": self.state.tick
                    }

                # Stream the event immediately to the websocket listeners
                for cb in self._on_event:
                    cb(stmt_data)
                
                return stmt_data
            except Exception as e:
                logger.error("Agent %s step failed: %s", agent.id, e)
                # Emit explicit failure event to unstuck frontend
                stmt_data = {
                    "agent_id": agent.id, "name": agent.name,
                    "statement": f"Skipped (Error: {str(e)})", "action_type": "DO_NOTHING", "tick": self.state.tick
                }
                for cb in self._on_event:
                    cb(stmt_data)
                return e

        # Step 1: All agents react concurrently, streaming responses as they finish
        tasks = [_step_and_emit(agent) for agent in self.state.agents.values()]
        reactions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process reactions to consolidate successful public statements
        for reaction in reactions:
            if isinstance(reaction, dict) and "agent_id" in reaction and "statement" in reaction:
                public_statements.append(reaction)
                        
        # Step 2: Distribute selected public statements to other agents' memories
        # To avoid O(N^2) memory explosion, we select the top statements based on agent influence
        if public_statements:
            # Sort by influencer score roughly
            top_stmts = sorted(
                public_statements, 
                key=lambda x: self.state.agents[x["agent_id"]].influence_score, 
                reverse=True
            )[:10]  # Max 10 broadcasted statements per tick
            
            async def _distribute(target_id: str):
                mem = self.memory.get_or_create(target_id)
                for stmt in top_stmts:
                    if stmt["agent_id"] != target_id:
                        text = f"{stmt['name']} publicly stated: {stmt['statement']}"
                        await mem.remember("social", text, tick=self.state.tick)
            
            await asyncio.gather(*[_distribute(aid) for aid in self.state.agents])

        # Step 3: Run analytics on generic properties (handled externally or lightly here)
        # Clear events for next tick
        self.state.global_events = []
        
        logger.info("Tick %d complete in %.2fs", self.state.tick, time.time() - start_t)

        # Trigger tick callbacks
        for cb in self._on_tick:
            cb(self.state)

    async def run(self, ticks: int) -> None:
        """Run the simulation for a structured number of ticks."""
        for _ in range(ticks):
            await self.tick()
