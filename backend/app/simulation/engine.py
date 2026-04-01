"""
SwarmIQ — Core Simulation Engine
Asynchronous event loop for ticking the swarm simulation.
"""

import asyncio
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

    async def initialize_agents(self, count: int) -> None:
        """Generate N agents using LLM and create their memories."""
        logger.info("Initializing %d agents for sim %s", count, self.sim_id)
        
        # We batch these requests
        prompts = []
        for i in range(count):
            arch = random.choice(self.archetypes)
            sys = f"You are creating a character profile for the {arch} archetype."
            usr = AGENT_GENERATE_PROMPT.format(
                seed_summary=self.state.graph.graph.get("summary", ""),
                topics=", ".join(self.state.active_topics)
            )
            prompts.append({"system": sys, "user": usr})

        # Run concurrent Ollama calls (limited by max_concurrent in router)
        results = await self.llm.ollama.batch_complete(prompts)
        
        for idx, result_json in enumerate(results):
            try:
                import json
                data = json.loads(result_json) if isinstance(result_json, str) else result_json
                
                # Assign ID
                agent_id = f"agent_{self.sim_id}_{idx}"
                data["id"] = agent_id
                
                # Personality
                pers = BigFivePersonality.for_occupation(data.get("occupation", "Worker"))
                data["personality"] = pers.to_dict()
                
                # Topics
                opinions = data.get("initial_opinions", {})
                for t in self.state.active_topics:
                    if t not in opinions:
                        opinions[t] = random.gauss(self.state.graph.graph.get("initial_sentiments", {}).get(t, 0.0), 0.2)
                data["opinions"] = opinions
                
                agent = Agent.from_dict(data)
                self.state.agents[agent_id] = agent
                
                # Create memory collection
                mem = self.memory.get_or_create(agent_id)
                await mem.remember("semantic", f"I am {agent.name}, a {agent.age} year old {agent.occupation}. {agent.background}")
                
            except Exception as e:
                logger.warning("Failed to parse generating agent %d: %s", idx, e)

        logger.info("Created %d valid agents", len(self.state.agents))

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

        # Step 1: All agents react concurrently
        tasks = [self._agent_step(agent) for agent in self.state.agents.values()]
        reactions = await asyncio.gather(*tasks, return_exceptions=True)
        
        public_statements = []
        
        # Process reactions for interactions and broadcasts
        for agent_id, reaction in zip(self.state.agents.keys(), reactions):
            if isinstance(reaction, Exception):
                logger.error("Agent %s step failed: %s", agent_id, reaction)
                continue
                
            if isinstance(reaction, dict):
                stmt = reaction.get("public_statement")
                if stmt:
                    agent = self.state.agents[agent_id]
                    public_statements.append({
                        "agent_id": agent.id,
                        "name": agent.name,
                        "statement": stmt,
                        "tick": self.state.tick
                    })

                    # Broadcast to callbacks immediately 
                    for cb in self._on_event:
                        cb(public_statements[-1])
                        
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
