# SwarmIQ

SwarmIQ is a social simulation engine leveraging local Large Language Models (LLMs) and GraphRAG to simulate human interactions, opinion dynamics, and predictive analysis based on initial inputs. 

## How It Works: End-to-End Workflow

The architecture of SwarmIQ is designed to flow seamlessly from initial document ingestion to an active, tick-based social simulation, concluding with a final comprehensive prediction report and an interactive chatbot. 

### Phase 1: Inputs & Generative Ingestion (GraphRAG)
**Key Files:** `backend/app/api/graph.py`, `backend/app/ingestion/document_processor.py`, `backend/app/graphrag/entity_extractor.py`, `backend/app/graphrag/world_builder.py`

1. **Providing the Inputs (`api/graph.py`)**
   The process begins by establishing the parameters of the simulation through user-uploaded source documents (PDFs, Markdown files) and a **Prediction Goal** (the prompt).
   
2. **Document Extraction & Local Graph Building (`document_processor.py`, `entity_extractor.py`)**
   * **Text Processing:** The backend systematically parses the uploaded files using the `DocumentProcessor`, converting them into plain text.
   * **Entity Extraction:** The text is fed into the `EntityExtractor`. Instead of relying on traditional keyword searches or paid external APIs, SwarmIQ uses a local LLM via Ollama to perform **Local GraphRAG**. The LLM reads the text and extracts entities (people, organizations, concepts) and the relationships between them.
   * **NetworkX Compilation:** These entities and relationships are structured directly into a local Python `NetworkX` graph representation encompassing nodes and edges.

3. **World Context Seeding (`world_builder.py`)**
   The `WorldBuilder` combines the prediction goal with the extracted graph data to identify the **Active Topics** of the simulation. It compiles this into a JSON "seed" file containing agent archetypes, default sentiments, and structural topic data.

---

### Phase 2: Agent Generation & Instantiation
**Key Files:** `backend/app/api/simulation.py`, `backend/app/simulation/engine.py`, `backend/app/simulation/agent.py`, `backend/app/memory/memory_manager.py`

When a simulation starts (via `/start` in `api/simulation.py`), the **Simulation Engine** (`engine.py`) boots up and populates the world context generated in Phase 1.

* **LLM Persona Creation:** For the requested number of agents, the engine constructs distinct prompts requesting the local LLM to generate character profiles. The generated `Agent` objects have names, ages, physical backgrounds, occupations, and a simulated "Big Five" personality.
* **Initial Opinions:** Each agent is assigned an initial numerical opinion value (ranging from -1.0 to 1.0) on the active topics using a Gaussian distribution based on the world context's default sentiments.
* **Memory Initialization (`memory_manager.py`):** Agents have complex memory structures. Upon creation, their persona identity is committed to their `semantic` memory bank so they remember who they are as the simulation runs.
* **Procedural Fallbacks:** If the LLM goes offline or fails to respond, SwarmIQ features a built-in heuristic fallback engine (`_generate_fallback_agent()`). It will automatically generate randomized names, occupations, and procedurally calculated opinions mathematically, ensuring the simulation never crashes.

---

### Phase 3: The Engine's Cognitive Loop
**Key Files:** `backend/app/simulation/engine.py`, `backend/app/simulation/world.py`

The engine operates asynchronously on discrete time loops called **Ticks** handled by `tick()`. Under the hood, for every single tick, every agent undergoes a dedicated "Cognitive Loop" inside `_agent_step()`:

1. **Memory Recall:** The agent dips into its `episodic` and `social` memory buffers, asking itself, *"What happened recently?"* It pulls up previous interactions or recent global events to form immediate context.
2. **Evaluating & Reacting:** A complex prompt containing the agent's identity, the active topics, and their recent memories is sent to the LLM router. The LLM evaluates how the agent digests this information based on their personality. 
3. **Opinion Shifting:** The agent's numerical opinion on topics dynamically slides closer to or further away from extreme ends (1.0/-1.0) based on how persuasive the recent events were to their specific archetype.
4. **Action & Broadcast:** The agent resolves to either hold an "internal reaction" (kept to themselves in semantic memory) or issue a "public statement."
5. **Viral Propagation:** To maintain computational efficiency, the engine sorts all public statements by an agent's established influence score. The top statements are propagated recursively out into the `social` memory banks of *other* agents to be recalled in the next tick. The frontend visualization updates with these new edge interactions in real time.
6. **God Mode (Runtime Injection in `api/simulation.py`):** During these ticks, the `/inject` API can be hit to synthesize a sudden news event immediately into all agents' episodic memories, forcefully disrupting the flow.

---

### Phase 4: Snapshot Analytics & The Report
**Key Files:** `backend/app/api/report.py`, `backend/app/simulation/analytics.py`, `backend/app/db/snapshot_store.py`

After the simulation finishes its allotted ticks, the engine takes a final freeze-frame **Snapshot** of the `WorldState` and saves it via the `SnapshotStore`.

* **Heuristic Overlays (`analytics.py`):** The backend analytics scanner sweeps through the agents' final memory blocks and numerical opinions. It uses mathematical calculations to detect "Echo Chambers" and measures dynamic influence shifts.
* **LLM Synthesis (`api/report.py` - `generate_report()`):** The system bundles the calculated opinion averages, agent behavior samples, the echo chambers detected, and the original goal, pushing it to the LLM one final time.
* **Markdown Output:** The LLM synthesizes this structured data into a highly readable predictive Markdown report outlining what the "Swarm" of characters thought, predicted, or argued over regarding your core problem.

---

### Phase 5: The Final Interactive Chatbot
**Key Files:** `backend/app/api/report.py` (specifically `chat_with_report()`)

Once the report generation is complete, the interactive chatbot (`/chat` endpoint) becomes available.

Instead of re-querying the massive, token-heavy graph directly every time the user asks a question, the chatbot functions as a **Focused Context RAG**.
It loads the final calculated Markdown Report and acts as an "Expert Analyst." When a user asks a question about the simulation results, it uses the report text as its direct context window. This ensures concise, factual responses backed exclusively by the socio-dynamic data printed in the report.

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/c64d6d8a-04e9-4302-b11e-f4bfea5c81f6" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/1ff4197a-9113-40ee-b8bf-b6e28be96e29" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/89c669c8-4e64-42df-9ca7-97ecdb29172d" />


