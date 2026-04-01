"""
SwarmIQ — Prompt templates for simulation LLM calls.
All prompts are in English. Keep prompts here; no hardcoded strings in business logic.
"""

AGENT_REACT_PROMPT = """\
The following events are happening in your world right now:
{events}

Your recent memories:
{memories}

Based on your personality and background, react authentically. Return JSON only:
{{
  "internal_reaction": "your private thoughts (1-2 sentences)",
  "opinion_updates": {{
    "topic_name": new_float_value
  }},
  "public_statement": "what you say out loud, or null if you stay silent",
  "wants_to_talk_to": ["agent_id_1"]
}}

Rules:
- opinion values must be floats between -1.0 and 1.0
- wants_to_talk_to must contain 0-2 agent IDs from the provided list
- public_statement can be null
"""

AGENT_INTERACT_PROMPT = """\
{agent_a_name} says to {agent_b_name}: "{statement}"

{agent_b_name}'s profile:
{agent_b_system}

How does {agent_b_name} respond? Return JSON only:
{{
  "response": "what {agent_b_name} says back",
  "opinion_shift": {{"topic": delta_float}},
  "trust_delta": 0.0
}}

Rules:
- opinion_shift deltas are small floats between -0.2 and 0.2
- trust_delta is between -0.1 and 0.1
"""

AGENT_GENERATE_PROMPT = """\
World context: {seed_summary}

Active topics in this world: {topics}

Generate a believable, diverse person who lives in this world. Return JSON only:
{{
  "name": "Full Name",
  "age": integer,
  "occupation": "job title",
  "background": "2-3 sentence backstory that connects them to the world context",
  "initial_opinions": {{
    "topic_name": float_between_neg1_and_1
  }}
}}

Make the person feel realistic and distinct. Vary age, occupation, and perspective.
"""

AGENT_UPDATE_BELIEF_PROMPT = """\
You are {name}, a {occupation}.
After the events of this simulation tick, update your beliefs.

What happened this tick:
{tick_summary}

Your current opinions: {opinions}

Return JSON only:
{{
  "updated_opinions": {{"topic": new_float}},
  "new_belief": "a short statement about something you now believe"
}}
"""

ENTITY_EXTRACT_PROMPT = """\
Extract entities and relationships from the following text chunk. Return JSON only:
{{
  "entities": [
    {{"name": "string", "type": "Person|Organization|Location|Policy|Event|Concept", "description": "brief desc"}}
  ],
  "relationships": [
    {{"source": "entity name", "target": "entity name", "relation": "relationship type", "strength": 0.0}}
  ]
}}

Text:
{text_chunk}

Focus on concrete, named entities. Strength is 0.0-1.0.
"""

WORLD_CONTEXT_PROMPT = """\
You are building a social simulation world from a knowledge graph.

Key entities found: {entities}
Key relationships: {relationships}
Prediction goal: {goal}

Generate the world context. Return JSON only:
{{
  "summary": "3-5 sentence description of this world and its tensions",
  "active_topics": ["topic1", "topic2"],
  "initial_sentiments": {{"topic": baseline_float}},
  "agent_archetypes": ["archetype1"],
  "key_entities": ["entity1"]
}}

active_topics should be 5-10 discussion topics relevant to the prediction goal.
initial_sentiments should reflect the general public mood on each topic.
"""
