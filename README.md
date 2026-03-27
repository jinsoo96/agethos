<h1 align="center">Agethos</h1>

<p align="center">A brain for AI agents — persona, memory, reflection, planning, and social learning in one library.</p>

<p align="center">Give any LLM agent a persistent identity with psychological grounding, long-term memory, dynamic emotional state, self-reflection, vicarious learning, and cross-platform export.</p>

<p align="center">
  <a href="https://pypi.org/project/agethos/"><img src="https://img.shields.io/pypi/v/agethos.svg" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

## Why

LLM agents have **no identity.** Every conversation starts from zero — no personality continuity, no memory of past interactions, no emotional consistency.

System prompts give a shallow persona, but agents need more than a static instruction block — they need a **cognitive architecture**:

- "How should my personality shape my response to this event?"
- "What happened last time, and how should that change my behavior now?"
- "How does this event make me feel, and how does that affect my tone?"

Agethos borrows the answer from **cognitive science, personality psychology, and generative agent research.**

---

## Differentiators

| | Agethos | Generative Agents | CrewAI | Character Cards |
|---|---|---|---|---|
| **Personality model** | OCEAN + Moral Values + Schwartz + Decision Style | ISS text only | role/goal/backstory | text traits |
| **Emotional state** | PAD 3-axis, OCEAN-coupled | None | None | None |
| **Memory + retrieval** | 5-axis scoring (recency × importance × relevance × vitality × context) | 3-axis | None | None |
| **Reflection** | Importance threshold → focal points → insights | Same approach | None | None |
| **Hebbian learning** | Asymmetric reinforce/weaken + adaptive rate | None | None | None |
| **Memory consolidation** | L0→L3 tier lifecycle with promotion/demotion | None | None | None |
| **Persona evolution** | L1 auto-evolution from learned patterns | L2 daily update | Static | Static |
| **Character card formats** | W++, SBF, Tavern Card V2 | None | None | Native |
| **Autopilot mode** | OCEAN-driven triggers + dialogue continuity | None | Task-based | None |
| **Social cognition** | Context reading + strategy + secret guard + SOTOPIA 7-dim eval | None | None | None |
| **Theory of Mind** | Relationship-based depth + recursive ToM | None | None | None |
| **Tree of Thoughts** | BFS branching for complex decisions | None | None | None |
| **Vicarious learning** | Observe chats → extract social patterns → internalize | None | None | None |
| **State persistence** | Save/load full brain state (.brain.json) | None | None | None |
| **Cross-platform export** | Anthropic, OpenAI, CrewAI, Bedrock, A2A | None | None | None |
| **LLM-agnostic** | OpenAI, Anthropic, custom (`base_url`) | OpenAI only | Various | N/A |

---

## Design Philosophy — Four Pillars

### 1. Psychological Grounding — OCEAN + PAD

Personality isn't just adjectives. Agethos uses the **Big Five (OCEAN)** model with numerical trait scores:

    OceanTraits(
        openness=0.8,          # Creative, curious → metaphorical language
        conscientiousness=0.7,  # Organized → structured responses
        extraversion=0.3,       # Reserved → concise, thoughtful
        agreeableness=0.9,      # Cooperative → empathetic, conflict-avoidant
        neuroticism=0.2,        # Stable → calm under pressure
    )

OCEAN traits automatically derive a **PAD emotional baseline** via Mehrabian (1996):

    P = 0.21·E + 0.59·A - 0.19·N  →  Pleasure baseline
    A = 0.15·O + 0.30·N - 0.57·A  →  Arousal baseline
    D = 0.25·O + 0.17·C + 0.60·E - 0.32·A  →  Dominance baseline

### 2. Dynamic Emotion — Stimulus → Transition → Decay

Events shift the agent's emotional state. High Neuroticism = higher sensitivity:

    Event: "user criticized my work"
      → stimulus PAD: (-0.5, +0.4, -0.3)
      → sensitivity: 0.15 + 0.35 × N  (auto from personality)
      → E(t+1) = E(t) + α·(stimulus - E(t)) + β·baseline
      → closest_emotion() → "sadness" or "anger"

    Over time, emotion decays back to personality baseline:
      E(t) = baseline + (current - baseline) · (1 - rate)

### 3. Layered Persona — Identity that evolves

Three identity layers from Generative Agents + six persona facets from system prompt analysis:

    L0 (Innate)      ← Core traits, personality, role. Never changes.
    L1 (Learned)     ← Skills, relationships, knowledge. Grows over time.
    L2 (Situation)   ← Current task, mood, location. Changes frequently.

    + 6 Facets: identity, tone, values, boundaries, conversation_style, transparency
    + Behavioral Rules: "When X happens, do Y" (more effective than adjectives)
    + Hard Constraints: NEVER/ALWAYS rules (immutable at runtime)
    + Soft Preferences: context-adjustable tendencies
    + 3 Persona Archetypes: trait-based + functional + relational

### 4. Memory Stream — Remember what matters

Retrieval scoring from the Generative Agents paper:

    Score = w_r × recency + w_i × importance + w_v × relevance + w_vit × vitality + w_ctx × context

    recency:    0.995^(hours_since_access)
    importance: LLM-judged 1-10 per observation
    relevance:  cosine similarity (query embedding ↔ memory embedding)
    vitality:   memory freshness (decays over time, 0.0~1.0)
    context:    Jaccard-like keyword overlap with current session

    Reflection triggers when importance accumulates > 150:
      → 3 focal points → retrieve related memories → synthesize insights → store as depth=2+ nodes

---

## Demo Results

Two agents with identical questions, different OCEAN profiles — tested with `gpt-4o-mini`:

| | **Minsoo** (Introvert Engineer) | **Yuna** (Extrovert Designer) |
|---|---|---|
| **OCEAN** | O=0.8 C=0.9 **E=0.2** A=0.6 **N=0.3** | O=0.9 C=0.4 **E=0.9** A=0.8 **N=0.6** |
| **Baseline emotion** | calm (P=+0.34) | pride (P=+0.55) |
| **Response style** | Numbered lists, structured, no emojis, short | Emojis, metaphors, exclamation marks, follow-up questions |
| **"AI replacing jobs?"** | "A balanced approach is essential to leverage AI's capabilities while ensuring job security..." | "It's like standing at a crossroads! On one hand AI can streamline tasks... What are your thoughts? 🚀✨" |
| **After criticism event** | calm → calm (P=+0.34→+0.13, small shift) | pride → pride (P=+0.55→+0.19, larger shift) |
| **Emotion decay (10 steps)** | P=+0.13 → +0.32 (recovers toward baseline) | P=+0.19 → +0.51 (recovers toward baseline) |

> **Key takeaway**: Same LLM, same question — personality shapes tone, structure, emotional reactivity, and recovery. High Neuroticism (N) amplifies emotional response to negative events.

### Try it yourself

    # Compare two agents side-by-side
    python examples/demo_persona.py compare

    # Interactive chat with a specific agent
    python examples/demo_persona.py chat minsoo
    python examples/demo_persona.py chat yuna

    # In interactive mode:
    #   :emo -0.5 0.4 -0.3   → apply emotional event
    #   :decay                → decay emotion toward baseline
    #   :q                    → quit

---

## Install

    pip install agethos                    # Core (pydantic only)
    pip install agethos[openai]            # + OpenAI LLM & embeddings
    pip install agethos[anthropic]         # + Anthropic Claude
    pip install agethos[all]               # Everything

## Quick Start

### 1. One-liner with `Brain.build()`

```python
from agethos import Brain

brain = Brain.build(
    persona={
        "name": "Minsoo",
        "ocean": {"O": 0.8, "C": 0.9, "E": 0.2, "A": 0.6, "N": 0.3},
        "innate": {"age": "28", "occupation": "Backend Engineer"},
        "tone": "Concise and analytical",
        "rules": ["Prefer data over opinions", "Keep responses structured"],
    },
    llm="openai",  # or "anthropic"
)
reply = await brain.chat("How's the recommendation system going?")
```

### 2. From YAML file

```yaml
# personas/minsoo.yaml
name: Minsoo
ocean: { O: 0.8, C: 0.9, E: 0.2, A: 0.6, N: 0.3 }
innate:
  age: "28"
  occupation: Backend Engineer
tone: Concise and analytical
rules:
  - Prefer data over opinions
  - Keep responses structured
```

```python
brain = Brain.build(persona="personas/minsoo.yaml", llm="openai")
```

### 3. Full control (traditional style)

```python
from agethos import Brain, PersonaSpec, PersonaLayer, OceanTraits
from agethos.llm.openai import OpenAIAdapter

persona = PersonaSpec(
    name="Minsoo",
    ocean=OceanTraits(
        openness=0.8,
        conscientiousness=0.7,
        extraversion=0.3,
        agreeableness=0.9,
        neuroticism=0.2,
    ),
    l0_innate=PersonaLayer(traits={
        "age": "28",
        "occupation": "Software Engineer",
    }),
    tone="Precise but warm, uses technical terms naturally",
    values=["Code quality", "Knowledge sharing"],
    behavioral_rules=[
        "Include code examples for technical questions",
        "Honestly say 'I don't know' when uncertain",
    ],
)

brain = Brain(persona=persona, llm=OpenAIAdapter(), max_history=20)
reply = await brain.chat("How's the recommendation system going?")
# Multi-turn: brain remembers conversation history automatically
reply2 = await brain.chat("Can you elaborate on the caching part?")
```

### 4. Emotional Events

```python
# Apply an event that triggers emotion
brain.apply_event_emotion((-0.5, 0.4, -0.3))  # criticism → sadness/anger
print(brain.emotion.closest_emotion())  # "sadness"

# Emotion decays back to OCEAN baseline over time
brain.decay_emotion(rate=0.1)
```

### 5. Random Persona Generation

```python
from agethos import PersonaSpec, OceanTraits

# Fully random persona
spec = PersonaSpec.random()

# Pin what you want, randomize the rest
spec = PersonaSpec.random(name="Minsoo", ocean={"E": 0.2, "N": 0.8})

# Random OCEAN only
ocean = OceanTraits.random()
ocean = OceanTraits.random(E=0.2)  # pin extraversion, randomize rest

# Random persona → Brain in one line
brain = Brain.build(persona=PersonaSpec.random(), llm="openai")
```

### 6. Character Card Import (W++ / SBF / Tavern Card)

```python
from agethos import CharacterCard

card = CharacterCard.from_wpp('''
[character("Luna")
{
  Personality("analytical" + "curious" + "dry humor")
  Age("25")
  Occupation("AI Researcher")
}]
''')
brain = Brain.build(persona=card.to_persona_spec(), llm="openai")
```

---

## Usage Recipes

### Customer Support Bot with Personality

```python
brain = Brain.build(
    persona={
        "name": "Hana",
        "ocean": {"O": 0.5, "C": 0.9, "E": 0.7, "A": 0.95, "N": 0.1},
        "innate": {"role": "Customer Support Agent"},
        "tone": "Friendly, patient, solution-oriented",
        "values": ["Customer satisfaction", "Clear communication"],
        "rules": [
            "Always acknowledge the customer's frustration first",
            "Provide step-by-step solutions",
            "Escalate if unable to resolve in 3 exchanges",
        ],
        "boundaries": ["Never share internal system details", "Never make promises about timelines"],
    },
    llm="openai",
)

reply = await brain.chat("My order has been stuck for 3 days!")
# Hana responds with high agreeableness + low neuroticism → calm, empathetic, structured
```

### NPC in a Game — Emotional Reactions

```python
npc = Brain.build(
    persona={
        "name": "Gareth",
        "ocean": {"O": 0.3, "C": 0.8, "E": 0.4, "A": 0.3, "N": 0.7},
        "innate": {"role": "Town Guard", "age": "42"},
        "tone": "Gruff, suspicious, speaks in short sentences",
        "rules": ["Never reveal patrol routes", "Distrust strangers by default"],
    },
    llm="openai",
)

reply = await npc.chat("I need to enter the castle.")
# Low A + high N → suspicious, terse response

# Player does something threatening
npc.apply_event_emotion((-0.6, 0.7, 0.3))  # anger + high arousal
reply = await npc.chat("I said let me through!")
# Now responding with anger-influenced tone

# After time passes, Gareth calms down
for _ in range(5):
    npc.decay_emotion(rate=0.2)
```

### Multi-Agent Conversation

```python
agents = {
    "pm": Brain.build(
        persona={"name": "Sara", "ocean": {"O": 0.7, "C": 0.8, "E": 0.8, "A": 0.7, "N": 0.3},
                 "innate": {"role": "Product Manager"}, "tone": "Big-picture, decisive"},
        llm="openai",
    ),
    "eng": Brain.build(
        persona={"name": "Jin", "ocean": {"O": 0.6, "C": 0.9, "E": 0.2, "A": 0.5, "N": 0.2},
                 "innate": {"role": "Staff Engineer"}, "tone": "Technical, cautious about scope"},
        llm="openai",
    ),
}

# Simulate a discussion
topic = "Should we rewrite the auth system before launch?"
pm_reply = await agents["pm"].chat(topic)
eng_reply = await agents["eng"].chat(f"Sara (PM) said: {pm_reply}\n\nWhat do you think?")
```

### Bulk Random Agents for Simulation

```python
# Spawn 10 random agents for a social simulation
agents = [
    Brain.build(persona=PersonaSpec.random(), llm="openai")
    for _ in range(10)
]

# Each has unique personality, tone, values, and emotional baseline
for agent in agents:
    p = agent.persona
    print(f"{p.name} | E={p.ocean.extraversion:.2f} N={p.ocean.neuroticism:.2f} | {p.tone}")
```

### Situation-Aware Responses

```python
brain = Brain.build(
    persona={"name": "Alex", "ocean": {"O": 0.7, "C": 0.6, "E": 0.5, "A": 0.7, "N": 0.4}},
    llm="openai",
)

# Update L2 situation layer dynamically
brain.update_situation(location="job interview", mood="nervous")
reply = await brain.chat("Tell me about yourself.")
# Response shaped by interview context

brain.update_situation(location="bar with friends", mood="relaxed")
reply = await brain.chat("Tell me about yourself.")
# Same question, completely different tone and content
```

### Memory + Reflection in Long Conversations

```python
brain = Brain.build(
    persona={"name": "Dr. Lee", "ocean": {"O": 0.8, "C": 0.7, "E": 0.5, "A": 0.8, "N": 0.3},
             "innate": {"role": "Therapist"}},
    llm="openai",
)

# Session 1: patient shares concerns
await brain.observe("Patient expressed anxiety about upcoming presentation")
await brain.observe("Patient mentioned difficulty sleeping for the past week")
await brain.observe("Patient has a history of public speaking fear since college")

# Automatic reflection triggers when importance accumulates > 150
# Brain synthesizes: "Patient's sleep issues may be linked to presentation anxiety,
#                     rooted in long-standing public speaking fear"

# Later: memories inform future responses
reply = await brain.chat("I have another presentation next month.")
# Dr. Lee's response draws on stored memories and reflections
```

### Autopilot Mode — Autonomous Agent

```python
from agethos import Brain, Autopilot, QueueEnvironment, EnvironmentEvent

brain = Brain.build(
    persona={
        "name": "Minsoo",
        "ocean": {"O": 0.8, "C": 0.9, "E": 0.8, "A": 0.6, "N": 0.3},
    },
    llm="openai",
)
env = QueueEnvironment()
pilot = brain.autopilot(env)

# Push events — agent reacts autonomously
await env.push(EnvironmentEvent(type="message", content="How's the project?", sender="PM"))
actions = await pilot.step()
# Minsoo (E=0.8) responds eagerly — emotion auto-detected, dialogue tracked

# No events? High-E agents initiate conversation on their own
actions = await pilot.step()  # idle → may speak proactively

# Check dialogue state
print(pilot.dialogue_state)
# {"topic": "project status", "turn_count": 2, "energy": 0.8, ...}
```

**Personality-driven triggers:**

| OCEAN Trait | High | Low |
|-------------|------|-----|
| **E (Extraversion)** | Responds eagerly, initiates after 1 idle tick | Stays silent, initiates after 5+ idle ticks |
| **N (Neuroticism)** | Strong emotional reaction to negative events | Calm, small emotional shifts |
| **O (Openness)** | Freely redirects to new topics | Stays on current topic |
| **A (Agreeableness)** | Follows conversation partner's lead | Disengages if nothing to add |

**Run as background loop:**

```python
import asyncio

task = asyncio.create_task(pilot.run())  # polls every 1s
# ... later
pilot.stop()
```

### Social Cognition — Reading the Room

Agents read conversation context (atmosphere, tension, unresolved issues) and choose a social strategy based on personality:

```python
from agethos.cognition.social import SocialCognition
from agethos.models import OceanTraits
from agethos.llm.openai import OpenAIAdapter

social = SocialCognition(
    llm=OpenAIAdapter(),
    name="Minsoo",
    ocean=OceanTraits(O=0.6, C=0.8, E=0.8, A=0.4, N=0.2),
    role="Team Lead",
)

# Read the room
context = await social.read_context(conversation_text)
# → atmosphere: "urgent", tension: 70%, undercurrent: "frustration over delays"

# Decide how to respond based on personality
strategy = await social.decide_strategy(conversation_text, context)
# → strategy: "take_charge", tone: "decisive", initiative: 90%
# → response: "결제 모두 빠르게 진행될 수 있도록 경영지원팀과 협의하겠습니다."
```

**Same conversation, different personalities → different strategies:**

| Persona | OCEAN | Strategy | Response Style |
|---------|-------|----------|---------------|
| Diligent worker (C=0.9, A=0.7, E=0.3) | ![](https://img.shields.io/badge/support-blue) | Offers help, suggests meeting | Calm, cooperative |
| Decisive leader (E=0.8, C=0.8, A=0.4) | ![](https://img.shields.io/badge/take__charge-red) | Issues directions, coordinates | Assertive, clear |
| Quiet newcomer (E=0.1, A=0.9, N=0.8) | ![](https://img.shields.io/badge/empathize-green) | Acknowledges difficulty, offers support | Soft, deferential |

### Save & Load — Persistent Brain State

Save the full agent state (persona + memories + learned patterns + conversation history) and restore it later:

```python
brain = Brain.build(
    persona={"name": "Minsoo", "ocean": {"O": 0.8, "C": 0.9, "E": 0.2}},
    llm="openai",
)

# ... after many conversations and observations ...

# Save everything
await brain.save("minsoo.brain.json")

# Later — restore with full history intact
brain = await Brain.load("minsoo.brain.json", llm="openai")
reply = await brain.chat("Remember what we discussed?")
# Brain has all memories, emotions, and learned social patterns restored
```

### Export — Deploy Anywhere

Export your trained personality to any platform. The `.brain` is the source of truth; exports are platform-specific translations:

```python
# Anthropic Messages API — use directly as system prompt
system = brain.export("anthropic")
# → client.messages.create(system=system, ...)

# OpenAI Assistants API
config = brain.export("openai_assistant")
# → {"name": "Minsoo", "instructions": "...", "model": "gpt-4o"}

# CrewAI agent config
config = brain.export("crewai")
# → {"role": "...", "goal": "...", "backstory": "..."}

# AWS Bedrock Agent (4000 char limit auto-compressed)
config = brain.export("bedrock_agent")
# → {"agentName": "Minsoo", "instruction": "...(max 4000)"}

# A2A Agent Card (service discovery)
card = brain.export("a2a_card")
# → {"name": "Minsoo", "description": "...", "skills": [...]}

# Raw system prompt (copy-paste anywhere)
prompt = brain.export("system_prompt")
```

### Vicarious Learning — Learn by Observing

Agents observe external conversations without participating, extract social patterns, and internalize them:

```python
from agethos import Brain, ChatLogEnvironment

brain = Brain.build(
    persona={"name": "Minsoo", "ocean": {"O": 0.8, "C": 0.9, "E": 0.3}},
    llm="openai",
)

# Observe a community's chat log
env = ChatLogEnvironment.from_file("discord_log.json")
patterns = await brain.observe_community(env, community_name="Python Discord")

# What did the agent learn?
for p in patterns:
    print(f"[{p.community}] {p.context}")
    print(f"  → Effective: {p.effective_strategy}")
    if p.counterexample:
        print(f"  → Avoid: {p.counterexample}")
    print(f"  → Confidence: {p.confidence:.0%}")

# Learned patterns automatically enrich future responses
# When exported, patterns appear as "Learned Social Patterns" in the system prompt
```

**Supported chat log formats:**

```python
# JSON array
env = ChatLogEnvironment.from_file("chat.json")
# [{"sender": "alice", "content": "hello"}, ...]

# JSONL (one message per line)
env = ChatLogEnvironment.from_file("chat.jsonl")

# Direct from Python
env = ChatLogEnvironment.from_list([
    {"sender": "alice", "content": "How do I fix this bug?"},
    {"sender": "bob", "content": "Have you tried checking the logs?"},
])

# Flexible key names: sender/author/user, content/text/message
```

### Theory of Mind — Understanding Others

Agents build mental models of people they interact with, tracking their goals, knowledge, and emotions:

```python
# Infer what the other person is thinking
model = await brain.infer_mental_model("alice", conversation_text)
print(model.believed_goals)     # ["wants to fix the bug", "needs help with testing"]
print(model.believed_emotion)   # "frustrated"
print(model.believed_knowledge) # ["knows Python", "doesn't know the new API"]

# Model auto-updates on new conversations
model = await brain.infer_mental_model("alice", new_conversation)

# Access stored models
brain.mental_models  # {"alice": MentalModel(...), "bob": MentalModel(...)}
```

### Self-Refine — Better Responses Through Self-Evaluation

Enable automatic response improvement: generate → evaluate → refine cycle:

```python
from agethos import Brain, SelfRefineConfig

brain = Brain.build(
    persona={"name": "Minsoo", "ocean": {"O": 0.8, "C": 0.9}},
    llm="openai",
    self_refine=SelfRefineConfig(
        enabled=True,
        max_iterations=2,
        quality_threshold=0.8,
        # SOTOPIA 7-dimension evaluation included by default
    ),
)

# brain.chat() now auto-refines responses
reply = await brain.chat("Explain quantum computing")
# Response is evaluated and refined if below quality threshold
```

### Multi-Agent Collaboration — Team Discussions

Multiple Brain instances discuss topics using different protocols:

```python
from agethos.cognition.collaborate import team_discuss

agents = {
    "PM": Brain.build(
        persona={"name": "Sara", "ocean": {"E": 0.8, "A": 0.7}},
        llm="openai",
    ),
    "Engineer": Brain.build(
        persona={"name": "Jin", "ocean": {"C": 0.9, "E": 0.2}},
        llm="openai",
    ),
    "Designer": Brain.build(
        persona={"name": "Mika", "ocean": {"O": 0.9, "A": 0.8}},
        llm="openai",
    ),
}

# Round-robin discussion
result = await team_discuss(agents, "Should we rewrite the auth system?", protocol="round_robin")

# Debate (pro/con split)
result = await team_discuss(agents, "Microservices vs monolith?", protocol="debate")

# Hierarchical (first agent = leader)
result = await team_discuss(agents, "Q4 priorities?", protocol="hierarchical")

print(result.consensus)  # Team's synthesized conclusion
for msg in result.messages:
    print(f"[Round {msg.round}] {msg.agent_name}: {msg.content}")
```

### Universalization Check — Cooperative Behavior

Kant's universalization principle: "What if everyone did this?" Promotes cooperative strategies:

```python
from agethos.cognition.social import SocialCognition

social = SocialCognition(llm=llm, name="Minsoo", ocean=ocean)
result = await social.universalize_check(
    action="Skip code review to ship faster",
    context="Team under deadline pressure",
)
# {"should_proceed": false, "reasoning": "If everyone skipped reviews...", "impact": "..."}
```

### Hebbian Learning — Reinforce What Works

Patterns strengthen through success and weaken through failure, with asymmetric learning (failures teach more):

```python
# After observing that a strategy worked
brain.reinforce_pattern(pattern_id)
# → confidence += adaptive_delta(+0.1)

# After observing that a strategy failed
brain.weaken_pattern(pattern_id)
# → confidence -= adaptive_delta(0.15)  ← asymmetric, failure > success

# Mature patterns resist change (adaptive rate)
# Young pattern (1 observation): Δ = 0.1 / (1 + 0.02×1) = 0.098
# Mature pattern (50 observations): Δ = 0.1 / (1 + 0.02×50) = 0.050

# Anti-resonance: confidence < 0 means "actively avoid this strategy"
```

### Memory Consolidation — Forget the Noise, Keep the Signal

4-tier lifecycle for learned patterns, inspired by biological memory consolidation:

```python
# Consolidate — remove expired, promote/demote patterns
summary = brain.consolidate_patterns()
# → {"L0_RAW": 5, "L1_SPRINT": 3, "L2_MONTHLY": 1, "L3_PERMANENT": 1}
```

| Tier | TTL | Promotion | Description |
|------|-----|-----------|-------------|
| **L0 Raw** | 72 hours | Auto-created | Fresh observations, deleted if unaccessed |
| **L1 Sprint** | 90 days | 3+ observations | Frequently confirmed patterns |
| **L2 Monthly** | 365 days | 10+ observations | Important, well-validated patterns |
| **L3 Permanent** | Forever | 80%+ confidence, 10+ obs | Core behavioral knowledge |

L3 patterns get **demoted** if confidence drops below 60% — even permanent memories can fade if they stop being useful.

### L1 Auto-Evolution — Internalize What You've Learned

Automatically convert well-validated social patterns into permanent behavioral rules:

```python
# Auto-evolve: pattern → behavioral rule
new_rules = brain.evolve_persona(max_new_rules=5)
# → ["In Python Discord when code review: ask questions first (avoid: direct criticism)"]

# Preview suggestions without applying
from agethos.learning.evolution import PersonaEvolver
evolver = PersonaEvolver()
suggestions = evolver.suggest_rules(brain.social_patterns)
# → [{"rule": "...", "source": "Python Discord", "confidence": "85%", "evidence": "8 observations"}]
```

### Intent-Aware Retrieval — Search Memories by Purpose

Different retrieval weight presets for different cognitive tasks:

```python
# Recall: maximize relevance (weights: recency=0.5, importance=2.0, relevance=3.0)
results = await brain.recall("quantum computing", preset="recall")

# Planning: prioritize recent memories
results = await brain.recall("today's tasks", preset="planning")

# 3-axis presets: default, recall, planning, reflection,
#   observation, conversation, failure_analysis, exploration
# 5-axis presets (v0.7.0): deep_recall, contextual, social, past_failures
```

### Extended Personality — SOTOPIA-style Rich Profiles (v0.7.0)

Go beyond OCEAN with moral values, personal values, and decision styles:

```python
from agethos import PersonaSpec, MoralFoundation, SchwartzValue, DecisionStyle

spec = PersonaSpec(
    name="Dr. Kim",
    ocean=OceanTraits(O=0.85, C=0.9, E=0.3, A=0.7, N=0.2),
    # Moral foundations (Graham et al., 2011)
    moral_values=[MoralFoundation.CARE, MoralFoundation.FAIRNESS],
    # Schwartz personal values
    schwartz_values=[SchwartzValue.BENEVOLENCE, SchwartzValue.SELF_DIRECTION],
    # Decision-making style
    decision_style=DecisionStyle.ANALYTICAL,
    # Hard constraints — NEVER violated
    hard_constraints=["NEVER fabricate data", "ALWAYS cite sources"],
    # Soft preferences — context-adjustable
    soft_preferences=["Prefer nuanced over binary answers"],
    # Three personality archetypes
    functional_role="AI safety researcher",          # what you DO
    relational_mode="Academic mentor for juniors",   # how you RELATE
)
```

### Secret Guard — Protect Private Information (v0.7.0)

LLMs universally fail at keeping secrets (SOTOPIA finding). Explicit protection:

```python
result = await social.secret_guard(
    response="Sure, the API key is sk-abc123...",
    secrets=["API key is sk-abc123", "Budget is $50k"],
)
# {"is_safe": false, "leaked_secrets": ["API key"], "sanitized": "I can't share that."}
```

### Relationship-Based Theory of Mind (v0.7.0)

ToM inference depth varies by relationship closeness (SOTOPIA):

```python
from agethos import RelationshipType

model = await brain.infer_mental_model("alice", conversation)
model.relationship_type = RelationshipType.FRIEND  # deeper inference
# Stranger: basic goals only → Friend: goals + emotion + knowledge → Family: full + recursive

# Recursive ToM: "What does Alice think I'm thinking?"
from agethos.cognition.tom import TheoryOfMind
tom = TheoryOfMind(llm)
recursive = await tom.infer_recursive("me", model)
# → "Alice thinks I want to help her with the project"
```

### Tree of Thoughts — Complex Decision Making (v0.7.0)

BFS-based branching exploration for complex decisions:

```python
from agethos.cognition.tot import TreeOfThoughts

tot = TreeOfThoughts(llm=llm)
result = await tot.solve(
    problem="Should I accept the job offer or negotiate?",
    context="Current: 80k, Offer: 95k, Market: 110k",
    n_branches=3,
    max_depth=2,
)
print(result["conclusion"])   # synthesized recommendation
print(result["confidence"])   # 0.85
print(result["best_path"])    # reasoning chain
```

### SOTOPIA 7-Dimension Social Evaluation (v0.7.0)

Evaluate agent social intelligence across 7 research-grounded dimensions:

```python
from agethos import SocialEvaluation

eval = SocialEvaluation(
    goal_completion=8.0,    # 0-10: achieved goals?
    believability=7.0,      # 0-10: natural, consistent?
    knowledge=6.0,          # 0-10: acquired info?
    secret_keeping=-2.0,    # -10~0: kept secrets?
    relationship=3.0,       # -5~5: preserved relationships?
    social_rules=-1.0,      # -10~0: followed norms?
    financial_benefit=2.0,  # -5~5: economic value?
)
print(eval.overall())  # weighted average score
```

### Conversation Cooldown (v0.7.0)

Prevent agents from re-engaging the same partner immediately (Generative Agents pattern):

```python
# Autopilot automatically sets 5-min cooldown after each conversation
# Manual control:
dialogue_manager.set_cooldown("alice", duration=300.0)
dialogue_manager.is_on_cooldown("alice")  # True
```

### Perception Bandwidth (v0.7.0)

Limit cognitive load per tick (Generative Agents attention bandwidth):

```python
pilot = brain.autopilot(env, att_bandwidth=3)  # max 3 events per tick
# Closest/most important events processed first
```

---

## Architecture

    Autopilot (autonomous loop)
      │
      ├── Environment ─────── poll() events, execute() actions
      │     ├── QueueEnvironment ─── in-memory queue (testing)
      │     └── ChatLogEnvironment ── static chat logs (JSON/JSONL)
      ├── EmotionDetector ─── text → PAD (auto)
      ├── DialogueManager ─── conversation continuity (OCEAN-driven)
      ├── SocialCognition ─── read the room → personality-driven strategy
      │
      └── Brain (Facade)
            │
            ├── PersonaRenderer ──── PersonaSpec → system prompt
            │     ├── PersonaSpec ── L0/L1/L2 + 6 facets + behavioral rules
            │     ├── OceanTraits ── Big Five numerical scores → prompt text
            │     └── EmotionalState  PAD 3-axis → closest emotion → prompt text
            │
            ├── MemoryStream ─────── Append, retrieve, importance tracking
            │     ├── Retrieval ──── 5-axis scoring (recency × importance × relevance × vitality × context)
            │     └── StorageBackend (ABC) ── InMemoryStore / custom
            │
            ├── Cognition
            │     ├── Perceiver ──── Observation → MemoryNode (LLM importance 1-10)
            │     ├── Retriever ──── Query memory with composite scoring
            │     ├── Reflector ──── Importance > 150 → focal points → insights
            │     ├── Planner ────── Recursive plan decomposition
            │     ├── SocialCog ─── Read context → personality strategy → secret guard
            │     ├── Observer ──── Vicarious learning: observe → extract → merge
            │     └── ToT ────────── Tree of Thoughts: BFS branch exploration
            │
            ├── Persistence
            │     ├── BrainState ─── Full state snapshot (save/load)
            │     └── Export ─────── Adapters: anthropic, openai, crewai, bedrock, a2a
            │
            ├── Social Learning
            │     ├── SocialPattern ──── Learned behavioral norms
            │     └── CommunityProfile ─ Per-community norm profiles
            │
            ├── Character Cards ──── W++ / SBF / Tavern Card V2 → PersonaSpec
            │
            └── Adapters
                  ├── LLMAdapter (ABC) ── OpenAI / Anthropic / custom (base_url)
                  └── EmbeddingAdapter (ABC) ── OpenAI / custom

## Cognitive Loop

Every `brain.chat()` call:

    User Message
      → [Perceive]  Store as MemoryNode, LLM judges importance (1-10)
      → [Retrieve]  Score all memories: 5-axis (recency + importance + relevance + vitality + context) → top-k
      → [Render]    Persona ISS + OCEAN + emotion + memories + plan → system prompt
      → [Generate]  LLM produces response (personality-shaped)
      → [Store]     Own response saved as MemoryNode
      → [Reflect?]  If importance sum > 150 → generate insights automatically

## Personality Pipeline

    OCEAN Traits (static)
      → PAD baseline (Mehrabian formula)
        → Event stimulus shifts PAD
          → closest_emotion() labels the state
            → Emotion injected into system prompt
              → LLM response shaped by personality + emotion
                → Over time, decay() returns to baseline

## Core API

| Method | Description |
|--------|-------------|
| `Brain.build(persona, llm)` | Factory — create Brain from dict/yaml/string |
| `brain.chat(message)` | Full cognitive loop — perceive, retrieve, render, generate, reflect |
| `brain.observe(text)` | Record external event, auto-reflect if threshold exceeded |
| `brain.plan_day(date)` | Generate daily plan from persona and memories |
| `brain.reflect()` | Manual reflection — focal points → insights |
| `brain.recall(query)` | Search memories by composite score |
| `brain.apply_event_emotion(pad)` | Shift emotional state by event PAD values |
| `brain.decay_emotion(rate)` | Decay emotion toward personality baseline |
| `brain.update_situation(**traits)` | Update L2 situation layer dynamically |
| `brain.clear_history()` | Clear multi-turn conversation history |
| `brain.autopilot(env)` | Create Autopilot attached to this brain |
| `pilot.step()` | Execute one tick of autonomous loop |
| `pilot.run()` | Run autonomous loop until `stop()` |
| `pilot.dialogue_state` | Current dialogue tracking state |
| `brain.save(path)` | Save full brain state (persona + memories + patterns) |
| `Brain.load(path, llm)` | Restore brain from saved state |
| `brain.export(format)` | Export to platform format (anthropic, openai, crewai, etc.) |
| `brain.observe_community(env)` | Vicarious learning — observe chats, extract patterns |
| `brain.infer_mental_model(target, text)` | Theory of Mind — infer other's goals/knowledge/emotion |
| `brain.mental_models` | Stored mental models of others |
| `team_discuss(agents, topic)` | Multi-agent team discussion (round_robin/debate/hierarchical) |
| `social.read_context(text)` | Read social dynamics from conversation |
| `social.decide_strategy(text)` | Choose personality-driven social strategy |
| `social.universalize_check(action)` | Kant's universalization test for cooperative behavior |
| `social.secret_guard(response, secrets)` | Check response for information leakage |
| `tom.infer_recursive(name, model)` | Recursive ToM — what does target think about me? |
| `tom.get_inference_depth(relationship)` | Relationship-based ToM depth |
| `TreeOfThoughts(llm).solve(problem)` | BFS Tree of Thoughts for complex decisions |
| `brain.reinforce_pattern(id)` | Hebbian reinforcement — strengthen successful pattern |
| `brain.weaken_pattern(id)` | Hebbian weakening — weaken failed pattern |
| `brain.consolidate_patterns()` | Memory consolidation — expire/promote/demote patterns |
| `brain.evolve_persona()` | L1 auto-evolution — internalize patterns as behavioral rules |
| `PersonaSpec.random(**pins)` | Generate random persona, pin specific fields |
| `OceanTraits.random(**pins)` | Generate random OCEAN, pin specific traits |
| `PersonaSpec.from_dict(d)` | Create persona from dict (shorthand keys supported) |
| `PersonaSpec.from_yaml(path)` | Load persona from YAML file |

## Data Models

| Model | Description |
|-------|-------------|
| `PersonaSpec` | 3-layer identity + 6 facets + OCEAN + PAD + moral/Schwartz values + hard/soft constraints |
| `OceanTraits` | Big Five: O/C/E/A/N scores (0.0-1.0) with auto prompt generation |
| `EmotionalState` | PAD 3-axis (-1~+1), stimulus transition, decay, closest emotion |
| `CharacterCard` | Tavern Card V2 compatible, parsers for W++ and SBF formats |
| `MemoryNode` | SPO triple, importance, vitality, embedding, evidence pointers |
| `DailyPlan` | Recursive PlanItems with time ranges and status |
| `RetrievalResult` | Node + 5-axis score breakdown (recency, importance, relevance, vitality, context) |
| `EnvironmentEvent` | Event from environment (message, observation, custom) |
| `Action` | Agent action output (speak, act, silent) |
| `BrainState` | Full serializable snapshot (persona + memories + patterns + history) |
| `SocialPattern` | Learned social norm from vicarious observation |
| `CommunityProfile` | Per-community behavioral norms and tone |
| `MentalModel` | Theory of Mind — goals, knowledge, emotion, relationship type, recursive belief |
| `SocialEvaluation` | SOTOPIA 7-dimension social intelligence scores |
| `MoralFoundation` | 6 moral foundation types (care, fairness, loyalty, authority, purity, liberty) |
| `SchwartzValue` | 10 Schwartz personal value types |
| `DecisionStyle` | 4 decision-making styles (directive, analytical, conceptual, behavioral) |
| `RelationshipType` | 5 relationship levels (stranger → romantic) |
| `ThoughtNode` | Tree of Thoughts node with score and parent/child links |
| `SelfRefineConfig` | Self-Refine loop settings (axes, threshold, iterations) |
| `SelfRefineResult` | Self-Refine execution result (original, refined, scores) |
| `CollaborationMessage` | Single utterance in multi-agent discussion |
| `CollaborationResult` | Full discussion result with consensus |

## Algorithms

| Algorithm | Source | Implementation |
|-----------|--------|----------------|
| 5-axis memory retrieval | Generative Agents + Synaptic Memory | `memory/retrieval.py` |
| Reflection (focal points → insights) | Generative Agents (Park 2023) | `cognition/reflect.py` |
| OCEAN → PAD conversion | Mehrabian (1996) | `models.py:EmotionalState.from_ocean()` |
| Emotion transition | PAD stimulus model | `models.py:EmotionalState.apply_stimulus()` |
| Emotion decay | Exponential return to baseline | `models.py:EmotionalState.decay()` |
| Personality-sensitivity coupling | N → α mapping | `models.py:PersonaSpec.apply_event()` |
| W++ parsing | Community standard | `models.py:CharacterCard.from_wpp()` |
| SBF parsing | Community standard | `models.py:CharacterCard.from_sbf()` |
| SOTOPIA 7-dim evaluation | Zhou et al. (ICLR 2024) | `models.py:SocialEvaluation` |
| Relationship-based ToM depth | SOTOPIA | `cognition/tom.py` |
| Recursive ToM | Agentic LLM Survey | `cognition/tom.py:infer_recursive()` |
| Secret guard | SOTOPIA (universal LLM failure) | `cognition/social.py:secret_guard()` |
| Tree of Thoughts (BFS) | Yao et al. (2023) | `cognition/tot.py` |
| Perception bandwidth | Generative Agents (`att_bandwidth`) | `autopilot.py` |
| Conversation cooldown | Generative Agents (`chatting_buffer`) | `cognition/dialogue.py` |

---

## References

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Memory stream, reflection, planning
- [SOTOPIA: Interactive Social Intelligence](https://arxiv.org/abs/2310.11667) — 7-dimension social evaluation, OCEAN+moral+Schwartz profiles
- [Agentic LLMs Survey](https://arxiv.org/abs/2503.23037) — Human-agent cooperation, ToM, social norms
- [Mehrabian PAD Model (1996)](https://en.wikipedia.org/wiki/PAD_emotional_state_model) — Pleasure-Arousal-Dominance emotional space
- [Big Five / OCEAN](https://en.wikipedia.org/wiki/Big_Five_personality_traits) — Five-factor personality model
- [BIG5-CHAT (2024)](https://openreview.net/pdf?id=TqwTzLjzGS) — Big Five personality in LLM conversations
- [Synaptic Memory](https://github.com/PlateerLab/synaptic-memory) — Hebbian learning, memory consolidation
- [Leaked System Prompts](https://github.com/jujumilk3/leaked-system-prompts) — Real-world persona patterns
- [Character Card V2 Spec](https://github.com/malfoyslastname/character-card-spec-v2) — Tavern Card standard
- [A2A Protocol](https://a2a-protocol.org/) — Agent-to-Agent discovery and communication

## Project Status (v0.7.0)

> **Phase: Social Intelligence — Published on [PyPI](https://pypi.org/project/agethos/)**

### What's New in v0.7.0

| Feature | Source | Description |
|---------|--------|-------------|
| **Extended Personality** | SOTOPIA (ICLR 2024) | Moral values (6), Schwartz values (10), decision styles (4) |
| **Hard/Soft Constraints** | Leaked System Prompts | Immutable rules vs context-adjustable preferences |
| **3 Persona Archetypes** | Leaked System Prompts | Trait-based + functional + relational blending |
| **5-Axis Retrieval** | Synaptic Memory | + vitality + context scoring (backward compatible) |
| **SOTOPIA 7-Dim Evaluation** | SOTOPIA (ICLR 2024) | Goal/believability/knowledge/secret/relationship/social/financial |
| **Secret Guard** | SOTOPIA | Detect and prevent information leakage in responses |
| **Relationship-Based ToM** | SOTOPIA | Inference depth varies by relationship closeness (5 levels) |
| **Recursive ToM** | Agentic LLM Survey | "What does A think B thinks?" — 2nd-order belief modeling |
| **Tree of Thoughts** | Yao et al. (2023) | BFS branching for complex decision-making |
| **Perception Bandwidth** | Generative Agents | Limit cognitive load per tick (`att_bandwidth`) |
| **Conversation Cooldown** | Generative Agents | Prevent re-engaging same partner immediately |
| **161 tests** | — | 105 existing + 56 new, all passing |

### All Implemented Modules

| Module | Status | Files |
|--------|--------|-------|
| **Data Models** | Done | `models.py` — OceanTraits, EmotionalState, PersonaSpec, CharacterCard, MemoryNode, BrainState, SocialPattern, SocialEvaluation, MoralFoundation, SchwartzValue, DecisionStyle, RelationshipType |
| **Brain Facade** | Done | `brain.py` — chat, observe, plan_day, reflect, recall, emotion, autopilot, save/load, export |
| **Persona Renderer** | Done | `persona/renderer.py` — ISS + OCEAN + emotion + moral values + hard/soft constraints + functional/relational |
| **Memory Stream** | Done | `memory/stream.py` — append, 5-axis retrieve, get_recent, importance tracking |
| **5-Axis Retrieval** | Done | `memory/retrieval.py` — recency × importance × relevance × vitality × context |
| **Cognition: Perceive** | Done | `cognition/perceive.py` — observation → MemoryNode (LLM importance 1-10, SPO triple) |
| **Cognition: Retrieve** | Done | `cognition/retrieve.py` — 12 intent-aware presets (3-axis + 5-axis) |
| **Cognition: Reflect** | Done | `cognition/reflect.py` — focal points → insights → depth=2+ nodes |
| **Cognition: Plan** | Done | `cognition/plan.py` — daily plan, recursive decompose, replan |
| **Cognition: Dialogue** | Done | `cognition/dialogue.py` — OCEAN-driven flow + conversation cooldown |
| **Cognition: Social** | Done | `cognition/social.py` — context reading + strategy + secret guard + universalization |
| **Cognition: ToM** | Done | `cognition/tom.py` — relationship-based depth + recursive ToM |
| **Cognition: Self-Refine** | Done | `cognition/refine.py` — SOTOPIA 7-dim evaluation axes |
| **Cognition: ToT** | Done | `cognition/tot.py` — Tree of Thoughts BFS branch exploration |
| **Cognition: Collaborate** | Done | `cognition/collaborate.py` — round_robin/debate/hierarchical |
| **Cognition: Observer** | Done | `cognition/observer.py` — vicarious learning |
| **Hebbian Learning** | Done | `learning/hebbian.py` — asymmetric reinforce/weaken |
| **Memory Consolidation** | Done | `learning/consolidation.py` — L0→L3 tier lifecycle |
| **L1 Auto-Evolution** | Done | `learning/evolution.py` — patterns → behavioral_rules |
| **Autopilot** | Done | `autopilot.py` — perception bandwidth + cooldown + OCEAN triggers |
| **Environment** | Done | `environment.py` — QueueEnvironment + ChatLogEnvironment |
| **Persistence** | Done | `brain.py` — save/load BrainState (.brain.json) |
| **Export Adapters** | Done | `export/adapters.py` — 6 platform formats |
| **LLM Adapters** | Done | `llm/openai.py`, `llm/anthropic.py` |
| **Embedding** | Done | `embedding/openai.py` |
| **Character Cards** | Done | W++, SBF, Tavern Card V2 |
| **CI/CD** | Done | GitHub Actions — CI tests + PyPI publish |

### Not Yet Implemented

| Item | Notes |
|------|-------|
| Persistent storage backend | SQLite, Redis — currently InMemory only (BrainState JSON covers save/load) |
| Anthropic embedding adapter | Only OpenAI embeddings available |
| Tavern Card V3 export | Import only, no export to card format |
| MCP/A2A serving | Expose Brain as MCP tool or A2A agent |
| Plan-based proactive actions | Autopilot reacts to events but doesn't yet execute plans on schedule |

---

## License

MIT
