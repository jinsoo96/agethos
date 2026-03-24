<h1 align="center">Agethos</h1>

<p align="center">A brain for AI agents — persona, memory, reflection, and planning in one library.</p>

<p align="center">Give any LLM agent a persistent identity with psychological grounding, long-term memory with retrieval scoring, dynamic emotional state, self-reflection, and daily planning.<br>Inspired by <a href="https://arxiv.org/abs/2304.03442">Generative Agents</a>, <a href="https://github.com/PlateerLab/synaptic-memory">Synaptic Memory</a>, and cognitive science.</p>

<p align="center">
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
| **Personality model** | OCEAN (Big Five) numerical | ISS text only | role/goal/backstory | text traits |
| **Emotional state** | PAD 3-axis, OCEAN-coupled | None | None | None |
| **Memory + retrieval** | recency × importance × relevance | Same approach | None | None |
| **Reflection** | Importance threshold → focal points → insights | Same approach | None | None |
| **Persona evolution** | L2 dynamic + emotion drift | L2 daily update | Static | Static |
| **Character card formats** | W++, SBF, Tavern Card V2 | None | None | Native |
| **LLM-agnostic** | OpenAI, Anthropic, custom | OpenAI only | Various | N/A |

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

### 4. Memory Stream — Remember what matters

Retrieval scoring from the Generative Agents paper:

    Score = w_r × recency + w_i × importance + w_v × relevance

    recency:    0.995^(hours_since_access)
    importance: LLM-judged 1-10 per observation
    relevance:  cosine similarity (query embedding ↔ memory embedding)

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

### 2. Emotional Events

```python
from agethos.models import EmotionalState

# Apply an event that triggers emotion
brain.apply_event_emotion((-0.5, 0.4, -0.3))  # criticism → sadness/anger
print(brain.emotion.closest_emotion())  # "sadness"
print(brain.emotion.to_prompt())  # "Current emotional state: sadness (P=-0.12, A=+0.15, D=+0.02)"

# Emotion decays back to OCEAN baseline over time
brain.decay_emotion(rate=0.1)
```

### 3. Character Card Import (W++ / SBF / Tavern Card)

```python
from agethos import CharacterCard

# From W++ format
card = CharacterCard.from_wpp('''
[character("Luna")
{
  Personality("analytical" + "curious" + "dry humor")
  Age("25")
  Occupation("AI Researcher")
  Speech("precise" + "uses scientific metaphors")
}]
''')
persona = card.to_persona_spec()
brain = Brain(persona=persona, llm=OpenAIAdapter())

# From SBF format
card = CharacterCard.from_sbf('''
[character: Marcus;
personality: stoic, loyal, pragmatic;
occupation: mercenary captain;
speech: blunt, few words, deep voice]
''')

# From Tavern Card V2 JSON
import json
card = CharacterCard(**json.loads(tavern_card_json)["data"])
```

### 4. Full Cognitive Loop

```python
import asyncio

async def main():
    brain = Brain(persona=persona, llm=OpenAIAdapter())

    # Chat — perceive → retrieve → render(persona+emotion+memories) → generate
    reply = await brain.chat("Hello!")

    # Observe — record events, auto-reflect when importance > 150
    await brain.observe("Team meeting: deadline moved to next week")

    # Apply emotional impact of the event
    brain.apply_event_emotion((-0.3, 0.5, -0.2))  # stress

    # Plan — generate daily plan from persona + memories
    plan = await brain.plan_day("2026-03-25", context="Deadline moved up")

    # Recall — search memories by composite score
    results = await brain.recall("deadline discussions")

    # Reflect — manual trigger
    insights = await brain.reflect()

asyncio.run(main())
```

---

## Architecture

    Brain (Facade)
      │
      ├── PersonaRenderer ──── PersonaSpec → system prompt
      │     ├── PersonaSpec ── L0/L1/L2 + 6 facets + behavioral rules
      │     ├── OceanTraits ── Big Five numerical scores → prompt text
      │     └── EmotionalState  PAD 3-axis → closest emotion → prompt text
      │
      ├── MemoryStream ─────── Append, retrieve, importance tracking
      │     ├── Retrieval ──── recency × importance × relevance scoring
      │     └── StorageBackend (ABC) ── InMemoryStore / custom
      │
      ├── Cognition
      │     ├── Perceiver ──── Observation → MemoryNode (LLM importance 1-10)
      │     ├── Retriever ──── Query memory with composite scoring
      │     ├── Reflector ──── Importance > 150 → focal points → insights
      │     └── Planner ────── Recursive plan decomposition
      │
      ├── Character Cards ──── W++ / SBF / Tavern Card V2 → PersonaSpec
      │
      └── Adapters
            ├── LLMAdapter (ABC) ── OpenAI / Anthropic / custom
            └── EmbeddingAdapter (ABC) ── OpenAI / custom

## Cognitive Loop

Every `brain.chat()` call:

    User Message
      → [Perceive]  Store as MemoryNode, LLM judges importance (1-10)
      → [Retrieve]  Score all memories: recency + importance + relevance → top-k
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
| `brain.chat(message)` | Full cognitive loop — perceive, retrieve, render, generate, reflect |
| `brain.observe(text)` | Record external event, auto-reflect if threshold exceeded |
| `brain.plan_day(date)` | Generate daily plan from persona and memories |
| `brain.reflect()` | Manual reflection — focal points → insights |
| `brain.recall(query)` | Search memories by composite score |
| `brain.apply_event_emotion(pad)` | Shift emotional state by event PAD values |
| `brain.decay_emotion(rate)` | Decay emotion toward personality baseline |
| `brain.update_situation(**traits)` | Update L2 situation layer dynamically |

## Data Models

| Model | Description |
|-------|-------------|
| `PersonaSpec` | 3-layer identity + 6 facets + OCEAN + PAD emotion + rules |
| `OceanTraits` | Big Five: O/C/E/A/N scores (0.0-1.0) with auto prompt generation |
| `EmotionalState` | PAD 3-axis (-1~+1), stimulus transition, decay, closest emotion |
| `CharacterCard` | Tavern Card V2 compatible, parsers for W++ and SBF formats |
| `MemoryNode` | SPO triple, importance, embedding, evidence pointers |
| `DailyPlan` | Recursive PlanItems with time ranges and status |
| `RetrievalResult` | Node + score breakdown (recency, importance, relevance) |

## Algorithms

| Algorithm | Source | Implementation |
|-----------|--------|----------------|
| Memory retrieval scoring | Generative Agents (Park 2023) | `memory/retrieval.py` |
| Reflection (focal points → insights) | Generative Agents (Park 2023) | `cognition/reflect.py` |
| OCEAN → PAD conversion | Mehrabian (1996) | `models.py:EmotionalState.from_ocean()` |
| Emotion transition | PAD stimulus model | `models.py:EmotionalState.apply_stimulus()` |
| Emotion decay | Exponential return to baseline | `models.py:EmotionalState.decay()` |
| Personality-sensitivity coupling | N → α mapping | `models.py:PersonaSpec.apply_event()` |
| W++ parsing | Community standard | `models.py:CharacterCard.from_wpp()` |
| SBF parsing | Community standard | `models.py:CharacterCard.from_sbf()` |

---

## References

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Memory stream, reflection, planning
- [Synaptic Memory](https://github.com/PlateerLab/synaptic-memory) — Brain-inspired knowledge graph, Hebbian learning
- [Mehrabian PAD Model (1996)](https://en.wikipedia.org/wiki/PAD_emotional_state_model) — Pleasure-Arousal-Dominance emotional space
- [Big Five / OCEAN](https://en.wikipedia.org/wiki/Big_Five_personality_traits) — Five-factor personality model
- [BIG5-CHAT (2024)](https://openreview.net/pdf?id=TqwTzLjzGS) — Big Five personality in LLM conversations
- [Machine Mindset (MBTI)](https://arxiv.org/html/2312.12999v3) — MBTI-based LLM personality tuning
- [JPAF: Evolving Personality](https://github.com/agent-topia/evolving_personality) — Jung function weights for dynamic personality
- [Character Card V2 Spec](https://github.com/malfoyslastname/character-card-spec-v2) — Tavern Card standard
- [Leaked System Prompts](https://github.com/jujumilk3/leaked-system-prompts) — Real-world persona patterns

## Project Status (v0.1.0)

> **Phase: Core Architecture Complete — Pre-release**

### Implemented

| Module | Status | Files |
|--------|--------|-------|
| **Data Models** | Done | `models.py` — OceanTraits, EmotionalState, PersonaSpec, PersonaLayer, CharacterCard, MemoryNode, PlanItem, DailyPlan, RetrievalResult |
| **Brain Facade** | Done | `brain.py` — chat, observe, plan_day, reflect, recall, emotion control |
| **Persona Renderer** | Done | `persona/renderer.py` — ISS + OCEAN + emotion + memories + plan → system prompt |
| **Memory Stream** | Done | `memory/stream.py` — append, retrieve (composite scoring), get_recent, importance tracking |
| **Retrieval Scoring** | Done | `memory/retrieval.py` — recency × importance × relevance, min-max normalization, cosine similarity |
| **Storage Backend** | Done | `memory/store.py` (ABC) + `storage/memory_store.py` (InMemoryStore) |
| **Cognition: Perceive** | Done | `cognition/perceive.py` — observation → MemoryNode (LLM importance 1-10, SPO triple extraction) |
| **Cognition: Retrieve** | Done | `cognition/retrieve.py` — composite scoring wrapper, reflection-specific retrieval |
| **Cognition: Reflect** | Done | `cognition/reflect.py` — importance threshold → focal points → insights → depth=2+ nodes |
| **Cognition: Plan** | Done | `cognition/plan.py` — daily plan, recursive decompose, replan on new observations |
| **LLM Adapters** | Done | `llm/openai.py` (OpenAI), `llm/anthropic.py` (Anthropic Claude) |
| **Embedding Adapter** | Done | `embedding/openai.py` (text-embedding-3-small/large/ada-002) |
| **Character Cards** | Done | `models.py` — W++ parser, SBF parser, Tavern Card V2 → PersonaSpec conversion |

### Not Yet Implemented

| Item | Notes |
|------|-------|
| Unit / integration tests | `tests/` directory exists but empty |
| Persistent storage backend | SQLite, Redis, etc. — currently InMemory only |
| Anthropic embedding adapter | Only OpenAI embeddings available |
| Multi-turn conversation history | Brain.chat() is single-turn (no message history accumulation) |
| PyPI publish | Package configured (`pyproject.toml`) but not yet published |
| CI/CD | No GitHub Actions / workflows |
| Tavern Card V2 export | Import only, no export to card format |
| L1/L2 persona auto-evolution | Layers exist but no automatic update logic from interactions |

---

## License

MIT
