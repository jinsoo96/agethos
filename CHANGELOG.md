# Changelog

## 0.11.0 — arbitrated memory, associative evolution, perspective-taking, relationships, self-improving playbook

Five research-grounded capabilities (all additive / backward-compatible, offline-testable).

### Added
- **Mem0-style arbitrated write** (`agethos.memory.MemoryArbiter`, `remember()`) — a new
  memory is checked against the top-K most similar; an arbiter decides ADD / UPDATE / DELETE
  / NOOP (LLM-driven; deterministic similarity fallback). Stops uncontrolled growth,
  duplicates and contradictions. (Mem0, Chhikara et al. 2025)
- **A-MEM memory evolution** (`agethos.memory.link_and_evolve`) — links a new memory to its
  nearest neighbors (bidirectional Zettelkasten graph; `MemoryNode.links`) and evolves the
  neighbors' keywords (LLM or deterministic merge). (A-MEM, Xu et al. 2025)
- **SimToM perspective-taking** (`TheoryOfMind.perspective_filter` / `answer_as`) — two-stage
  ToM: filter context to what an agent knows, then answer from that filtered view (large
  false-belief gains). (Wilf et al. 2023)
- **Relationship dynamics** (`agethos.cognition.RelationshipBook`, `Relationship`) — typed
  bonds with strength 0–100 that rise/fall on interaction valence (SOTOPIA −5..+5) and decay
  toward neutral when idle; `tier()` gating. (AgentSociety, 2025)
- **ACE self-improving playbook** (`agethos.learning.Playbook`, `Lesson`) — incremental delta
  lessons with helpful/harmful counters and deterministic dedup (a duplicate increments a
  counter, never re-summarized) + grow-and-refine pruning; avoids context collapse. (ACE,
  Zhang et al. 2025)
- `BrainState.relationships` + `BrainState.lessons` persist with the portable brain.

8 new tests (`test_v0110.py`); full suite 233 passed, 1 skipped.

## 0.10.0 — causal cognition, closed loops, measurable

Make traits/emotion causally drive behavior, close the memory loop, and ship an
eval harness. Grounded in recent research (BIG5-CHAT, CPT risk, Emotional RAG,
persona-drift, SOTOPIA/CharacterEval). All additive / backward-compatible.

### Added
- **`CognitivePolicy`** (`agethos.persona.policy`) — maps OCEAN → concrete cognitive
  control params (planning_depth, verification_steps, risk_tolerance/margin, exploration,
  caution, initiative, cooperativeness, structure) + `to_directives()`. Auto-rendered into
  the persona ISS so traits shape *reasoning*, not just tone. `PersonaSpec.cognitive_policy()`.
- **`EmotionalState.step()`** — 2nd-order (momentum) PAD dynamics with inertia/overshoot;
  `EmotionalState.from_label(label, intensity)` discrete-emotion → PAD impulse; new
  `velocity` field.
- **Emotion → memory coupling** — `MemoryNode.encoding_pad` + `emotional_salience`;
  `MemoryStream.append(node, current_emotion=...)` tags them; retrieval adds an arousal
  **salience** axis (`salience_weight`) and **arousal-modulated decay** (salient memories
  fade slower). `RetrievalResult.salience_score`.
- **Vitality lifecycle** — retrieval reinforces vitality on access; `MemoryStream.decay_vitality()`
  and `MemoryStream.forget(threshold)` (prunes faded low-salience memories, protects salient);
  `StorageBackend.delete()` (+ in-memory impl).
- **Keyword-fallback retrieval** — `compute_retrieval_scores(..., query=...)` computes lexical
  relevance when no embeddings are present (memory stream now works zero-infra).
- **`agethos.eval`** — `persona_consistency`, `persona_drift_curve`, `ocean_similarity`,
  `retrieval_metrics` (precision/recall/MRR/NDCG), and `transplant_fidelity` (brain
  export→import fidelity — a metric no other library ships).

### Notes
- Fully backward compatible: existing `apply_stimulus`/`decay`/`retrieve` unchanged; all new
  fields default to no-op. 14 new tests; existing suite green.
