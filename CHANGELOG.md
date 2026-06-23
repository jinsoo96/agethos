# Changelog

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
