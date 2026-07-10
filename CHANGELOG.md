# Changelog

## 0.16.0 — LLM selection layer: mode × provider × model

### Added
- **`agethos.llm.select`** — `LLMConfig` (mode: `api` / `subscription` / `auto`,
  provider, model, api_key, base_url, timeout, custom CLI `command`) + `resolve_llm()`
  mapping the full matrix: claude/anthropic, openai/chatgpt/codex, gemini/google,
  vllm/ollama/openai-compatible (base_url), litellm catch-all for everything else;
  subscription side maps to the CLI adapters.
- **auto mode** — prefer an available API key, fall back to an installed subscription
  CLI, raise with a machine-discovery report when neither exists.
  `available_backends()` reports which API keys are set and which CLIs are on PATH.
- **Env defaults** — `AGETHOS_LLM_MODE` / `AGETHOS_LLM_PROVIDER` / `AGETHOS_LLM_MODEL` /
  `AGETHOS_LLM_API_KEY` / `AGETHOS_LLM_BASE_URL` via `LLMConfig.from_env()`.
- **Brain wiring** — `llm=` on `Brain.build` / `Brain.from_description` / `Brain.load`
  now also accepts a config dict, an `LLMConfig`, or `"auto"`; existing provider strings
  and adapter instances behave exactly as before.

12 new tests (`test_v0160.py`); full suite 290 passed, 1 skipped.

## 0.15.0 — subscription CLI adapters (no API key)

### Added
- **CLI adapters** (`agethos.llm.cli`) — run inference through a locally-authenticated
  AI CLI instead of a metered API key: `ClaudeCodeAdapter` (`claude -p`, Claude Pro/Max
  subscription via OAuth; `--system-prompt-file` + `--strict-mcp-config` for fast, safe
  headless calls), `GeminiCLIAdapter`, `CodexCLIAdapter`, and a generic
  `CLIAdapter(argv)` with `{system}` / `{system_file}` / `{prompt}` placeholders
  (stdin prompt by default; system folded into the prompt for CLIs without a system
  flag; temp-file system prompt survives Windows `.cmd` shim quoting).
- **Provider strings** — `Brain.build(..., llm="subscription")` (aliases `claude-code`,
  `claude-cli`), `"gemini-cli"`, `"codex-cli"`.
- Multi-turn history folding for single-shot CLIs; CRLF-normalized output.

### Fixed
- `LLMAdapter.generate_json` now extracts the outermost JSON object when a backend
  wraps the JSON in prose, instead of failing to parse.

10 new tests (`test_v0150.py`); full suite 278 passed, 1 skipped.

## 0.14.0 — judge panel, behavioral verification, GPU-free steering

### Added
- **Multi-sample forge + judge panel** (`agethos.forge.panel`) — `forge(..., samples=N,
  judges=M)`: N independent drafts (varied readings), each scored by a panel of
  differently-lensed judges (fidelity / coverage / overreach) aggregated by median
  (`aggregate_reports`), then facet-level `graft()` composes the best-scoring parts into
  one config. `judge_spec(..., lens=...)` / `draft_spec(..., variant=...)` support the
  panel. (Self-consistency, Wang et al. 2022; PoLL, Verga et al. 2024.)
- **Behavioral verification** (`agethos.forge.verify`) — `verify_persona()` administers
  the Mini-IPIP inventory (Donnellan et al. 2006; public-domain IPIP) to the mounted
  persona in one batched call, reverse-keys and normalizes, and returns a
  `BehavioralReport` (measured OCEAN, `ocean_fidelity`, per-trait gaps) — the
  questionnaire method of MPI (Jiang et al., NeurIPS 2023) / PersonaLLM (2024).
  `verify_social()` runs a short two-persona scenario episode and scores it on the
  SOTOPIA rubric via the existing adapter. `ForgeResult.verify()` convenience.
- **GPU-free steering for black-box LLMs** (`agethos.steering.rerank`) —
  `attribute_score()` (deterministic trait-pole attribute model from the contrastive
  vocabulary) + `steered_generate()` (sample n, re-rank by attribute + optional LLM
  judge, best-of-n). PPLM/FUDGE attribute-guided generation (Dathathri et al. 2020;
  Yang & Klein 2021) recast as rejection sampling (Stiennon et al. 2020; Nakano et al.
  2021). Wired into the cognitive loop: `Brain.chat(msg, steer_n=3)`.
- **Steering plans promoted to `agethos.steering.plan`** — `SteeringIntent`,
  `plan_from_ocean()`, `plan_vectors()` now live beside the other steering machinery
  (re-exported from `agethos.forge` unchanged, fully backward compatible).

12 new tests (`test_v0140.py`); full suite 268 passed, 1 skipped.

## 0.13.0 — the persona forge: description → typed config → any LLM

### Added
- **Persona forge** (`agethos.forge`) — compile a free-text personality description
  (rough or detailed, any language) into the full typed `PersonaSpec` config, judge the
  config's fidelity to the description facet-by-facet, and re-forge only the weak facets
  until it converges. Personality as config values, not a hand-written system prompt.
  - `draft_spec` / `repair_spec` — LLM-driven compiler with lenient coercion (OCEAN
    clamped to [0,1], invalid enum values dropped, layers/emotion label mapped); zero-LLM
    deterministic fallback via an EN+KO substring lexicon (`estimate_ocean`).
  - `judge_spec` → `ForgeReport` (per-facet `FacetScore` + `weak()` / `issues()`);
    deterministic coverage judge for offline runs.
  - `forge()` → `ForgeResult` — the loop (draft → judge → targeted repair) with a
    convergence `trace`, plus config layering: `base` (existing spec) < forged draft <
    `pin` (fields the forge may never overwrite).
  - Steering handoff — `ForgeResult.steering_plan()` (traits deviating from the 0.5
    prior → direction + strength) and `plan_vectors()` (plan → scaled persona vectors
    for open-weight steering).
- **`Brain.from_description()`** — one call from description to a mounted brain on any
  provider (`openai` / `anthropic` / `litellm` / adapter instance); the forge trace lands
  on `brain.forge_result`.

Measured with a real LLM (Claude Sonnet 4.5): a rough one-liner forged to fidelity 0.91
and a detailed paragraph to 0.98, both converging in one round; the two mounted personas
answer the same message in completely different, description-faithful ways.

13 new tests (`test_v0130.py`); full suite 256 passed, 1 skipped.

## 0.12.0 — persona-vector steering, real-benchmark adapters, async concurrency

### Added
- **Activation steering / persona vectors** (`agethos.steering`) — extract an OCEAN persona
  vector from contrastive prompt pairs (mean activation difference, Anthropic persona
  vectors) and steer it at inference; multi-trait vectors orthogonalized to avoid
  interference (PERSONA). Pure-Python vector math + prompt generation
  (`trait_contrastive_prompts`, `mean_diff`, `orthogonalize`, `combine`, `steer`,
  `extract_persona_vectors`); `MockSteeringBackend` for offline use, optional
  `TransformersSteeringBackend` (`pip install agethos[steering]`) for real open-weight models.
- **Benchmark adapters** — `agethos.eval` is now a package:
  - `eval.ingest_conversation` + `eval.evaluate_recall` — LoCoMo-style long-term memory
    harness (ingest a conversation, score whether retrieval surfaces each question's evidence).
  - `eval.persona_to_sotopia_profile` + `eval.score_episode` — Sotopia adapter (map a persona
    to a Sotopia profile; score an episode on the 7-dimension `SocialEvaluation` rubric).
  - Existing metrics (`persona_consistency`, `transplant_fidelity`, `retrieval_metrics`, …)
    unchanged and re-exported.
- **Async concurrency** (`agethos.concurrency`) — `gather_bounded` / `amap` (bounded-concurrency
  helpers). Reflection now generates per-focal-point insights concurrently, and A-MEM
  `link_and_evolve` updates neighbors concurrently — wall-clock = slowest call, not the sum.

10 new tests (`test_v0120.py`); full suite 243 passed, 1 skipped.

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
