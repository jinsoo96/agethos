"""Description → PersonaSpec compiler (the "draft" stage of the forge).

Turns a free-text personality description — one line or a full paragraph, any
language — into the complete typed config (``PersonaSpec``): OCEAN, tone, values,
constraints, decision style, seed memory. Config, not prose: everything downstream
(``CognitivePolicy``, renderer, steering, transplant) derives from these values.

Layering (harness-config style): ``base`` (existing spec) < forged draft < ``pin``
(user-fixed fields, never overwritten by any forge round).
"""
from __future__ import annotations

from agethos.forge.lexicon import estimate_ocean, extract_innate
from agethos.llm.base import LLMAdapter
from agethos.models import (
    DecisionStyle,
    EmotionalState,
    MoralFoundation,
    OceanTraits,
    PersonaLayer,
    PersonaSpec,
    SchwartzValue,
)

# Fields the LLM is asked to produce. Free-text fields follow the description's language.
_DRAFT_SYSTEM = """You compile a personality description into a typed persona config.
Extract ONLY what the description states or strongly implies; leave fields you cannot
ground as empty string / empty list / omit them. Write free-text fields in the same
language as the description. Respond with a single JSON object with these keys:

- name: string (invent a fitting one only if the description names nobody)
- identity: 1-3 sentence self-description ("You are ...")
- tone: how they speak (concrete: rhythm, bluntness, warmth, humor)
- conversation_style: interaction habits (question-asking, listening, interrupting...)
- transparency: how openly they show inner state
- functional_role: role-based persona ("Senior backend engineer"), if any
- relational_mode: relationship stance ("mentor", "rival", "pair-programming partner"), if any
- values: list of short strings
- boundaries: list of short strings (topics/behaviors they refuse)
- behavioral_rules: list of short imperative strings
- hard_constraints: list of NEVER/ALWAYS rules stated in the description
- soft_preferences: list of adjustable preferences
- moral_values: subset of ["care","fairness","loyalty","authority","purity","liberty"]
- schwartz_values: subset of ["self_direction","stimulation","hedonism","achievement","power","security","conformity","tradition","benevolence","universalism"]
- decision_style: one of "directive","analytical","conceptual","behavioral" or ""
- ocean: {"openness","conscientiousness","extraversion","agreeableness","neuroticism"} each 0.0-1.0
- innate: object of innate traits ({"age": "34", "occupation": "..."}), strings only
- learned: object of learned traits/skills, strings only
- situation: object describing their current situation, strings only
- seed_memory: a short first-person background paragraph seeding their memory
- emotion_label: baseline emotion, one of ["joy","anger","sadness","fear","disgust","surprise","contempt","shame","pride","calm","excitement","boredom"] or ""
"""

_REPAIR_SYSTEM = _DRAFT_SYSTEM + """
You are REPAIRING an existing config. Regenerate ONLY the listed weak fields (grounded
in the description); return a JSON object containing just those fields."""

_LIST_FIELDS = ("values", "boundaries", "behavioral_rules", "hard_constraints", "soft_preferences")
_TEXT_FIELDS = ("name", "identity", "tone", "conversation_style", "transparency",
                "functional_role", "relational_mode", "seed_memory")
_LAYER_FIELDS = {"innate": "l0_innate", "learned": "l1_learned", "situation": "l2_situation"}


def _clean_enum_list(values, enum_cls) -> list:
    """Keep only entries that are valid enum values (LLMs invent categories)."""
    valid = {e.value for e in enum_cls}
    return [enum_cls(v) for v in values if isinstance(v, str) and v in valid]


def coerce_draft(data: dict, base: PersonaSpec | None = None, pin: dict | None = None) -> PersonaSpec:
    """Lenient dict → PersonaSpec: clamp OCEAN, drop invalid enums, apply base/pin layers."""
    spec = base.model_copy(deep=True) if base else PersonaSpec(name=str(data.get("name") or "Persona"))

    if data.get("name"):
        spec.name = str(data["name"])

    for f in _TEXT_FIELDS[1:]:  # name handled above
        v = data.get(f)
        if isinstance(v, str) and v.strip():
            setattr(spec, f, v.strip())

    for f in _LIST_FIELDS:
        v = data.get(f)
        if isinstance(v, list):
            items = [str(x).strip() for x in v if str(x).strip()]
            if items:
                setattr(spec, f, items)

    o = data.get("ocean")
    if isinstance(o, dict):
        kw = {}
        for dim in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            raw = o.get(dim, o.get(dim[0].upper(), 0.5))
            try:
                kw[dim] = max(0.0, min(1.0, float(raw)))
            except (TypeError, ValueError):
                kw[dim] = 0.5
        spec.ocean = OceanTraits(**kw)

    if isinstance(data.get("moral_values"), list):
        mv = _clean_enum_list(data["moral_values"], MoralFoundation)
        if mv:
            spec.moral_values = mv
    if isinstance(data.get("schwartz_values"), list):
        sv = _clean_enum_list(data["schwartz_values"], SchwartzValue)
        if sv:
            spec.schwartz_values = sv
    ds = data.get("decision_style")
    if isinstance(ds, str) and ds in {e.value for e in DecisionStyle}:
        spec.decision_style = DecisionStyle(ds)

    for short, full in _LAYER_FIELDS.items():
        v = data.get(short)
        if isinstance(v, dict) and v:
            traits = {str(k): str(val) for k, val in v.items() if str(val).strip()}
            if traits:
                setattr(spec, full, PersonaLayer(traits=traits))

    label = data.get("emotion_label")
    if isinstance(label, str) and label in EmotionalState().EMOTION_MAP:
        spec.emotion = EmotionalState.from_label(label, intensity=0.5)

    if pin:
        pinned = PersonaSpec.from_dict({"name": spec.name, **pin})
        for f, v in pin.items():
            full = _LAYER_FIELDS.get(f, "behavioral_rules" if f == "rules" else f)
            setattr(spec, full, getattr(pinned, full))

    spec.init_emotion_from_ocean()
    return spec


def deterministic_draft(description: str, name: str | None = None) -> dict:
    """Zero-LLM draft dict from the lexicon: OCEAN + evidence-based tone + seed memory."""
    scores, evidence = estimate_ocean(description)
    matched = [w for words in evidence.values() for w in words]
    persona_name = name or "Persona"
    return {
        "name": persona_name,
        "identity": f"You are {persona_name}. {description.strip()}",
        "tone": ", ".join(matched) if matched else "",
        "ocean": scores,
        "innate": extract_innate(description),
        "seed_memory": description.strip(),
    }


async def draft_spec(
    description: str,
    llm: LLMAdapter | None = None,
    name: str | None = None,
    base: PersonaSpec | None = None,
    pin: dict | None = None,
    variant: str = "",
) -> PersonaSpec:
    """Compile a description into a full PersonaSpec (LLM-driven; lexicon fallback).

    ``variant`` nudges this draft toward a different reading (used for multi-sampling)."""
    if llm is None:
        data = deterministic_draft(description, name=name)
    else:
        user = f"Personality description:\n{description}"
        if name:
            user += f"\n\nThe persona's name MUST be: {name}"
        if variant:
            user += f"\n\n{variant}"
        data = await llm.generate_json(_DRAFT_SYSTEM, user)
        if name:
            data["name"] = name
    return coerce_draft(data, base=base, pin=pin)


async def repair_spec(
    description: str,
    spec: PersonaSpec,
    weak_facets: list[str],
    issues: dict[str, str] | None = None,
    llm: LLMAdapter | None = None,
    pin: dict | None = None,
) -> PersonaSpec:
    """Regenerate only the weak facets of an existing spec (pin always wins)."""
    if llm is None:
        data = _deterministic_fill(description, spec, weak_facets)
    else:
        issue_lines = "\n".join(f"- {f}: {issues.get(f, 'low fidelity')}" for f in weak_facets) if issues \
            else "\n".join(f"- {f}" for f in weak_facets)
        user = (
            f"Personality description:\n{description}\n\n"
            f"Current config:\n{spec.model_dump_json(exclude_none=True)}\n\n"
            f"Weak fields to regenerate:\n{issue_lines}"
        )
        data = await llm.generate_json(_REPAIR_SYSTEM, user)
        data.pop("name", None)
    return coerce_draft(data, base=spec, pin=pin)


# judge facet → spec fields the deterministic filler can ground from the description
def _deterministic_fill(description: str, spec: PersonaSpec, weak_facets: list[str]) -> dict:
    draft = deterministic_draft(description, name=spec.name)
    fill: dict = {}
    if "identity" in weak_facets and not spec.identity:
        fill["identity"] = draft["identity"]
    if "ocean" in weak_facets and spec.ocean is None:
        fill["ocean"] = draft["ocean"]
    if "tone" in weak_facets and not spec.tone and draft["tone"]:
        fill["tone"] = draft["tone"]
    if "background" in weak_facets:
        if not spec.seed_memory:
            fill["seed_memory"] = draft["seed_memory"]
        if not spec.l0_innate.traits and draft["innate"]:
            fill["innate"] = draft["innate"]
    return fill
