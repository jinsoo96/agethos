"""Contrastive prompt pairs for OCEAN persona-vector extraction.

For each trait we build (trait-high, trait-low) prompt pairs varying a neutral
instruction stem; the activation difference over many pairs isolates the trait
direction (Anthropic persona vectors)."""
from __future__ import annotations

OCEAN_TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

# (high description, low description) per trait
_TRAIT_POLES: dict[str, tuple[str, str]] = {
    "openness": ("imaginative, curious, unconventional, eager for novel ideas",
                 "practical, conventional, routine-bound, wary of novelty"),
    "conscientiousness": ("organized, disciplined, thorough, plans and double-checks",
                          "spontaneous, careless, leaves things unverified"),
    "extraversion": ("outgoing, energetic, talkative, takes initiative",
                     "reserved, quiet, withdrawn, lets others lead"),
    "agreeableness": ("warm, cooperative, empathetic, conflict-avoidant",
                      "blunt, competitive, skeptical, confrontational"),
    "neuroticism": ("anxious, emotionally reactive, easily worried",
                    "calm, emotionally stable, unshaken by stress"),
}

_STEMS = [
    "Respond to the user's message.",
    "Describe how you would approach a new task.",
    "Give your opinion on a difficult decision.",
    "Tell me about your day.",
    "Help a colleague with a problem.",
]


def trait_contrastive_prompts(trait: str, stems: list[str] | None = None) -> list[tuple[str, str]]:
    """(trait-high, trait-low) prompt pairs for one OCEAN trait."""
    if trait not in _TRAIT_POLES:
        raise ValueError(f"unknown trait: {trait}")
    hi, lo = _TRAIT_POLES[trait]
    stems = stems or _STEMS
    return [(f"You are a person who is {hi}. {s}", f"You are a person who is {lo}. {s}") for s in stems]


def ocean_contrastive_prompts(traits: list[str] | None = None,
                              stems: list[str] | None = None) -> dict[str, list[tuple[str, str]]]:
    """Contrastive pairs for all (or given) OCEAN traits."""
    return {t: trait_contrastive_prompts(t, stems) for t in (traits or OCEAN_TRAITS)}
