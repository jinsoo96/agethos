"""플랫폼별 인격 내보내기 어댑터.

.brain(완전체) → 타겟 플랫폼 맞춤 변환.
변환 시 정보 손실 있음 — 원본 .brain은 항상 보존.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agethos.brain import Brain


def export_brain(brain: Brain, format: str, **kwargs) -> str | dict:
    """Brain을 지정 포맷으로 내보내기."""
    exporters = {
        "system_prompt": _export_system_prompt,
        "anthropic": _export_anthropic,
        "openai_assistant": _export_openai_assistant,
        "crewai": _export_crewai,
        "bedrock_agent": _export_bedrock,
        "a2a_card": _export_a2a_card,
    }

    exporter = exporters.get(format)
    if exporter is None:
        supported = ", ".join(sorted(exporters.keys()))
        raise ValueError(f"Unknown export format: {format!r}. Supported: {supported}")

    return exporter(brain, **kwargs)


def _export_system_prompt(brain: Brain, **kwargs) -> str:
    """범용 시스템 프롬프트 텍스트."""
    from agethos.persona.renderer import PersonaRenderer
    renderer = PersonaRenderer(brain.persona)
    prompt = renderer.render_full()

    # Append learned social patterns as behavioral context
    if brain.social_patterns:
        pattern_lines = []
        for p in brain.social_patterns:
            if p.confidence >= 0.5:
                line = f"- In {p.context}: {p.effective_strategy}"
                if p.counterexample:
                    line += f" (avoid: {p.counterexample})"
                pattern_lines.append(line)
        if pattern_lines:
            prompt += "\n\n## Learned Social Patterns\n" + "\n".join(pattern_lines)

    return prompt


def _export_anthropic(brain: Brain, **kwargs) -> str:
    """Anthropic Messages API system 필드용.

    사용법::

        client.messages.create(
            model="claude-opus-4-6",
            system=brain.export("anthropic"),
            messages=[...],
        )
    """
    return _export_system_prompt(brain, **kwargs)


def _export_openai_assistant(brain: Brain, **kwargs) -> dict:
    """OpenAI Assistants API 형태.

    Returns::

        {"instructions": "...", "name": "...", "model": "gpt-4o"}
    """
    instructions = _export_system_prompt(brain, **kwargs)
    return {
        "name": brain.persona.name,
        "instructions": instructions[:256000],  # OpenAI limit
        "model": kwargs.get("model", "gpt-4o"),
    }


def _export_crewai(brain: Brain, **kwargs) -> dict:
    """CrewAI agent config.

    Returns::

        {"role": "...", "goal": "...", "backstory": "..."}
    """
    persona = brain.persona

    # Role from identity or innate traits
    role = persona.identity
    if not role and persona.l0_innate.traits:
        occupation = persona.l0_innate.traits.get("occupation", "")
        role = f"{persona.name}, {occupation}" if occupation else persona.name
    if not role:
        role = persona.name

    # Goal from values
    goal = ", ".join(persona.values) if persona.values else "Assist effectively"

    # Backstory from personality + rules
    backstory_parts = []
    if persona.ocean:
        backstory_parts.append(persona.ocean.to_prompt())
    if persona.tone:
        backstory_parts.append(f"Communication style: {persona.tone}")
    if persona.behavioral_rules:
        backstory_parts.append("Rules: " + "; ".join(persona.behavioral_rules))

    return {
        "role": role,
        "goal": goal,
        "backstory": "\n".join(backstory_parts) if backstory_parts else f"{persona.name}",
    }


def _export_bedrock(brain: Brain, **kwargs) -> dict:
    """AWS Bedrock Agent instruction (4000자 제한).

    Returns::

        {"instruction": "...(max 4000)", "agentName": "..."}
    """
    prompt = _export_system_prompt(brain, **kwargs)

    # Bedrock limit: 4000 chars
    if len(prompt) > 4000:
        # Priority: ISS core > rules > patterns
        from agethos.persona.renderer import PersonaRenderer
        renderer = PersonaRenderer(brain.persona)
        prompt = renderer.render_iss()[:4000]

    return {
        "agentName": brain.persona.name.replace(" ", "_")[:100],
        "instruction": prompt[:4000],
    }


def _export_a2a_card(brain: Brain, **kwargs) -> dict:
    """A2A Agent Card (서비스 디스커버리용).

    Returns::

        {"name": "...", "description": "...", "version": "...", "skills": [...]}
    """
    persona = brain.persona
    description = persona.identity or f"AI agent with personality: {persona.name}"

    skills = []
    if persona.values:
        for v in persona.values:
            skills.append({"name": v.lower().replace(" ", "_"), "description": v})

    return {
        "name": persona.name,
        "description": description,
        "version": "0.8.0",
        "skills": skills,
        "provider": {"name": "agethos"},
    }
