"""PersonaSpec → 시스템 프롬프트 텍스트 변환."""

from __future__ import annotations

from agethos.models import DailyPlan, MemoryNode, PersonaSpec


class PersonaRenderer:
    """PersonaSpec을 LLM 시스템 프롬프트로 렌더링.

    Identity Stable Set (ISS): 모든 LLM 호출에 주입되는 핵심 정체성 텍스트.
    OCEAN 성격 특성, PAD 감정 상태를 포함.
    """

    def __init__(self, spec: PersonaSpec):
        self._spec = spec

    def render_iss(self) -> str:
        """Identity Stable Set — 핵심 정체성 블록."""
        parts: list[str] = []

        # Identity declaration
        if self._spec.identity:
            parts.append(self._spec.identity)
        else:
            parts.append(f"Your name is {self._spec.name}.")

        # L0: Innate traits
        if self._spec.l0_innate.traits:
            trait_lines = [f"- {k}: {v}" for k, v in self._spec.l0_innate.traits.items()]
            parts.append("## Core Traits\n" + "\n".join(trait_lines))

        # L1: Learned traits
        if self._spec.l1_learned.traits:
            trait_lines = [f"- {k}: {v}" for k, v in self._spec.l1_learned.traits.items()]
            parts.append("## Experience & Relationships\n" + "\n".join(trait_lines))

        # L2: Current situation
        if self._spec.l2_situation.traits:
            trait_lines = [f"- {k}: {v}" for k, v in self._spec.l2_situation.traits.items()]
            parts.append("## Current Situation\n" + "\n".join(trait_lines))

        # OCEAN personality
        if self._spec.ocean:
            parts.append("## Personality Profile (Big Five / OCEAN)\n" + self._spec.ocean.to_prompt())

        # Tone
        if self._spec.tone:
            parts.append(f"## Tone\n{self._spec.tone}")

        # Values
        if self._spec.values:
            parts.append("## Values\n" + "\n".join(f"- {v}" for v in self._spec.values))

        # Boundaries
        if self._spec.boundaries:
            parts.append("## Boundaries\n" + "\n".join(f"- {b}" for b in self._spec.boundaries))

        # Conversation style
        if self._spec.conversation_style:
            parts.append(f"## Conversation Style\n{self._spec.conversation_style}")

        # Transparency
        if self._spec.transparency:
            parts.append(f"## Self-awareness\n{self._spec.transparency}")

        # Behavioral rules
        if self._spec.behavioral_rules:
            parts.append(
                "## Behavioral Rules\n" + "\n".join(f"- {r}" for r in self._spec.behavioral_rules)
            )

        return "\n\n".join(parts)

    def render_full(
        self,
        context_memories: list[MemoryNode] | None = None,
        current_plan: DailyPlan | None = None,
    ) -> str:
        """전체 시스템 프롬프트 (ISS + 감정 + 기억 + 계획)."""
        sections: list[str] = [self.render_iss()]

        # Emotional state
        if self._spec.emotion:
            sections.append("## Emotional State\n" + self._spec.emotion.to_prompt())

        # Relevant memories
        if context_memories:
            memory_lines = []
            for m in context_memories:
                memory_lines.append(f"- [{m.node_type.value}] {m.description}")
            sections.append("## Relevant Memories\n" + "\n".join(memory_lines))

        # Current plan
        if current_plan:
            plan_lines = [f"Date: {current_plan.date}"]
            if current_plan.summary:
                plan_lines.append(f"Summary: {current_plan.summary}")
            for item in current_plan.items:
                status_mark = "x" if item.status == "done" else " "
                time_info = f" ({item.time_range})" if item.time_range else ""
                plan_lines.append(f"- [{status_mark}] {item.description}{time_info}")
            sections.append("## Current Plan\n" + "\n".join(plan_lines))

        # Seed memory
        if self._spec.seed_memory:
            sections.append(f"## Background\n{self._spec.seed_memory}")

        return "\n\n".join(sections)
