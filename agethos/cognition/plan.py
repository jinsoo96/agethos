"""계획 모듈 — 재귀적 계획 분해.

Generative Agents의 Planning 메커니즘:
daily summary → hourly blocks → 5-15분 단위 행동
"""

from __future__ import annotations

from agethos.llm.base import LLMAdapter
from agethos.models import DailyPlan, MemoryNode, PersonaSpec, PlanItem

_DAILY_PLAN_PROMPT = """\
You are {name}.
{persona_context}

Date: {date}
{context}

Plan today's schedule with 5-8 main activities.
Each activity should include a time range and estimated duration.

Respond in JSON:
{{"summary": "<day summary>", "items": [{{"description": "<activity>", "time_range": "<HH:MM-HH:MM>", "duration_minutes": <minutes>}}]}}"""

_DECOMPOSE_PROMPT = """\
Break down the following activity into sub-actions at {granularity}-minute granularity.

Activity: {description}
Time: {time_range}

Respond in JSON:
{{"sub_items": [{{"description": "<sub-action>", "duration_minutes": <minutes>}}]}}"""

_REPLAN_PROMPT = """\
You are {name}.

Current plan:
{current_plan}

New situation: {observation}

Revise the remaining plan to reflect this new situation.

Respond in JSON:
{{"summary": "<revised summary>", "items": [{{"description": "<activity>", "time_range": "<HH:MM-HH:MM>", "duration_minutes": <minutes>, "status": "<pending|done>"}}]}}"""


class Planner:
    """계획 수립 엔진."""

    def __init__(self, llm: LLMAdapter, persona: PersonaSpec):
        self._llm = llm
        self._persona = persona

    async def create_daily_plan(
        self,
        date: str,
        context: str = "",
        existing_memories: list[MemoryNode] | None = None,
    ) -> DailyPlan:
        """일일 계획 생성."""
        persona_context = ""
        if self._persona.l0_innate.traits:
            persona_context = "\n".join(
                f"{k}: {v}" for k, v in self._persona.l0_innate.traits.items()
            )

        memory_context = ""
        if existing_memories:
            memory_context = "\n최근 기억:\n" + "\n".join(
                f"- {m.description}" for m in existing_memories[:10]
            )

        full_context = context
        if memory_context:
            full_context += memory_context

        try:
            data = await self._llm.generate_json(
                system_prompt="You are a helper that plans schedules.",
                user_prompt=_DAILY_PLAN_PROMPT.format(
                    name=self._persona.name,
                    persona_context=persona_context,
                    date=date,
                    context=full_context or "No special context",
                ),
            )

            items = [
                PlanItem(
                    description=item["description"],
                    time_range=item.get("time_range", ""),
                    duration_minutes=item.get("duration_minutes", 30),
                )
                for item in data.get("items", [])
            ]

            return DailyPlan(
                date=date,
                summary=data.get("summary", ""),
                items=items,
            )
        except Exception:
            return DailyPlan(date=date, summary="계획 생성 실패")

    async def decompose(
        self,
        item: PlanItem,
        granularity_minutes: int = 15,
    ) -> PlanItem:
        """계획 항목을 세부 행동으로 분해."""
        try:
            data = await self._llm.generate_json(
                system_prompt="You are a helper that breaks down activities into sub-steps.",
                user_prompt=_DECOMPOSE_PROMPT.format(
                    description=item.description,
                    time_range=item.time_range or f"{item.duration_minutes}분",
                    granularity=granularity_minutes,
                ),
            )

            item.sub_items = [
                PlanItem(
                    description=sub["description"],
                    duration_minutes=sub.get("duration_minutes", granularity_minutes),
                    parent_id=item.id,
                )
                for sub in data.get("sub_items", [])
            ]
            return item
        except Exception:
            return item

    async def replan(
        self,
        current_plan: DailyPlan,
        new_observation: str,
    ) -> DailyPlan:
        """새로운 관찰에 따라 계획 수정."""
        plan_text = "\n".join(
            f"- [{item.status}] {item.description} ({item.time_range})"
            for item in current_plan.items
        )

        try:
            data = await self._llm.generate_json(
                system_prompt="You are a helper that adjusts schedules.",
                user_prompt=_REPLAN_PROMPT.format(
                    name=self._persona.name,
                    current_plan=plan_text,
                    observation=new_observation,
                ),
            )

            items = [
                PlanItem(
                    description=item["description"],
                    time_range=item.get("time_range", ""),
                    duration_minutes=item.get("duration_minutes", 30),
                    status=item.get("status", "pending"),
                )
                for item in data.get("items", [])
            ]

            return DailyPlan(
                date=current_plan.date,
                summary=data.get("summary", current_plan.summary),
                items=items,
            )
        except Exception:
            return current_plan
