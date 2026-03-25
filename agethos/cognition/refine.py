"""Self-Refine 모듈 — 응답 자기 평가 + 수정 사이클.

Reflexion 패턴 (Shinn et al., 2023) 기반:
1. Generate: 초기 응답 생성
2. Evaluate: 페르소나 일관성, 사회적 적절성, 유용성 평가
3. Refine: 평가 피드백으로 응답 수정
4. 반복 (quality_threshold 도달 또는 max_iterations)
"""

from __future__ import annotations

from agethos.llm.base import LLMAdapter
from agethos.models import SelfRefineConfig, SelfRefineResult

_EVALUATE_PROMPT = """\
Evaluate the following response on these axes:
{axes}

Persona description:
{persona_summary}

User message: {user_message}
Agent response: {response}

Score each axis from 0.0 to 1.0 and provide brief feedback.

Respond in JSON:
{{
  "scores": {{"axis_name": <score>, ...}},
  "feedback": "<specific improvement suggestions>",
  "overall": <0.0 to 1.0>
}}"""

_REFINE_PROMPT = """\
Improve the following response based on the feedback.

Original response: {response}

Feedback: {feedback}

Persona to maintain: {persona_summary}

Rewrite the response to address the feedback while keeping the same core message.
Return ONLY the improved response text, nothing else."""


class SelfRefiner:
    """Self-Refine 엔진 — 응답 품질 자동 개선.

    Usage::

        refiner = SelfRefiner(llm=llm, config=SelfRefineConfig(enabled=True))
        result = await refiner.refine(response, user_message, persona_summary)
        # result.refined = 개선된 응답
        # result.iterations = 수정 횟수
    """

    def __init__(self, llm: LLMAdapter, config: SelfRefineConfig | None = None):
        self._llm = llm
        self._config = config or SelfRefineConfig()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    async def refine(
        self,
        response: str,
        user_message: str,
        persona_summary: str,
    ) -> SelfRefineResult:
        """응답 self-refine 실행."""
        if not self._config.enabled:
            return SelfRefineResult(original=response, refined=response)

        current = response
        all_scores: list[dict[str, float]] = []

        for i in range(self._config.max_iterations):
            # Evaluate
            eval_result = await self._evaluate(current, user_message, persona_summary)
            all_scores.append(eval_result.get("scores", {}))

            overall = eval_result.get("overall", 1.0)
            if overall >= self._config.quality_threshold:
                break

            # Refine
            feedback = eval_result.get("feedback", "")
            if not feedback:
                break

            refined = await self._refine_response(current, feedback, persona_summary)
            if refined and refined != current:
                current = refined
            else:
                break

        return SelfRefineResult(
            original=response,
            refined=current,
            iterations=len(all_scores),
            scores=all_scores,
        )

    async def _evaluate(
        self,
        response: str,
        user_message: str,
        persona_summary: str,
    ) -> dict:
        """응답 평가."""
        axes_text = "\n".join(f"- {axis}" for axis in self._config.evaluate_axes)
        try:
            return await self._llm.generate_json(
                system_prompt="You evaluate AI agent responses for quality and persona consistency.",
                user_prompt=_EVALUATE_PROMPT.format(
                    axes=axes_text,
                    persona_summary=persona_summary,
                    user_message=user_message,
                    response=response,
                ),
            )
        except Exception:
            return {"overall": 1.0, "scores": {}, "feedback": ""}

    async def _refine_response(
        self,
        response: str,
        feedback: str,
        persona_summary: str,
    ) -> str:
        """피드백 기반 응답 수정."""
        try:
            return await self._llm.generate(
                system_prompt="You improve AI agent responses based on feedback.",
                user_prompt=_REFINE_PROMPT.format(
                    response=response,
                    feedback=feedback,
                    persona_summary=persona_summary,
                ),
            )
        except Exception:
            return response
