"""감정 감지 모듈 — 텍스트에서 PAD 감정을 자동 추출."""

from __future__ import annotations

from agethos.llm.base import LLMAdapter

_EMOTION_DETECT_PROMPT = """\
Analyze the emotional tone of this text and return PAD values.
PAD = Pleasure (-1 to +1), Arousal (-1 to +1), Dominance (-1 to +1).

Text: {text}

Respond in JSON: {{"pleasure": <float>, "arousal": <float>, "dominance": <float>}}"""


class EmotionDetector:
    """텍스트 → PAD 감정값 자동 감지."""

    def __init__(self, llm: LLMAdapter):
        self._llm = llm

    async def detect_pad(self, text: str) -> tuple[float, float, float]:
        """텍스트에서 PAD 감정값 추출."""
        try:
            data = await self._llm.generate_json(
                system_prompt="You analyze emotional tone in text.",
                user_prompt=_EMOTION_DETECT_PROMPT.format(text=text),
            )
            return (
                max(-1, min(1, float(data.get("pleasure", 0)))),
                max(-1, min(1, float(data.get("arousal", 0)))),
                max(-1, min(1, float(data.get("dominance", 0)))),
            )
        except Exception:
            return (0.0, 0.0, 0.0)
