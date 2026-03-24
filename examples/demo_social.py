"""눈치 모듈 데모 — 같은 대화에 성격별로 다른 반응."""

import asyncio
from agethos.cognition.social import SocialCognition
from agethos.models import OceanTraits
from agethos.llm.openai import OpenAIAdapter

CONVERSATION = """
정우문 팀장님, GLM의 경우 선결제 방식이지만 알리바바 클라우드의 경우 매달 결제 방식인 것 같습니다.
두군데 모두 x2bee@plateer.com dusrnth123!으로 가입해두었습니다.
GLM은 기안 올렸습니다

알리바바는 경영지원팀 카드 등록된거고?

아직 품의 결재가 안끝나서

둘다 결제는 안되었습니다.

알리바바도 품의는 올려야 되는거 아닌가요?

그런 방식은 어떻게 해야하는지 문의 넣어 놓았습니다.

전무님, 결재 부탁드립니다.

하나 더 올렸습니다 승인 부탁드립니다

전무님 그룹웨어 결재요청 2건 결재 부탁드려요.

다운프로 전무님 옆에 계시면 전달해주세요.

둘다 제주은행 개발을 위한 용도 ?

네, 제주은행보다는 온프렘 개발 지원용입니다. 중국회사들은 오픈소스모델도 api 서비스를 하거든요.

GPU가 귀하니, 똑같이 시뮬레이션 하려면, 오히려 클라우드가 쌉니다.

결제 완료되었습니다

금일 서버 다운건 내용 공유드립니다.

[내용] : 온프렘 서버 다운 현상 발생 (243,244,248,249 등)
[일시] : 2026-03-24 13시 55분경 ~
[이유] : 서버 shutdown suspend 발생
[이슈사항] :
jenkins-xgen 다운
x2bee prd 다운
xgen.x2bee.com 다운
등등
[조치사항] :
suspend,shutdown 발발지 추적 못하였음. (OS 펌웨어 자동 업데이트로 인한 설정값 변경으로 추정 중)
suspend, shutdown 명령어 자체를 금지.
jenkins-xgen 복구
x2bee prd 복구
추가적으로 문제 발생건 및 재발 방지는 지속적으로 진행하겠습니다.
""".strip()

PERSONAS = [
    {
        "name": "김대리 (꼼꼼한 실무자)",
        "ocean": OceanTraits(openness=0.4, conscientiousness=0.9, extraversion=0.3, agreeableness=0.7, neuroticism=0.6),
        "role": "주니어 개발자",
    },
    {
        "name": "박팀장 (추진력 있는 리더)",
        "ocean": OceanTraits(openness=0.6, conscientiousness=0.8, extraversion=0.8, agreeableness=0.4, neuroticism=0.2),
        "role": "개발팀장",
    },
    {
        "name": "이사원 (조용한 신입)",
        "ocean": OceanTraits(openness=0.5, conscientiousness=0.5, extraversion=0.1, agreeableness=0.9, neuroticism=0.8),
        "role": "신입사원",
    },
]


async def main():
    llm = OpenAIAdapter(model="gpt-4o-mini")

    print("=" * 70)
    print("대화 맥락 분석")
    print("=" * 70)

    # 맥락 읽기 (한번만)
    reader = SocialCognition(llm=llm, name="분석기", ocean=OceanTraits())
    ctx = await reader.read_context(CONVERSATION)
    print(f"분위기: {ctx.atmosphere}")
    print(f"긴장도: {ctx.tension_level:.0%}")
    print(f"핵심 역학: {ctx.key_dynamics}")
    print(f"미해결: {ctx.unresolved}")
    print(f"숨은 감정: {ctx.emotional_undercurrent}")

    print("\n" + "=" * 70)
    print("성격별 반응")
    print("=" * 70)

    for persona in PERSONAS:
        social = SocialCognition(
            llm=llm,
            name=persona["name"],
            ocean=persona["ocean"],
            role=persona["role"],
        )
        strategy = await social.decide_strategy(CONVERSATION, context=ctx)

        o = persona["ocean"]
        print(f"\n{'─' * 60}")
        print(f"👤 {persona['name']} ({persona['role']})")
        print(f"   OCEAN: O={o.openness:.1f} C={o.conscientiousness:.1f} E={o.extraversion:.1f} A={o.agreeableness:.1f} N={o.neuroticism:.1f}")
        print(f"   전략: {strategy.strategy}")
        print(f"   톤: {strategy.tone}")
        print(f"   적극성: {strategy.initiative_level:.0%}")
        print(f"   스타일 미러링: {'예' if strategy.mirror_style else '아니오'}")
        print(f"   응답: {strategy.response}")
        print(f"   이유: {strategy.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
