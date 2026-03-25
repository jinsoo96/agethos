"""멀티에이전트 협력 모듈 — 여러 Brain이 팀으로 토론/합의.

CAMEL (Li et al., 2023)의 inception prompting과
Society of Mind (Minsky) 패턴을 결합.
OCEAN 차이가 자연스러운 역할 분화를 유도.

프로토콜:
- round_robin: 순서대로 발언
- debate: 찬반 토론 후 합의
- hierarchical: 리더가 정리, 나머지가 의견 제시
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agethos.models import CollaborationMessage, CollaborationResult

if TYPE_CHECKING:
    from agethos.brain import Brain


async def team_discuss(
    agents: dict[str, "Brain"],
    topic: str,
    protocol: str = "round_robin",
    max_rounds: int = 3,
) -> CollaborationResult:
    """여러 인격체가 주제에 대해 토론.

    Args:
        agents: {"역할명": Brain} 딕셔너리.
        topic: 토론 주제.
        protocol: "round_robin" | "debate" | "hierarchical".
        max_rounds: 최대 라운드 수.

    Returns:
        CollaborationResult with messages and consensus.

    Usage::

        result = await team_discuss(
            agents={"PM": pm_brain, "Engineer": eng_brain, "Designer": design_brain},
            topic="Should we rewrite the auth system?",
            protocol="round_robin",
            max_rounds=3,
        )
        print(result.consensus)
    """
    if protocol == "debate":
        return await _debate(agents, topic, max_rounds)
    elif protocol == "hierarchical":
        return await _hierarchical(agents, topic, max_rounds)
    else:
        return await _round_robin(agents, topic, max_rounds)


async def _round_robin(
    agents: dict[str, "Brain"],
    topic: str,
    max_rounds: int,
) -> CollaborationResult:
    """라운드 로빈 — 순서대로 발언, 이전 발언을 컨텍스트로 전달."""
    messages: list[CollaborationMessage] = []
    agent_names = list(agents.keys())

    for round_num in range(max_rounds):
        for name in agent_names:
            brain = agents[name]

            # Build context from previous messages
            if messages:
                context_lines = [
                    f"{m.agent_name}: {m.content}" for m in messages[-len(agent_names) * 2:]
                ]
                context = "\n".join(context_lines)
                prompt = f"Team discussion on: {topic}\n\nPrevious discussion:\n{context}\n\nYour turn as {name}. Share your perspective."
            else:
                prompt = f"Team discussion on: {topic}\n\nYou are {name}. Share your opening perspective."

            reply = await brain.chat(prompt, context=f"[Team discussion round {round_num + 1}]")
            messages.append(CollaborationMessage(
                agent_name=name,
                content=reply,
                round=round_num,
            ))

    # Generate consensus from the last speaker
    last_brain = agents[agent_names[-1]]
    summary_lines = [f"{m.agent_name}: {m.content}" for m in messages]
    consensus_prompt = (
        f"Summarize the team consensus on: {topic}\n\n"
        f"Full discussion:\n" + "\n".join(summary_lines) + "\n\n"
        "What does the team agree on? What remains unresolved? Be concise."
    )
    consensus = await last_brain.chat(consensus_prompt)

    return CollaborationResult(
        topic=topic,
        messages=messages,
        consensus=consensus,
        total_rounds=max_rounds,
    )


async def _debate(
    agents: dict[str, "Brain"],
    topic: str,
    max_rounds: int,
) -> CollaborationResult:
    """찬반 토론 — 에이전트를 찬성/반대로 나눠 토론 후 합의."""
    messages: list[CollaborationMessage] = []
    agent_names = list(agents.keys())

    if len(agent_names) < 2:
        return await _round_robin(agents, topic, max_rounds)

    # Split into two sides
    mid = len(agent_names) // 2
    pro_names = agent_names[:mid] or agent_names[:1]
    con_names = agent_names[mid:] or agent_names[1:]

    for round_num in range(max_rounds):
        # Pro side
        for name in pro_names:
            brain = agents[name]
            context = "\n".join(f"{m.agent_name}: {m.content}" for m in messages[-6:])
            prompt = (
                f"Debate on: {topic}\n\n"
                f"You are {name}, arguing IN FAVOR.\n"
                + (f"Previous:\n{context}\n\n" if context else "")
                + "Present your argument."
            )
            reply = await brain.chat(prompt)
            messages.append(CollaborationMessage(agent_name=name, content=reply, round=round_num))

        # Con side
        for name in con_names:
            brain = agents[name]
            context = "\n".join(f"{m.agent_name}: {m.content}" for m in messages[-6:])
            prompt = (
                f"Debate on: {topic}\n\n"
                f"You are {name}, arguing AGAINST.\n"
                f"Previous:\n{context}\n\n"
                "Present your counterargument."
            )
            reply = await brain.chat(prompt)
            messages.append(CollaborationMessage(agent_name=name, content=reply, round=round_num))

    # Consensus from first agent (as moderator)
    first_brain = agents[agent_names[0]]
    all_text = "\n".join(f"{m.agent_name}: {m.content}" for m in messages)
    consensus = await first_brain.chat(
        f"As moderator, summarize the debate on: {topic}\n\n{all_text}\n\n"
        "What are the key arguments for and against? What's the balanced conclusion?"
    )

    return CollaborationResult(
        topic=topic,
        messages=messages,
        consensus=consensus,
        total_rounds=max_rounds,
    )


async def _hierarchical(
    agents: dict[str, "Brain"],
    topic: str,
    max_rounds: int,
) -> CollaborationResult:
    """계층적 — 첫 번째 에이전트가 리더, 나머지가 의견 제시 후 리더가 정리."""
    messages: list[CollaborationMessage] = []
    agent_names = list(agents.keys())

    leader_name = agent_names[0]
    member_names = agent_names[1:]
    leader_brain = agents[leader_name]

    for round_num in range(max_rounds):
        # Leader frames the question
        if round_num == 0:
            leader_prompt = f"As team lead ({leader_name}), frame the discussion: {topic}\nWhat specific questions should the team address?"
        else:
            recent = "\n".join(f"{m.agent_name}: {m.content}" for m in messages[-len(agent_names) * 2:])
            leader_prompt = f"Based on team input:\n{recent}\n\nAs {leader_name}, what follow-up questions or directions should we explore?"

        leader_reply = await leader_brain.chat(leader_prompt)
        messages.append(CollaborationMessage(agent_name=leader_name, content=leader_reply, round=round_num))

        # Members respond
        for name in member_names:
            brain = agents[name]
            context = "\n".join(f"{m.agent_name}: {m.content}" for m in messages[-6:])
            prompt = f"{leader_name} (lead) asked:\n{leader_reply}\n\nPrevious context:\n{context}\n\nYour input as {name}:"
            reply = await brain.chat(prompt)
            messages.append(CollaborationMessage(agent_name=name, content=reply, round=round_num))

    # Leader synthesizes
    all_text = "\n".join(f"{m.agent_name}: {m.content}" for m in messages)
    consensus = await leader_brain.chat(
        f"As {leader_name}, synthesize the team discussion on: {topic}\n\n{all_text}\n\n"
        "Provide the final decision/recommendation."
    )

    return CollaborationResult(
        topic=topic,
        messages=messages,
        consensus=consensus,
        total_rounds=max_rounds,
    )
