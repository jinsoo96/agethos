"""Transplant Adapter — 타 프레임워크에 뇌를 런타임 이식.

"몸은 다른 프레임워크, 뇌는 agethos" 비전의 핵심.

지원 프레임워크:
- CrewAI: Agent에 brain의 인격/기억/감정을 주입
- AutoGen: AssistantAgent에 인격 시스템 프롬프트 + 콜백 주입
- LangGraph: State에 brain 상태를 바인딩하는 노드 함수 생성
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agethos.brain import Brain


class TransplantAdapter:
    """Brain을 타 프레임워크에 이식하는 어댑터 기반 클래스."""

    def __init__(self, brain: Brain):
        self._brain = brain

    @property
    def brain(self) -> Brain:
        return self._brain

    def _render_system_prompt(self) -> str:
        """Brain의 전체 시스템 프롬프트 생성."""
        from agethos.export.adapters import export_brain
        return export_brain(self._brain, "system_prompt")


class CrewAITransplant(TransplantAdapter):
    """CrewAI Agent에 agethos 뇌를 이식.

    Usage::

        from crewai import Agent, Task, Crew
        from agethos import Brain
        from agethos.export.transplant import CrewAITransplant

        brain = Brain.build(persona={...}, llm="openai")
        transplant = CrewAITransplant(brain)

        agent = transplant.create_agent(
            tools=[...],            # CrewAI tools
            allow_delegation=True,
        )

        task = Task(description="...", agent=agent)
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
    """

    def create_agent(self, **crewai_kwargs) -> Any:
        """CrewAI Agent를 생성하고 agethos 인격을 이식.

        Args:
            **crewai_kwargs: CrewAI Agent에 전달할 추가 인자
                (tools, allow_delegation, verbose 등).

        Returns:
            crewai.Agent 인스턴스.
        """
        try:
            from crewai import Agent
        except ImportError as e:
            raise ImportError("pip install crewai") from e

        persona = self._brain.persona
        config = self._brain.export("crewai")

        # 시스템 프롬프트에 기억/감정 포함
        full_prompt = self._render_system_prompt()

        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=full_prompt,
            **crewai_kwargs,
        )

    async def sync_after_task(self, task_output: str) -> None:
        """CrewAI 태스크 완료 후 Brain 상태 동기화.

        태스크 결과를 기억으로 저장, 감정 업데이트.
        """
        await self._brain.observe(f"Completed task: {task_output[:300]}")


class AutoGenTransplant(TransplantAdapter):
    """AutoGen AssistantAgent에 agethos 뇌를 이식.

    Usage::

        from autogen import AssistantAgent, UserProxyAgent
        from agethos import Brain
        from agethos.export.transplant import AutoGenTransplant

        brain = Brain.build(persona={...}, llm="openai")
        transplant = AutoGenTransplant(brain)

        assistant = transplant.create_agent(
            llm_config={"model": "gpt-4o"},
        )

        user_proxy = UserProxyAgent(name="user")
        user_proxy.initiate_chat(assistant, message="Hello!")
    """

    def create_agent(self, **autogen_kwargs) -> Any:
        """AutoGen AssistantAgent를 생성하고 agethos 인격을 이식.

        Args:
            **autogen_kwargs: AutoGen AssistantAgent에 전달할 추가 인자
                (llm_config, code_execution_config 등).

        Returns:
            autogen.AssistantAgent 인스턴스.
        """
        try:
            from autogen import AssistantAgent
        except ImportError as e:
            raise ImportError("pip install pyautogen") from e

        persona = self._brain.persona
        system_message = self._render_system_prompt()

        return AssistantAgent(
            name=persona.name.replace(" ", "_"),
            system_message=system_message,
            **autogen_kwargs,
        )

    def create_reply_func(self):
        """AutoGen reply function으로 Brain.chat을 래핑.

        Brain의 인지 루프(기억 검색, 감정, 반성)를 AutoGen 대화에 통합.

        Usage::

            assistant = transplant.create_agent(llm_config={...})
            reply_func = transplant.create_reply_func()
            assistant.register_reply([autogen.Agent], reply_func)
        """
        import asyncio

        brain = self._brain

        def reply_func(recipient, messages, sender, config):
            last_msg = messages[-1].get("content", "") if messages else ""
            # Py3.12-safe: get_event_loop() raises with no running loop; probe the running one.
            try:
                asyncio.get_running_loop()
                running = True
            except RuntimeError:
                running = False
            if running:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    reply = pool.submit(asyncio.run, brain.chat(last_msg)).result()
            else:
                reply = asyncio.run(brain.chat(last_msg))
            return True, reply

        return reply_func


class LangGraphTransplant(TransplantAdapter):
    """LangGraph에 agethos 뇌를 이식.

    Brain의 인지 루프를 LangGraph 노드 함수로 변환.

    Usage::

        from langgraph.graph import StateGraph, MessagesState
        from agethos import Brain
        from agethos.export.transplant import LangGraphTransplant

        brain = Brain.build(persona={...}, llm="openai")
        transplant = LangGraphTransplant(brain)

        graph = StateGraph(MessagesState)
        graph.add_node("agent", transplant.as_node())
        graph.set_entry_point("agent")
        app = graph.compile()

        result = await app.ainvoke({"messages": [("user", "Hello!")]})
    """

    def as_node(self):
        """Brain.chat을 LangGraph 노드 함수로 변환.

        Returns:
            async 함수 (state → state).
        """
        brain = self._brain
        system_prompt = self._render_system_prompt()

        async def brain_node(state: dict) -> dict:
            messages = state.get("messages", [])

            # 마지막 사용자 메시지 추출
            last_user_msg = ""
            for msg in reversed(messages):
                if isinstance(msg, tuple):
                    role, content = msg
                    if role == "user":
                        last_user_msg = content
                        break
                elif isinstance(msg, dict):
                    if msg.get("role") == "user":
                        last_user_msg = msg.get("content", "")
                        break
                elif hasattr(msg, "type") and msg.type == "human":
                    last_user_msg = msg.content
                    break

            if last_user_msg:
                reply = await brain.chat(last_user_msg)
            else:
                reply = await brain.chat("Hello")

            # 응답을 메시지로 추가
            return {"messages": [("assistant", reply)]}

        return brain_node

    def get_state_snapshot(self) -> dict:
        """현재 Brain 상태를 LangGraph state에 주입 가능한 dict로 반환."""
        persona = self._brain.persona
        return {
            "brain_name": persona.name,
            "brain_emotion": persona.emotion.closest_emotion() if persona.emotion else "neutral",
            "brain_system_prompt": self._render_system_prompt(),
            "brain_memory_count": len(self._brain.social_patterns),
        }


def transplant(brain: Brain, framework: str, **kwargs) -> Any:
    """Brain을 지정 프레임워크에 이식 — 편의 함수.

    Args:
        brain: Brain 인스턴스.
        framework: 타겟 프레임워크 ("crewai", "autogen", "langgraph").
        **kwargs: 프레임워크별 추가 인자.

    Returns:
        프레임워크 에이전트 인스턴스.

    Usage::

        # CrewAI
        agent = transplant(brain, "crewai", tools=[...])

        # AutoGen
        agent = transplant(brain, "autogen", llm_config={...})

        # LangGraph
        node_func = transplant(brain, "langgraph")
    """
    adapters = {
        "crewai": lambda: CrewAITransplant(brain).create_agent(**kwargs),
        "autogen": lambda: AutoGenTransplant(brain).create_agent(**kwargs),
        "langgraph": lambda: LangGraphTransplant(brain).as_node(),
    }

    factory = adapters.get(framework.lower())
    if factory is None:
        supported = ", ".join(sorted(adapters.keys()))
        raise ValueError(f"Unknown framework: {framework!r}. Supported: {supported}")

    return factory()
