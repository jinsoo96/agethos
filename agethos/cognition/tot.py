"""Tree of Thoughts — 복잡한 의사결정을 위한 분기 탐색.

Yao et al. (2023) + 서베이 논문(2503.23037) 기반.
BFS/DFS로 여러 추론 경로를 탐색하고, 가장 유망한 경로를 선택.
단순 CoT로는 부족한 복잡한 의사결정 상황에서 사용.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agethos.llm.base import LLMAdapter


class ThoughtNode(BaseModel):
    """사고 트리의 노드."""

    id: int = 0
    content: str = ""
    score: float = 0.0
    parent_id: int | None = None
    children_ids: list[int] = Field(default_factory=list)
    depth: int = 0


class TreeOfThoughts:
    """Tree of Thoughts 엔진 — BFS 기반 분기 탐색.

    Usage::

        tot = TreeOfThoughts(llm=llm_adapter)
        result = await tot.solve(
            problem="Should I accept the job offer or negotiate?",
            context="Current salary: 80k, Offer: 95k, Market rate: 110k",
            n_branches=3,
            max_depth=2,
        )
        # result = {"best_path": [...], "conclusion": "...", "confidence": 0.85}
    """

    def __init__(self, llm: LLMAdapter):
        self._llm = llm

    async def solve(
        self,
        problem: str,
        context: str = "",
        n_branches: int = 3,
        max_depth: int = 2,
    ) -> dict:
        """BFS 기반 Tree of Thoughts 탐색.

        Args:
            problem: 해결할 문제.
            context: 추가 맥락.
            n_branches: 각 노드에서 생성할 분기 수.
            max_depth: 최대 탐색 깊이.

        Returns:
            {"best_path": [str], "conclusion": str, "confidence": float}
        """
        nodes: list[ThoughtNode] = []
        node_counter = 0

        # Root node
        root = ThoughtNode(id=node_counter, content=problem, depth=0)
        nodes.append(root)
        node_counter += 1

        # BFS
        frontier = [root]
        for depth in range(max_depth):
            next_frontier: list[ThoughtNode] = []
            for parent in frontier:
                branches = await self._generate_branches(
                    problem, parent.content, context, n_branches
                )
                for branch_text in branches:
                    child = ThoughtNode(
                        id=node_counter,
                        content=branch_text,
                        parent_id=parent.id,
                        depth=depth + 1,
                    )
                    child.score = await self._evaluate(problem, branch_text, context)
                    parent.children_ids.append(node_counter)
                    nodes.append(child)
                    next_frontier.append(child)
                    node_counter += 1

            # Prune: keep top n_branches nodes
            next_frontier.sort(key=lambda n: n.score, reverse=True)
            frontier = next_frontier[:n_branches]

        # Find best leaf
        leaves = [n for n in nodes if not n.children_ids and n.depth > 0]
        if not leaves:
            return {"best_path": [problem], "conclusion": "", "confidence": 0.0}

        best = max(leaves, key=lambda n: n.score)

        # Trace path
        path: list[str] = []
        current: ThoughtNode | None = best
        while current is not None:
            path.append(current.content)
            current = next((n for n in nodes if n.id == current.parent_id), None) if current.parent_id is not None else None
        path.reverse()

        # Generate conclusion
        conclusion = await self._conclude(problem, path, context)

        return {
            "best_path": path,
            "conclusion": conclusion,
            "confidence": best.score,
        }

    async def _generate_branches(
        self, problem: str, current_thought: str, context: str, n: int
    ) -> list[str]:
        """현재 사고에서 N개 분기 생성."""
        prompt = (
            f"Problem: {problem}\n"
            + (f"Context: {context}\n" if context else "")
            + f"Current reasoning: {current_thought}\n\n"
            f"Generate {n} different next steps or approaches to continue this reasoning.\n"
            f"Each should explore a distinct direction.\n\n"
            f'Respond in JSON: {{"branches": ["step1", "step2", "step3"]}}'
        )
        try:
            data = await self._llm.generate_json(
                system_prompt="You explore multiple reasoning paths for complex decisions.",
                user_prompt=prompt,
            )
            return data.get("branches", [])[:n]
        except Exception:
            return [current_thought]

    async def _evaluate(self, problem: str, thought: str, context: str) -> float:
        """사고 경로의 유망도를 0~1로 평가."""
        prompt = (
            f"Problem: {problem}\n"
            + (f"Context: {context}\n" if context else "")
            + f"Proposed reasoning step: {thought}\n\n"
            f"Rate how promising this reasoning direction is (0.0 = dead end, 1.0 = very promising).\n"
            f'Respond in JSON: {{"score": <float>}}'
        )
        try:
            data = await self._llm.generate_json(
                system_prompt="You evaluate reasoning paths for quality and promise.",
                user_prompt=prompt,
            )
            return max(0.0, min(1.0, float(data.get("score", 0.5))))
        except Exception:
            return 0.5

    async def _conclude(self, problem: str, path: list[str], context: str) -> str:
        """최적 경로에서 최종 결론 도출."""
        path_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(path))
        prompt = (
            f"Problem: {problem}\n"
            + (f"Context: {context}\n" if context else "")
            + f"Best reasoning path:\n{path_text}\n\n"
            f"Based on this reasoning, provide a clear, actionable conclusion."
        )
        try:
            return await self._llm.generate(
                system_prompt="You synthesize reasoning into clear conclusions.",
                user_prompt=prompt,
            )
        except Exception:
            return path[-1] if path else ""
