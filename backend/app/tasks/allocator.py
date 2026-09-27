from dataclasses import dataclass

from ..agents.models import Agent
from .filter import CandidateFilter
from .scoring import AgentScorer
from .task import Task


@dataclass(frozen=True)
class AllocationResult:
    task_id: str
    agent_id: str
    score: float


class TaskAllocator:
    def __init__(
        self,
        candidate_filter: CandidateFilter,
        scorer: AgentScorer,
    ) -> None:
        self._candidate_filter = candidate_filter
        self._scorer = scorer

    def get_candidates(
        self,
        task: Task,
        agents: list[Agent],
    ) -> list[Agent]:
        return self._candidate_filter.filter(
            task,
            agents,
        )    

    def allocate(
        self,
        task: Task,
        agents: list[Agent],
    ) -> AllocationResult | None:
        candidates = self._candidate_filter.filter(
            task,
            agents,
        )

        if not candidates:
            return None

        scored_candidates = [
            (agent, self._scorer.score(agent))
            for agent in candidates
        ]

        selected_agent, selected_score = max(
            scored_candidates,
            key=lambda item: item[1],
        )

        return AllocationResult(
            task_id=task.task_id,
            agent_id=selected_agent.agent_id,
            score=selected_score,
        )