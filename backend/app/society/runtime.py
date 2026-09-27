from app.agents.models import Agent
from app.experience.models import Experience
from app.experience.store import ExperienceStore
from app.policy.engine import PolicyEngine
from app.tasks.allocator import AllocationResult
from app.tasks.allocator import TaskAllocator
from app.tasks.task import Task
from app.trust.engine import TrustEngine

from .models import AgentPolicyScore, TaskExecutionResult


class SocietyRuntime:
    def __init__(
        self,
        agents: list[Agent],
        task_allocator: TaskAllocator,
        experience_store: ExperienceStore,
        trust_engine: TrustEngine,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        self._agents = agents
        self._task_allocator = task_allocator
        self._experience_store = experience_store
        self._trust_engine = trust_engine
        self._policy_engine = policy_engine or PolicyEngine()

    def allocate_task(
        self,
        task: Task,
    ):
        candidates = self._task_allocator._candidate_filter.filter(
            task,
            self._agents,
        )
        
        candidates = self._task_allocator.get_candidates(
            task,
            self._agents,
        )

        if not candidates:
            return None

        scored_candidates = []

        for agent in candidates:
            if task.required_capabilities:
                matched = sum(
                    capability in agent.capabilities
                    for capability in task.required_capabilities
                )

                capability_score = (
                    matched / len(task.required_capabilities)
                )
            else:
                capability_score = 1.0

            policy_score = self.score_agent(
                agent,
                capability_score,
            )

            scored_candidates.append(
                (agent, policy_score.score)
            )

        selected_agent, selected_score = max(
            scored_candidates,
            key=lambda item: item[1],
        )

        return AllocationResult(
            task_id=task.task_id,
            agent_id=selected_agent.agent_id,
            score=selected_score,
        )

    def score_agent(
        self,
        agent: Agent,
        capability_score: float,
    ) -> AgentPolicyScore:
        score = self._policy_engine.score(
            trust=agent.trust,
            capability=capability_score,
            latency=min(agent.latency_ms / 100.0, 1.0),
            resource=agent.load,
        )

        return AgentPolicyScore(
            agent_id=agent.agent_id,
            score=score,
        )

    def record_outcome(
        self,
        task: Task,
        experience: Experience,
    ) -> TaskExecutionResult:
        self._experience_store.record(experience)

        agent = next(
            (
                agent
                for agent in self._agents
                if agent.agent_id == experience.agent_id
            ),
            None,
        )

        if agent is None:
            raise ValueError(
                f"Agent not found: {experience.agent_id}"
            )

        if experience.success:
            trust_update = self._trust_engine.record_success(
                agent.trust
            )
        else:
            trust_update = self._trust_engine.record_failure(
                agent.trust
            )

        agent.trust = trust_update.new_trust

        allocation = self._task_allocator.allocate(
            task,
            self._agents,
        )

        return TaskExecutionResult(
            allocation=allocation,
            success=experience.success,
            experience_id=experience.experience_id,
            trust_update_reason=trust_update.reason,
        )