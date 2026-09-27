from datetime import datetime, timezone

from app.agents.models import Agent
from app.experience.models import Experience
from app.experience.store import ExperienceStore
from app.tasks.allocator import TaskAllocator
from app.tasks.task import Task
from app.trust.engine import TrustEngine

from .models import ExecutionResult


class TaskExecutionEngine:
    def __init__(
        self,
        allocator: TaskAllocator,
        experience_store: ExperienceStore,
        trust_engine: TrustEngine,
    ) -> None:
        self._allocator = allocator
        self._experience_store = experience_store
        self._trust_engine = trust_engine

    def execute(
        self,
        task: Task,
        agents: list[Agent],
        success: bool,
        latency_ms: float,
        resource_used: float,
        failure_reason: str | None = None,
        timestamp: datetime | None = None,
    ) -> ExecutionResult:
        allocation = self._allocator.allocate(task, agents)

        if allocation is None:
            return ExecutionResult(
                task_id=task.task_id,
                agent_id=None,
                success=False,
                latency_ms=latency_ms,
                resource_used=resource_used,
                failure_reason="no_eligible_agent",
            )

        agent = next(
            agent
            for agent in agents
            if agent.agent_id == allocation.agent_id
        )

        experience = Experience(
            experience_id=f"{task.task_id}:{agent.agent_id}:{len(self._experience_store.list()) + 1}",
            task_id=task.task_id,
            agent_id=agent.agent_id,
            success=success,
            latency_ms=latency_ms,
            resource_used=resource_used,
            failure_reason=failure_reason,
            timestamp=timestamp or datetime.now(timezone.utc),
        )

        self._experience_store.record(experience)

        if success:
            trust_update = self._trust_engine.record_success(agent.trust)
        else:
            trust_update = self._trust_engine.record_failure(agent.trust)

        agent.trust = trust_update.new_trust

        return ExecutionResult(
            task_id=task.task_id,
            agent_id=agent.agent_id,
            success=success,
            latency_ms=latency_ms,
            resource_used=resource_used,
            failure_reason=failure_reason,
        )