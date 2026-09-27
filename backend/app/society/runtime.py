from app.agents.models import Agent
from app.experience.models import Experience
from app.experience.store import ExperienceStore
from app.tasks.allocator import TaskAllocator
from app.tasks.task import Task
from app.trust.engine import TrustEngine

from .models import TaskExecutionResult


class SocietyRuntime:
    def __init__(
        self,
        agents: list[Agent],
        task_allocator: TaskAllocator,
        experience_store: ExperienceStore,
        trust_engine: TrustEngine,
    ) -> None:
        self._agents = agents
        self._task_allocator = task_allocator
        self._experience_store = experience_store
        self._trust_engine = trust_engine

    def allocate_task(
        self,
        task: Task,
    ):
        return self._task_allocator.allocate(
            task,
            self._agents,
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