from .task import Task
from ..agents.models import Agent, AgentStatus


class CandidateFilter:
    def filter(
        self,
        task: Task,
        agents: list[Agent],
    ) -> list[Agent]:
        candidates: list[Agent] = []

        for agent in agents:
            if agent.status != AgentStatus.ONLINE:
                continue

            if not all(
                capability in agent.capabilities
                for capability in task.required_capabilities
            ):
                continue

            if agent.trust < task.min_trust:
                continue

            if agent.latency_ms > task.max_latency_ms:
                continue

            if agent.resource < task.resource_requirement:
                continue

            candidates.append(agent)

        return candidates