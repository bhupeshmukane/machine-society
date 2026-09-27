from dataclasses import dataclass
from datetime import datetime, timezone

from .models import Agent, AgentStatus
from .registry import AgentRegistry


@dataclass(frozen=True)
class HeartbeatConfig:
    degraded_after_seconds: float = 10.0
    offline_after_seconds: float = 20.0


class HeartbeatManager:
    def __init__(
        self,
        registry: AgentRegistry,
        config: HeartbeatConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or HeartbeatConfig()

    def record_heartbeat(
        self,
        agent_id: str,
        timestamp: datetime | None = None,
    ) -> Agent | None:
        agent = self._registry.get(agent_id)

        if agent is None:
            return None

        heartbeat_time = timestamp or datetime.now(timezone.utc)

        agent.last_heartbeat = heartbeat_time
        agent.status = AgentStatus.ONLINE

        return agent

    def evaluate_agent(
        self,
        agent_id: str,
        now: datetime | None = None,
    ) -> Agent | None:
        agent = self._registry.get(agent_id)

        if agent is None:
            return None

        current_time = now or datetime.now(timezone.utc)

        elapsed = (
            current_time - agent.last_heartbeat
        ).total_seconds()

        if elapsed > self._config.offline_after_seconds:
            agent.status = AgentStatus.OFFLINE
        elif elapsed > self._config.degraded_after_seconds:
            agent.status = AgentStatus.DEGRADED
        else:
            agent.status = AgentStatus.ONLINE

        return agent

    def evaluate_all(
        self,
        now: datetime | None = None,
    ) -> list[Agent]:
        current_time = now or datetime.now(timezone.utc)

        agents = self._registry.list_agents()

        for agent in agents:
            self.evaluate_agent(
                agent.agent_id,
                current_time,
            )

        return agents

    def get_agent(self, agent_id: str) -> Agent | None:
        return self._registry.get(agent_id)