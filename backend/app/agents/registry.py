from datetime import datetime, timezone

from .models import Agent, AgentStatus


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> Agent:
        self._agents[agent.agent_id] = agent
        return agent

    def get(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def update_heartbeat(self, agent_id: str) -> Agent | None:
        agent = self._agents.get(agent_id)

        if agent is None:
            return None

        agent.last_heartbeat = datetime.now(timezone.utc)
        agent.status = AgentStatus.ONLINE

        return agent

    def remove(self, agent_id: str) -> bool:
        if agent_id not in self._agents:
            return False

        del self._agents[agent_id]
        return True

    def count(self) -> int:
        return len(self._agents)