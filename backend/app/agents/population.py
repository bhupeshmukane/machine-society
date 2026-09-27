from .factory import VirtualAgentFactory
from .models import Agent
from .registry import AgentRegistry


class PopulationManager:
    def __init__(
        self,
        registry: AgentRegistry,
        factory: VirtualAgentFactory,
    ) -> None:
        self._registry = registry
        self._factory = factory

    def create_virtual_population(
        self,
        count: int,
        start_index: int = 1,
    ) -> list[Agent]:
        agents = self._factory.create_many(
            count=count,
            start_index=start_index,
        )

        for agent in agents:
            self._registry.register(agent)

        return agents

    def get_population(self) -> list[Agent]:
        return self._registry.list_agents()