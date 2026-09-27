import random

from .models import Agent, AgentType


DEFAULT_CAPABILITIES = [
    "temperature_sensing",
    "occupancy_sensing",
    "light_sensing",
    "actuation",
]


class VirtualAgentFactory:
    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def create(self, agent_id: str) -> Agent:
        capabilities = self._random.sample(
            DEFAULT_CAPABILITIES,
            k=self._random.randint(1, len(DEFAULT_CAPABILITIES)),
        )

        return Agent(
            agent_id=agent_id,
            agent_type=AgentType.VIRTUAL,
            capabilities=capabilities,
            trust=round(self._random.uniform(0.70, 1.00), 3),
            load=round(self._random.uniform(0.10, 0.90), 3),
            latency_ms=round(self._random.uniform(20.0, 150.0), 2),
            resource=round(self._random.uniform(0.30, 1.00), 3),
        )

    def create_many(self, count: int, start_index: int = 1) -> list[Agent]:
        if count < 1:
            raise ValueError("count must be at least 1")

        return [
            self.create(f"A{index:02d}")
            for index in range(start_index, start_index + count)
        ]