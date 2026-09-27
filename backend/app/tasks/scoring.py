from dataclasses import dataclass

from ..agents.models import Agent


@dataclass(frozen=True)
class ScoringWeights:
    trust: float = 0.30
    load: float = 0.20
    latency: float = 0.20
    resource: float = 0.15
    capability: float = 0.15


class AgentScorer:
    def __init__(
        self,
        weights: ScoringWeights | None = None,
    ) -> None:
        self._weights = weights or ScoringWeights()

    def score(self, agent: Agent) -> float:
        trust_score = agent.trust
        load_score = 1.0 - agent.load
        latency_score = 1.0 / (1.0 + agent.latency_ms / 100.0)
        resource_score = agent.resource
        capability_score = 1.0

        return (
            self._weights.trust * trust_score
            + self._weights.load * load_score
            + self._weights.latency * latency_score
            + self._weights.resource * resource_score
            + self._weights.capability * capability_score
        )