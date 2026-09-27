from collections.abc import Callable, Iterable

from app.experience.models import Experience
from app.policy.models import Policy

from .engine import FitnessEngine


class PolicyFitnessEvaluator:
    def __init__(
        self,
        fitness_engine: FitnessEngine | None = None,
    ) -> None:
        self._fitness_engine = fitness_engine or FitnessEngine()

    def evaluate(
        self,
        policy: Policy,
        experiences: Iterable[Experience],
    ) -> float:
        result = self._fitness_engine.evaluate(experiences)

        fitness = (
            policy.trust_weight * result.success_rate
            + policy.capability_weight * result.success_rate
            + policy.latency_weight * result.latency_score
            + policy.resource_weight * result.resource_score
        )

        return round(
            max(0.0, min(1.0, fitness)),
            6,
        )

    def create_fitness_function(
        self,
        experiences: Iterable[Experience],
    ) -> Callable[[Policy], float]:
        experiences = list(experiences)

        def fitness_function(policy: Policy) -> float:
            return self.evaluate(
                policy,
                experiences,
            )

        return fitness_function