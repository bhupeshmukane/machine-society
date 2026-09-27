from dataclasses import dataclass

from app.policy.models import Policy


@dataclass(frozen=True)
class Individual:
    policy: Policy
    fitness: float = 0.0


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    population: tuple[Individual, ...]
    best_fitness: float
    average_fitness: float