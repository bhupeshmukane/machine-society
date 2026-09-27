from dataclasses import dataclass


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    success_rate: float
    latency_score: float
    resource_score: float
    failure_rate: float