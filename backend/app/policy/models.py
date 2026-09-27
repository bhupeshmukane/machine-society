from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    trust_weight: float = 0.35
    capability_weight: float = 0.30
    latency_weight: float = 0.20
    resource_weight: float = 0.15