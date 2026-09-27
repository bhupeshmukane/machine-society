from .models import Policy


class PolicyEngine:
    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or Policy()

    def set_policy(self, policy: Policy) -> None:
        self.policy = policy

    def score(
        self,
        trust: float,
        capability: float,
        latency: float,
        resource: float,
    ) -> float:
        score = (
            self.policy.trust_weight * trust
            + self.policy.capability_weight * capability
            + self.policy.latency_weight * (1.0 - latency)
            + self.policy.resource_weight * (1.0 - resource)
        )

        return round(max(0.0, min(1.0, score)), 6)