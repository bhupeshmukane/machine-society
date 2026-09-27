from dataclasses import dataclass


@dataclass(frozen=True)
class TrustUpdate:
    previous_trust: float
    new_trust: float
    delta: float
    reason: str


class TrustEngine:
    def __init__(
        self,
        success_reward: float = 0.02,
        failure_penalty: float = 0.05,
        timeout_penalty: float = 0.08,
        min_trust: float = 0.0,
        max_trust: float = 1.0,
    ) -> None:
        self._success_reward = success_reward
        self._failure_penalty = failure_penalty
        self._timeout_penalty = timeout_penalty
        self._min_trust = min_trust
        self._max_trust = max_trust

    def _clamp(self, value: float) -> float:
        clamped = max(
            self._min_trust,
            min(self._max_trust, value),
        )

        return round(clamped, 6)

    def record_success(self, trust: float) -> TrustUpdate:
        new_trust = self._clamp(
            trust + self._success_reward
        )

        return TrustUpdate(
            previous_trust=trust,
            new_trust=new_trust,
            delta=new_trust - trust,
            reason="success",
        )

    def record_failure(self, trust: float) -> TrustUpdate:
        new_trust = self._clamp(
            trust - self._failure_penalty
        )

        return TrustUpdate(
            previous_trust=trust,
            new_trust=new_trust,
            delta=new_trust - trust,
            reason="failure",
        )

    def record_timeout(self, trust: float) -> TrustUpdate:
        new_trust = self._clamp(
            trust - self._timeout_penalty
        )

        return TrustUpdate(
            previous_trust=trust,
            new_trust=new_trust,
            delta=new_trust - trust,
            reason="timeout",
        )