from collections.abc import Iterable

from app.experience.models import Experience

from .models import FitnessResult


class FitnessEngine:
    def evaluate(
        self,
        experiences: Iterable[Experience],
        latency_target_ms: float = 100.0,
    ) -> FitnessResult:
        experiences = list(experiences)

        if not experiences:
            return FitnessResult(
                fitness=0.0,
                success_rate=0.0,
                latency_score=0.0,
                resource_score=0.0,
                failure_rate=0.0,
            )

        total = len(experiences)
        successful = sum(1 for experience in experiences if experience.success)
        failed = total - successful

        success_rate = successful / total
        failure_rate = failed / total

        average_latency = sum(
            experience.latency_ms for experience in experiences
        ) / total

        average_resource = sum(
            experience.resource_used for experience in experiences
        ) / total

        latency_score = max(
            0.0,
            min(1.0, 1.0 - (average_latency / latency_target_ms)),
        )

        resource_score = max(
            0.0,
            min(1.0, 1.0 - average_resource),
        )

        fitness = (
            0.50 * success_rate
            + 0.20 * latency_score
            + 0.15 * resource_score
            + 0.15 * (1.0 - failure_rate)
        )

        return FitnessResult(
            fitness=round(max(0.0, min(1.0, fitness)), 6),
            success_rate=round(success_rate, 6),
            latency_score=round(latency_score, 6),
            resource_score=round(resource_score, 6),
            failure_rate=round(failure_rate, 6),
        )