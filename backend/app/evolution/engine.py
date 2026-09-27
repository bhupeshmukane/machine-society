import random
from collections.abc import Callable, Sequence

from app.policy.models import Policy

from .models import GenerationResult, Individual


PolicyFitness = Callable[[Policy], float]


class EvolutionEngine:
    def __init__(
        self,
        fitness_function: PolicyFitness,
        population_size: int = 10,
        mutation_rate: float = 0.10,
        mutation_strength: float = 0.05,
        elite_count: int = 2,
        tournament_size: int = 3,
        seed: int | None = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be at least 2")

        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")

        if mutation_strength < 0.0:
            raise ValueError("mutation_strength must be non-negative")

        if elite_count < 1 or elite_count >= population_size:
            raise ValueError(
                "elite_count must be at least 1 and smaller than population_size"
            )

        if tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")

        self.fitness_function = fitness_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self._rng = random.Random(seed)

    @staticmethod
    def _normalize_weights(values: Sequence[float]) -> tuple[float, ...]:
        bounded = tuple(max(0.0, min(1.0, value)) for value in values)
        total = sum(bounded)

        if total == 0.0:
            equal = 1.0 / len(bounded)
            return tuple(equal for _ in bounded)

        return tuple(value / total for value in bounded)

    def _mutate_value(self, value: float) -> float:
        if self._rng.random() > self.mutation_rate:
            return value

        return max(
            0.0,
            min(
                1.0,
                value + self._rng.gauss(0.0, self.mutation_strength),
            ),
        )

    def mutate(self, policy: Policy) -> Policy:
        values = (
            self._mutate_value(policy.trust_weight),
            self._mutate_value(policy.capability_weight),
            self._mutate_value(policy.latency_weight),
            self._mutate_value(policy.resource_weight),
        )

        normalized = self._normalize_weights(values)

        return Policy(
            trust_weight=normalized[0],
            capability_weight=normalized[1],
            latency_weight=normalized[2],
            resource_weight=normalized[3],
        )

    def crossover(self, first: Policy, second: Policy) -> Policy:
        alpha = self._rng.random()

        values = (
            alpha * first.trust_weight
            + (1.0 - alpha) * second.trust_weight,
            alpha * first.capability_weight
            + (1.0 - alpha) * second.capability_weight,
            alpha * first.latency_weight
            + (1.0 - alpha) * second.latency_weight,
            alpha * first.resource_weight
            + (1.0 - alpha) * second.resource_weight,
        )

        normalized = self._normalize_weights(values)

        return Policy(
            trust_weight=normalized[0],
            capability_weight=normalized[1],
            latency_weight=normalized[2],
            resource_weight=normalized[3],
        )

    def _evaluate(self, policies: Sequence[Policy]) -> tuple[Individual, ...]:
        individuals = [
            Individual(
                policy=policy,
                fitness=self.fitness_function(policy),
            )
            for policy in policies
        ]

        return tuple(
            sorted(
                individuals,
                key=lambda individual: individual.fitness,
                reverse=True,
            )
        )

    def _select_parent(
        self,
        population: Sequence[Individual],
    ) -> Individual:
        candidates = self._rng.sample(
            population,
            min(self.tournament_size, len(population)),
        )

        return max(
            candidates,
            key=lambda individual: individual.fitness,
        )

    def evolve(
        self,
        initial_population: Sequence[Policy],
        generations: int = 1,
    ) -> list[GenerationResult]:
        if len(initial_population) != self.population_size:
            raise ValueError(
                "initial_population size must equal population_size"
            )

        if generations < 1:
            raise ValueError("generations must be at least 1")

        population = self._evaluate(initial_population)
        results: list[GenerationResult] = []

        for generation in range(generations):
            best_fitness = population[0].fitness
            average_fitness = sum(
                individual.fitness for individual in population
            ) / len(population)

            results.append(
                GenerationResult(
                    generation=generation,
                    population=population,
                    best_fitness=round(best_fitness, 6),
                    average_fitness=round(average_fitness, 6),
                )
            )

            if generation == generations - 1:
                break

            next_population = [
                individual.policy
                for individual in population[: self.elite_count]
            ]

            while len(next_population) < self.population_size:
                parent_a = self._select_parent(population)
                parent_b = self._select_parent(population)

                child = self.crossover(
                    parent_a.policy,
                    parent_b.policy,
                )
                child = self.mutate(child)

                next_population.append(child)

            population = self._evaluate(next_population)

        return results