from collections.abc import Sequence

from app.experience.models import Experience
from app.fitness.policy_evaluator import PolicyFitnessEvaluator
from app.policy.engine import PolicyEngine
from app.policy.models import Policy

from .engine import EvolutionEngine
from .models import GenerationResult


class PolicyEvolutionCoordinator:
    def __init__(
        self,
        policy_engine: PolicyEngine,
        fitness_evaluator: PolicyFitnessEvaluator | None = None,
    ) -> None:
        self._policy_engine = policy_engine
        self._fitness_evaluator = (
            fitness_evaluator or PolicyFitnessEvaluator()
        )

    def evolve(
        self,
        experiences: Sequence[Experience],
        initial_population: Sequence[Policy],
        generations: int = 3,
        population_size: int | None = None,
        seed: int | None = None,
    ) -> tuple[Policy, list[GenerationResult]]:
        if population_size is None:
            population_size = len(initial_population)

        fitness_function = (
            self._fitness_evaluator.create_fitness_function(
                experiences
            )
        )

        evolution_engine = EvolutionEngine(
            fitness_function=fitness_function,
            population_size=population_size,
            seed=seed,
        )

        results = evolution_engine.evolve(
            initial_population=initial_population,
            generations=generations,
        )

        best_policy = evolution_engine.best_policy(results)

        self._policy_engine.set_policy(best_policy)

        return best_policy, results