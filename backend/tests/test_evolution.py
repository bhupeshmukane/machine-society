import pytest

from app.evolution.engine import EvolutionEngine
from app.policy.models import Policy


def make_population(size: int = 6) -> list[Policy]:
    return [
        Policy(
            trust_weight=0.40,
            capability_weight=0.30,
            latency_weight=0.20,
            resource_weight=0.10,
        )
        for _ in range(size)
    ]


def test_population_size_is_preserved():
    engine = EvolutionEngine(
        fitness_function=lambda policy: policy.trust_weight,
        population_size=6,
        seed=42,
    )

    results = engine.evolve(
        make_population(),
        generations=3,
    )

    assert len(results) == 3

    for result in results:
        assert len(result.population) == 6


def test_policy_weights_remain_normalized():
    engine = EvolutionEngine(
        fitness_function=lambda policy: policy.trust_weight,
        population_size=6,
        mutation_rate=1.0,
        seed=42,
    )

    results = engine.evolve(
        make_population(),
        generations=4,
    )

    for result in results:
        for individual in result.population:
            policy = individual.policy

            total = (
                policy.trust_weight
                + policy.capability_weight
                + policy.latency_weight
                + policy.resource_weight
            )

            assert total == pytest.approx(1.0)
            assert 0.0 <= policy.trust_weight <= 1.0
            assert 0.0 <= policy.capability_weight <= 1.0
            assert 0.0 <= policy.latency_weight <= 1.0
            assert 0.0 <= policy.resource_weight <= 1.0


def test_fitness_is_sorted_descending():
    engine = EvolutionEngine(
        fitness_function=lambda policy: policy.trust_weight,
        population_size=6,
        seed=42,
    )

    results = engine.evolve(
        make_population(),
        generations=2,
    )

    for result in results:
        fitness_values = [
            individual.fitness
            for individual in result.population
        ]

        assert fitness_values == sorted(
            fitness_values,
            reverse=True,
        )


def test_elitism_preserves_best_policy():
    population = [
        Policy(
            trust_weight=0.90,
            capability_weight=0.05,
            latency_weight=0.03,
            resource_weight=0.02,
        )
    ] + make_population(5)

    engine = EvolutionEngine(
        fitness_function=lambda policy: policy.trust_weight,
        population_size=6,
        elite_count=1,
        mutation_rate=1.0,
        seed=42,
    )

    results = engine.evolve(
        population,
        generations=2,
    )

    first_best = results[0].population[0]

    second_population = results[1].population

    assert first_best in second_population


def test_same_seed_is_reproducible():
    population = make_population()

    engine_a = EvolutionEngine(
        fitness_function=lambda policy: policy.trust_weight,
        population_size=6,
        mutation_rate=0.5,
        seed=42,
    )

    engine_b = EvolutionEngine(
        fitness_function=lambda policy: policy.trust_weight,
        population_size=6,
        mutation_rate=0.5,
        seed=42,
    )

    results_a = engine_a.evolve(population, generations=5)
    results_b = engine_b.evolve(population, generations=5)

    assert results_a == results_b


def test_different_seed_can_produce_different_evolution():
    population = make_population()

    engine_a = EvolutionEngine(
        fitness_function=lambda policy: policy.trust_weight,
        population_size=6,
        mutation_rate=1.0,
        seed=42,
    )

    engine_b = EvolutionEngine(
        fitness_function=lambda policy: policy.trust_weight,
        population_size=6,
        mutation_rate=1.0,
        seed=99,
    )

    results_a = engine_a.evolve(population, generations=5)
    results_b = engine_b.evolve(population, generations=5)

    assert results_a != results_b


def test_generation_statistics_are_bounded():
    engine = EvolutionEngine(
        fitness_function=lambda policy: policy.trust_weight,
        population_size=6,
        seed=42,
    )

    results = engine.evolve(
        make_population(),
        generations=3,
    )

    for result in results:
        assert 0.0 <= result.best_fitness <= 1.0
        assert 0.0 <= result.average_fitness <= 1.0


def test_invalid_population_size_is_rejected():
    with pytest.raises(ValueError):
        EvolutionEngine(
            fitness_function=lambda policy: 0.5,
            population_size=1,
        )


def test_invalid_generation_count_is_rejected():
    engine = EvolutionEngine(
        fitness_function=lambda policy: 0.5,
        population_size=4,
        seed=42,
    )

    with pytest.raises(ValueError):
        engine.evolve(make_population(4), generations=0)

def test_best_policy_returns_highest_fitness_policy():
    population = [
        Policy(
            trust_weight=0.90,
            capability_weight=0.05,
            latency_weight=0.03,
            resource_weight=0.02,
        ),
        Policy(
            trust_weight=0.40,
            capability_weight=0.30,
            latency_weight=0.20,
            resource_weight=0.10,
        ),
        Policy(
            trust_weight=0.20,
            capability_weight=0.30,
            latency_weight=0.30,
            resource_weight=0.20,
        ),
        Policy(
            trust_weight=0.25,
            capability_weight=0.25,
            latency_weight=0.25,
            resource_weight=0.25,
        ),
    ]

    engine = EvolutionEngine(
        fitness_function=lambda policy: policy.trust_weight,
        population_size=4,
        seed=42,
    )

    results = engine.evolve(
        population,
        generations=2,
    )

    best_policy = engine.best_policy(results)

    assert best_policy == results[-1].population[0].policy

def test_best_policy_rejects_empty_results():
    engine = EvolutionEngine(
        fitness_function=lambda policy: 0.5,
        population_size=4,
    )

    with pytest.raises(ValueError):
        engine.best_policy([])