from app.evolution.engine import EvolutionEngine
from app.experience.models import Experience
from app.fitness.policy_evaluator import PolicyFitnessEvaluator
from app.policy.models import Policy


def make_experience(
    experience_id: str,
    success: bool,
    latency_ms: float,
    resource_used: float,
) -> Experience:
    return Experience(
        experience_id=experience_id,
        task_id="TASK-001",
        agent_id="A01",
        success=success,
        latency_ms=latency_ms,
        resource_used=resource_used,
        failure_reason=None if success else "failure",
    )


def test_policy_fitness_evaluator_returns_fitness():
    policy = Policy()

    experiences = [
        make_experience(
            "EXP-001",
            True,
            20.0,
            0.20,
        ),
        make_experience(
            "EXP-002",
            True,
            30.0,
            0.30,
        ),
    ]

    evaluator = PolicyFitnessEvaluator()

    fitness = evaluator.evaluate(
        policy,
        experiences,
    )

    assert 0.0 <= fitness <= 1.0


def test_policy_fitness_evaluator_returns_zero_for_empty_experiences():
    policy = Policy()

    evaluator = PolicyFitnessEvaluator()

    fitness = evaluator.evaluate(
        policy,
        [],
    )

    assert fitness == 0.0

def test_policy_is_part_of_fitness_evaluation():
    experiences = [
        make_experience(
            "EXP-001",
            True,
            20.0,
            0.20,
        ),
        make_experience(
            "EXP-002",
            False,
            120.0,
            0.80,
        ),
    ]

    evaluator = PolicyFitnessEvaluator()

    balanced_policy = Policy(
        trust_weight=0.35,
        capability_weight=0.30,
        latency_weight=0.20,
        resource_weight=0.15,
    )

    reliability_policy = Policy(
        trust_weight=0.70,
        capability_weight=0.10,
        latency_weight=0.10,
        resource_weight=0.10,
    )

    balanced_fitness = evaluator.evaluate(
        balanced_policy,
        experiences,
    )

    reliability_fitness = evaluator.evaluate(
        reliability_policy,
        experiences,
    )

    assert balanced_fitness != reliability_fitness

def test_create_fitness_function_evaluates_policy():
    experiences = [
        make_experience(
            "EXP-001",
            True,
            20.0,
            0.20,
        ),
        make_experience(
            "EXP-002",
            False,
            120.0,
            0.80,
        ),
    ]

    evaluator = PolicyFitnessEvaluator()

    fitness_function = evaluator.create_fitness_function(
        experiences
    )

    policy = Policy()

    fitness = fitness_function(policy)

    assert 0.0 <= fitness <= 1.0

def test_policy_fitness_function_works_with_evolution_engine():
    experiences = [
        make_experience(
            "EXP-001",
            True,
            20.0,
            0.20,
        ),
        make_experience(
            "EXP-002",
            False,
            120.0,
            0.80,
        ),
    ]

    evaluator = PolicyFitnessEvaluator()

    fitness_function = evaluator.create_fitness_function(
        experiences
    )

    engine = EvolutionEngine(
        fitness_function=fitness_function,
        population_size=4,
        seed=42,
    )

    population = [
        Policy(
            trust_weight=0.35,
            capability_weight=0.30,
            latency_weight=0.20,
            resource_weight=0.15,
        ),
        Policy(
            trust_weight=0.50,
            capability_weight=0.20,
            latency_weight=0.20,
            resource_weight=0.10,
        ),
        Policy(
            trust_weight=0.20,
            capability_weight=0.50,
            latency_weight=0.20,
            resource_weight=0.10,
        ),
        Policy(
            trust_weight=0.25,
            capability_weight=0.25,
            latency_weight=0.30,
            resource_weight=0.20,
        ),
    ]

    results = engine.evolve(
        population,
        generations=2,
    )

    assert len(results) == 2

    for result in results:
        assert 0.0 <= result.best_fitness <= 1.0
        assert 0.0 <= result.average_fitness <= 1.0

def test_evolution_tracks_best_policy_fitness():
    experiences = [
        make_experience(
            "EXP-001",
            True,
            20.0,
            0.20,
        ),
        make_experience(
            "EXP-002",
            False,
            120.0,
            0.80,
        ),
    ]

    evaluator = PolicyFitnessEvaluator()

    fitness_function = evaluator.create_fitness_function(
        experiences
    )

    engine = EvolutionEngine(
        fitness_function=fitness_function,
        population_size=4,
        seed=42,
    )

    population = [
        Policy(
            trust_weight=0.10,
            capability_weight=0.10,
            latency_weight=0.40,
            resource_weight=0.40,
        ),
        Policy(
            trust_weight=0.40,
            capability_weight=0.30,
            latency_weight=0.20,
            resource_weight=0.10,
        ),
        Policy(
            trust_weight=0.70,
            capability_weight=0.10,
            latency_weight=0.10,
            resource_weight=0.10,
        ),
        Policy(
            trust_weight=0.25,
            capability_weight=0.25,
            latency_weight=0.25,
            resource_weight=0.25,
        ),
    ]

    results = engine.evolve(
        population,
        generations=3,
    )

    best_fitness_values = [
        result.best_fitness
        for result in results
    ]

    assert best_fitness_values[0] <= best_fitness_values[-1]