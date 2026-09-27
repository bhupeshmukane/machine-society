from app.experience.models import Experience
from app.fitness.engine import FitnessEngine


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


def test_empty_experiences_have_zero_fitness():
    result = FitnessEngine().evaluate([])

    assert result.fitness == 0.0
    assert result.success_rate == 0.0
    assert result.failure_rate == 0.0


def test_all_successful_experiences():
    experiences = [
        make_experience("EXP-001", True, 20.0, 0.20),
        make_experience("EXP-002", True, 30.0, 0.30),
    ]

    result = FitnessEngine().evaluate(experiences)

    assert result.success_rate == 1.0
    assert result.failure_rate == 0.0


def test_failure_rate_is_derived_from_experiences():
    experiences = [
        make_experience("EXP-001", True, 20.0, 0.20),
        make_experience("EXP-002", False, 40.0, 0.40),
    ]

    result = FitnessEngine().evaluate(experiences)

    assert result.success_rate == 0.5
    assert result.failure_rate == 0.5


def test_latency_score_uses_average_latency():
    experiences = [
        make_experience("EXP-001", True, 20.0, 0.20),
        make_experience("EXP-002", True, 40.0, 0.20),
    ]

    result = FitnessEngine().evaluate(
        experiences,
        latency_target_ms=100.0,
    )

    assert result.latency_score == 0.7


def test_resource_score_uses_average_resource():
    experiences = [
        make_experience("EXP-001", True, 20.0, 0.20),
        make_experience("EXP-002", True, 20.0, 0.40),
    ]

    result = FitnessEngine().evaluate(experiences)

    assert result.resource_score == 0.7


def test_fitness_is_bounded():
    experiences = [
        make_experience("EXP-001", True, 0.0, 0.0),
    ]

    result = FitnessEngine().evaluate(experiences)

    assert 0.0 <= result.fitness <= 1.0


def test_better_experiences_produce_higher_fitness():
    engine = FitnessEngine()

    good = [
        make_experience("GOOD-001", True, 20.0, 0.20),
        make_experience("GOOD-002", True, 30.0, 0.20),
    ]

    poor = [
        make_experience("POOR-001", False, 200.0, 0.90),
        make_experience("POOR-002", False, 200.0, 0.90),
    ]

    good_result = engine.evaluate(good)
    poor_result = engine.evaluate(poor)

    assert good_result.fitness > poor_result.fitness


def test_latency_score_is_clamped_to_zero():
    experiences = [
        make_experience("EXP-001", True, 500.0, 0.20),
    ]

    result = FitnessEngine().evaluate(
        experiences,
        latency_target_ms=100.0,
    )

    assert result.latency_score == 0.0


def test_fitness_result_exposes_all_components():
    experiences = [
        make_experience("EXP-001", True, 50.0, 0.25),
    ]

    result = FitnessEngine().evaluate(experiences)

    assert hasattr(result, "fitness")
    assert hasattr(result, "success_rate")
    assert hasattr(result, "latency_score")
    assert hasattr(result, "resource_score")
    assert hasattr(result, "failure_rate")