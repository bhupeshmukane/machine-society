import pytest
from app.evolution.coordinator import PolicyEvolutionCoordinator
from app.experience.models import Experience
from app.policy.engine import PolicyEngine
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


def make_population() -> list[Policy]:
    return [
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


def test_coordinator_returns_best_policy_and_results():
    policy_engine = PolicyEngine()

    coordinator = PolicyEvolutionCoordinator(
        policy_engine=policy_engine,
    )

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

    best_policy, results = coordinator.evolve(
        experiences=experiences,
        initial_population=make_population(),
        generations=3,
        seed=42,
    )

    assert isinstance(best_policy, Policy)
    assert len(results) == 3


def test_coordinator_applies_best_policy_to_policy_engine():
    policy_engine = PolicyEngine()

    original_policy = policy_engine.policy

    coordinator = PolicyEvolutionCoordinator(
        policy_engine=policy_engine,
    )

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

    best_policy, _ = coordinator.evolve(
        experiences=experiences,
        initial_population=make_population(),
        generations=3,
        seed=42,
    )

    assert policy_engine.policy == best_policy
    assert policy_engine.policy != original_policy


def test_coordinator_preserves_policy_weight_normalization():
    policy_engine = PolicyEngine()

    coordinator = PolicyEvolutionCoordinator(
        policy_engine=policy_engine,
    )

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

    best_policy, _ = coordinator.evolve(
        experiences=experiences,
        initial_population=make_population(),
        generations=3,
        seed=42,
    )

    total = (
        best_policy.trust_weight
        + best_policy.capability_weight
        + best_policy.latency_weight
        + best_policy.resource_weight
    )

    assert total == pytest.approx(1.0)