import pytest

from app.policy.engine import PolicyEngine
from app.policy.models import Policy


def test_default_policy_weights():
    policy = Policy()

    assert policy.trust_weight == 0.35
    assert policy.capability_weight == 0.30
    assert policy.latency_weight == 0.20
    assert policy.resource_weight == 0.15


def test_custom_policy_weights():
    policy = Policy(
        trust_weight=0.50,
        capability_weight=0.20,
        latency_weight=0.20,
        resource_weight=0.10,
    )

    assert policy.trust_weight == 0.50
    assert policy.capability_weight == 0.20
    assert policy.latency_weight == 0.20
    assert policy.resource_weight == 0.10


def test_score_calculation():
    engine = PolicyEngine()

    score = engine.score(
        trust=1.0,
        capability=1.0,
        latency=0.0,
        resource=0.0,
    )

    assert score == 1.0


def test_score_is_bounded():
    engine = PolicyEngine()

    low_score = engine.score(
        trust=-10.0,
        capability=-10.0,
        latency=10.0,
        resource=10.0,
    )

    high_score = engine.score(
        trust=10.0,
        capability=10.0,
        latency=-10.0,
        resource=-10.0,
    )

    assert 0.0 <= low_score <= 1.0
    assert 0.0 <= high_score <= 1.0


def test_high_quality_agent_scores_higher():
    engine = PolicyEngine()

    high_score = engine.score(
        trust=0.90,
        capability=0.90,
        latency=0.20,
        resource=0.20,
    )

    low_score = engine.score(
        trust=0.40,
        capability=0.40,
        latency=0.80,
        resource=0.80,
    )

    assert high_score > low_score


def test_zero_inputs_do_not_produce_negative_score():
    engine = PolicyEngine()

    score = engine.score(
        trust=0.0,
        capability=0.0,
        latency=1.0,
        resource=1.0,
    )

    assert score == 0.0


def test_maximum_inputs_do_not_exceed_one():
    engine = PolicyEngine()

    score = engine.score(
        trust=1.0,
        capability=1.0,
        latency=0.0,
        resource=0.0,
    )

    assert score <= 1.0


def test_default_weights_sum_to_one():
    policy = Policy()

    total = (
        policy.trust_weight
        + policy.capability_weight
        + policy.latency_weight
        + policy.resource_weight
    )

    assert total == pytest.approx(1.0)