import pytest

from app.trust.engine import TrustEngine


def test_success_increases_trust():
    engine = TrustEngine()

    result = engine.record_success(0.70)

    assert result.previous_trust == 0.70
    assert result.new_trust == 0.72
    assert result.delta == pytest.approx(0.02)
    assert result.reason == "success"


def test_failure_decreases_trust():
    engine = TrustEngine()

    result = engine.record_failure(0.70)

    assert result.previous_trust == 0.70
    assert result.new_trust == 0.65
    assert result.delta == pytest.approx(-0.05)
    assert result.reason == "failure"


def test_timeout_has_larger_penalty():
    engine = TrustEngine()

    result = engine.record_timeout(0.70)

    assert result.new_trust == 0.62
    assert result.delta == pytest.approx(-0.08)
    assert result.reason == "timeout"


def test_trust_cannot_exceed_one():
    engine = TrustEngine()

    result = engine.record_success(0.99)

    assert result.new_trust == 1.0


def test_trust_cannot_go_below_zero():
    engine = TrustEngine()

    result = engine.record_failure(0.02)

    assert result.new_trust == 0.0


def test_timeout_cannot_go_below_zero():
    engine = TrustEngine()

    result = engine.record_timeout(0.05)

    assert result.new_trust == 0.0


def test_custom_parameters():
    engine = TrustEngine(
        success_reward=0.10,
        failure_penalty=0.15,
        timeout_penalty=0.20,
    )

    success = engine.record_success(0.50)
    failure = engine.record_failure(0.50)
    timeout = engine.record_timeout(0.50)

    assert success.new_trust == 0.60
    assert failure.new_trust == 0.35
    assert timeout.new_trust == 0.30