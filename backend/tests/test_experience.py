from datetime import datetime, timezone

import pytest

from app.experience.models import Experience
from app.experience.store import ExperienceStore


def make_experience(
    experience_id: str = "EXP-001",
    task_id: str = "TASK-001",
    agent_id: str = "A01",
    success: bool = True,
    latency_ms: float = 42.5,
    resource_used: float = 0.35,
    failure_reason: str | None = None,
) -> Experience:
    return Experience(
        experience_id=experience_id,
        task_id=task_id,
        agent_id=agent_id,
        success=success,
        latency_ms=latency_ms,
        resource_used=resource_used,
        failure_reason=failure_reason,
    )


def test_record_experience():
    store = ExperienceStore()
    experience = make_experience()

    result = store.record(experience)

    assert result == experience
    assert store.count() == 1


def test_get_experience_by_id():
    store = ExperienceStore()
    experience = make_experience()

    store.record(experience)

    result = store.get("EXP-001")

    assert result == experience


def test_get_missing_experience_returns_none():
    store = ExperienceStore()

    assert store.get("DOES-NOT-EXIST") is None


def test_list_experiences():
    store = ExperienceStore()

    first = make_experience("EXP-001")
    second = make_experience("EXP-002", task_id="TASK-002")

    store.record(first)
    store.record(second)

    experiences = store.list()

    assert experiences == [first, second]


def test_count_experiences():
    store = ExperienceStore()

    assert store.count() == 0

    store.record(make_experience("EXP-001"))
    store.record(make_experience("EXP-002"))

    assert store.count() == 2


def test_duplicate_experience_id_is_rejected():
    store = ExperienceStore()

    store.record(make_experience("EXP-001"))

    with pytest.raises(ValueError, match="Experience already exists"):
        store.record(make_experience("EXP-001"))


def test_failed_experience_preserves_failure_reason():
    store = ExperienceStore()

    experience = make_experience(
        success=False,
        failure_reason="timeout",
    )

    store.record(experience)

    result = store.get("EXP-001")

    assert result is not None
    assert result.success is False
    assert result.failure_reason == "timeout"


def test_timestamp_is_generated():
    before = datetime.now(timezone.utc)

    experience = make_experience()

    after = datetime.now(timezone.utc)

    assert before <= experience.timestamp <= after