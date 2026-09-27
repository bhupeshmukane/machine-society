from datetime import datetime, timedelta, timezone

from app.agents.factory import VirtualAgentFactory
from app.agents.failure import FailureDetector
from app.agents.heartbeat import HeartbeatConfig, HeartbeatManager
from app.agents.models import AgentStatus
from app.agents.registry import AgentRegistry


def create_detector() -> tuple[HeartbeatManager, FailureDetector]:
    registry = AgentRegistry()

    factory = VirtualAgentFactory(seed=42)

    for agent in factory.create_many(3):
        registry.register(agent)

    heartbeat_manager = HeartbeatManager(
        registry=registry,
        config=HeartbeatConfig(
            degraded_after_seconds=10,
            offline_after_seconds=20,
        ),
    )

    detector = FailureDetector(heartbeat_manager)

    return heartbeat_manager, detector


def test_detect_offline_agent():
    heartbeat_manager, detector = create_detector()

    heartbeat_time = datetime.now(timezone.utc)
    failure_time = heartbeat_time + timedelta(seconds=25)

    heartbeat_manager.record_heartbeat(
        "A01",
        heartbeat_time,
    )

    event = detector.detect(
        "A01",
        failure_time,
    )

    assert event is not None
    assert event.agent_id == "A01"
    assert event.previous_status == AgentStatus.ONLINE
    assert event.current_status == AgentStatus.OFFLINE


def test_no_failure_for_online_agent():
    heartbeat_manager, detector = create_detector()

    now = datetime.now(timezone.utc)

    heartbeat_manager.record_heartbeat(
        "A01",
        now,
    )

    event = detector.detect(
        "A01",
        now,
    )

    assert event is None


def test_degraded_agent_does_not_generate_failure_event():
    heartbeat_manager, detector = create_detector()

    heartbeat_time = datetime.now(timezone.utc)
    delayed_time = heartbeat_time + timedelta(seconds=15)

    heartbeat_manager.record_heartbeat(
        "A01",
        heartbeat_time,
    )

    event = detector.detect(
        "A01",
        delayed_time,
    )

    assert event is None

    agent = heartbeat_manager.get_agent("A01")

    assert agent is not None
    assert agent.status == AgentStatus.DEGRADED


def test_unknown_agent_returns_no_event():
    _, detector = create_detector()

    event = detector.detect("UNKNOWN")

    assert event is None


def test_repeated_detection_does_not_create_new_failure_event():
    heartbeat_manager, detector = create_detector()

    heartbeat_time = datetime.now(timezone.utc)
    first_failure_time = heartbeat_time + timedelta(seconds=25)
    second_failure_time = heartbeat_time + timedelta(seconds=30)

    heartbeat_manager.record_heartbeat(
        "A01",
        heartbeat_time,
    )

    first_event = detector.detect(
        "A01",
        first_failure_time,
    )

    second_event = detector.detect(
        "A01",
        second_failure_time,
    )

    assert first_event is not None
    assert second_event is None