from datetime import datetime, timedelta, timezone

from app.agents.factory import VirtualAgentFactory
from app.agents.heartbeat import HeartbeatConfig, HeartbeatManager
from app.agents.models import AgentStatus
from app.agents.registry import AgentRegistry


def create_manager() -> tuple[AgentRegistry, HeartbeatManager]:
    registry = AgentRegistry()

    factory = VirtualAgentFactory(seed=42)

    agents = factory.create_many(3)

    for agent in agents:
        registry.register(agent)

    heartbeat_manager = HeartbeatManager(
        registry=registry,
        config=HeartbeatConfig(
            degraded_after_seconds=10,
            offline_after_seconds=20,
        ),
    )

    return registry, heartbeat_manager


def test_recent_heartbeat_is_online():
    registry, manager = create_manager()

    now = datetime.now(timezone.utc)

    manager.record_heartbeat("A01", now)

    agent = manager.evaluate_agent("A01", now)

    assert agent is not None
    assert agent.status == AgentStatus.ONLINE


def test_delayed_heartbeat_is_degraded():
    registry, manager = create_manager()

    heartbeat = datetime.now(timezone.utc)
    now = heartbeat + timedelta(seconds=15)

    manager.record_heartbeat("A01", heartbeat)

    agent = manager.evaluate_agent("A01", now)

    assert agent is not None
    assert agent.status == AgentStatus.DEGRADED


def test_missing_heartbeat_is_offline():
    registry, manager = create_manager()

    heartbeat = datetime.now(timezone.utc)
    now = heartbeat + timedelta(seconds=25)

    manager.record_heartbeat("A01", heartbeat)

    agent = manager.evaluate_agent("A01", now)

    assert agent is not None
    assert agent.status == AgentStatus.OFFLINE


def test_heartbeat_recovers_agent():
    registry, manager = create_manager()

    heartbeat = datetime.now(timezone.utc)
    failed_time = heartbeat + timedelta(seconds=25)

    manager.record_heartbeat("A01", heartbeat)
    manager.evaluate_agent("A01", failed_time)

    assert registry.get("A01").status == AgentStatus.OFFLINE

    manager.record_heartbeat("A01", failed_time)

    assert registry.get("A01").status == AgentStatus.ONLINE


def test_unknown_agent_heartbeat_is_ignored():
    registry, manager = create_manager()

    result = manager.record_heartbeat("UNKNOWN")

    assert result is None