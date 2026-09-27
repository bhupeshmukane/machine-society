from app.agents.models import Agent, AgentStatus, AgentType
from app.agents.registry import AgentRegistry


def make_agent(agent_id: str = "A01") -> Agent:
    return Agent(
        agent_id=agent_id,
        agent_type=AgentType.VIRTUAL,
        capabilities=["temperature_sensing"],
        trust=0.94,
        load=0.25,
        latency_ms=40,
        resource=0.82,
        status=AgentStatus.ONLINE,
    )


def test_register_and_get_agent():
    registry = AgentRegistry()
    agent = make_agent()

    registry.register(agent)

    result = registry.get("A01")

    assert result is not None
    assert result.agent_id == "A01"
    assert result.trust == 0.94


def test_list_agents():
    registry = AgentRegistry()

    registry.register(make_agent("A01"))
    registry.register(make_agent("A02"))

    agents = registry.list_agents()

    assert len(agents) == 2
    assert {agent.agent_id for agent in agents} == {"A01", "A02"}


def test_heartbeat_updates_timestamp_and_status():
    registry = AgentRegistry()
    agent = make_agent()

    registry.register(agent)

    agent.status = AgentStatus.DEGRADED
    old_heartbeat = agent.last_heartbeat

    updated = registry.update_heartbeat("A01")

    assert updated is not None
    assert updated.status == AgentStatus.ONLINE
    assert updated.last_heartbeat >= old_heartbeat


def test_unknown_agent_returns_none():
    registry = AgentRegistry()

    assert registry.get("UNKNOWN") is None
    assert registry.update_heartbeat("UNKNOWN") is None


def test_remove_agent():
    registry = AgentRegistry()
    registry.register(make_agent())

    assert registry.remove("A01") is True
    assert registry.get("A01") is None
    assert registry.remove("A01") is False