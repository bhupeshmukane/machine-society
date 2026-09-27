import pytest

from app.agents.factory import VirtualAgentFactory
from app.agents.models import AgentType


def test_factory_creates_virtual_agent():
    factory = VirtualAgentFactory(seed=42)

    agent = factory.create("A01")

    assert agent.agent_id == "A01"
    assert agent.agent_type == AgentType.VIRTUAL
    assert len(agent.capabilities) >= 1
    assert 0.70 <= agent.trust <= 1.00
    assert 0.10 <= agent.load <= 0.90


def test_factory_creates_multiple_agents():
    factory = VirtualAgentFactory(seed=42)

    agents = factory.create_many(100)

    assert len(agents) == 100
    assert agents[0].agent_id == "A01"
    assert agents[-1].agent_id == "A100"

    assert len({agent.agent_id for agent in agents}) == 100


def test_seed_makes_generation_reproducible():
    factory_a = VirtualAgentFactory(seed=42)
    factory_b = VirtualAgentFactory(seed=42)

    agents_a = factory_a.create_many(10)
    agents_b = factory_b.create_many(10)

    for agent_a, agent_b in zip(agents_a, agents_b):
        assert agent_a.agent_id == agent_b.agent_id
        assert agent_a.agent_type == agent_b.agent_type
        assert agent_a.capabilities == agent_b.capabilities
        assert agent_a.trust == agent_b.trust
        assert agent_a.load == agent_b.load
        assert agent_a.latency_ms == agent_b.latency_ms
        assert agent_a.resource == agent_b.resource
        assert agent_a.status == agent_b.status


def test_invalid_count_is_rejected():
    factory = VirtualAgentFactory(seed=42)

    with pytest.raises(ValueError):
        factory.create_many(0)