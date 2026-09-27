from app.agents.factory import VirtualAgentFactory
from app.agents.population import PopulationManager
from app.agents.registry import AgentRegistry


def create_manager(seed: int = 42) -> PopulationManager:
    registry = AgentRegistry()
    factory = VirtualAgentFactory(seed=seed)

    return PopulationManager(
        registry=registry,
        factory=factory,
    )


def test_create_small_population():
    manager = create_manager()

    agents = manager.create_virtual_population(5)

    assert len(agents) == 5
    assert len(manager.get_population()) == 5


def test_create_100_virtual_agents():
    manager = create_manager()

    agents = manager.create_virtual_population(100)

    assert len(agents) == 100
    assert agents[0].agent_id == "A01"
    assert agents[-1].agent_id == "A100"
    assert len({agent.agent_id for agent in agents}) == 100


def test_population_is_registered():
    manager = create_manager()

    manager.create_virtual_population(10)

    population = manager.get_population()

    assert len(population) == 10
    assert population[0].agent_id == "A01"