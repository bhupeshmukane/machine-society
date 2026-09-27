from app.agents.models import Agent, AgentStatus, AgentType
from app.tasks.allocator import TaskAllocator
from app.tasks.filter import CandidateFilter
from app.tasks.scoring import AgentScorer
from app.tasks.task import Task


def make_agent(
    agent_id: str,
    capabilities: list[str],
    trust: float,
    load: float,
    latency: float,
    resource: float,
    status: AgentStatus = AgentStatus.ONLINE,
):
    return Agent(
        agent_id=agent_id,
        agent_type=AgentType.VIRTUAL,
        capabilities=capabilities,
        trust=trust,
        load=load,
        latency_ms=latency,
        resource=resource,
        status=status,
    )

def test_filter_removes_ineligible_agents():
    task = Task(
        task_id="T01",
        required_capabilities=["temperature_sensing"],
        min_trust=0.75,
        max_latency_ms=100,
    )

    agents = [
        make_agent(
            "A01",
            ["temperature_sensing"],
            0.90,
            0.20,
            40,
            0.80,
        ),
        make_agent(
            "A02",
            ["temperature_sensing"],
            0.60,
            0.20,
            40,
            0.80,
        ),
        make_agent(
            "A03",
            ["motion_sensing"],
            0.90,
            0.20,
            40,
            0.80,
        ),
        make_agent(
            "A04",
            ["temperature_sensing"],
            0.90,
            0.20,
            150,
            0.80,
        ),
    ]

    candidates = CandidateFilter().filter(task, agents)

    assert [agent.agent_id for agent in candidates] == ["A01"]


def test_offline_agent_is_not_candidate():
    task = Task(
        task_id="T02",
        required_capabilities=["temperature_sensing"],
    )

    agent = make_agent(
        "A01",
        ["temperature_sensing"],
        0.95,
        0.10,
        30,
        0.90,
        AgentStatus.OFFLINE,
    )

    candidates = CandidateFilter().filter(task, [agent])

    assert candidates == []


def test_scorer_prefers_lower_load():
    scorer = AgentScorer()

    low_load = make_agent(
        "A01",
        ["temperature_sensing"],
        0.90,
        0.10,
        50,
        0.80,
    )

    high_load = make_agent(
        "A02",
        ["temperature_sensing"],
        0.90,
        0.90,
        50,
        0.80,
    )

    assert scorer.score(low_load) > scorer.score(high_load)


def test_allocator_selects_best_candidate():
    task = Task(
        task_id="T03",
        required_capabilities=["temperature_sensing"],
        min_trust=0.70,
        max_latency_ms=100,
    )

    agents = [
        make_agent(
            "A01",
            ["temperature_sensing"],
            0.80,
            0.70,
            80,
            0.60,
        ),
        make_agent(
            "A02",
            ["temperature_sensing"],
            0.95,
            0.15,
            35,
            0.90,
        ),
    ]

    allocator = TaskAllocator(
        candidate_filter=CandidateFilter(),
        scorer=AgentScorer(),
    )

    result = allocator.allocate(task, agents)

    assert result is not None
    assert result.task_id == "T03"
    assert result.agent_id == "A02"


def test_allocator_returns_none_when_no_candidate_exists():
    task = Task(
        task_id="T04",
        required_capabilities=["temperature_sensing"],
        min_trust=0.95,
    )

    agent = make_agent(
        "A01",
        ["temperature_sensing"],
        0.70,
        0.20,
        30,
        0.80,
    )

    allocator = TaskAllocator(
        candidate_filter=CandidateFilter(),
        scorer=AgentScorer(),
    )

    result = allocator.allocate(task, [agent])

    assert result is None