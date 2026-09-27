import pytest

from app.evolution.coordinator import PolicyEvolutionCoordinator
from app.policy.engine import PolicyEngine
from app.policy.models import Policy
from app.agents.models import Agent, AgentStatus, AgentType
from app.experience.models import Experience
from app.experience.store import ExperienceStore
from app.society.runtime import SocietyRuntime
from app.tasks.allocator import TaskAllocator
from app.tasks.filter import CandidateFilter
from app.tasks.scoring import AgentScorer
from app.tasks.task import Task
from app.trust.engine import TrustEngine


def make_agent(
    agent_id: str,
    trust: float = 0.80,
) -> Agent:
    return Agent(
        agent_id=agent_id,
        agent_type=AgentType.VIRTUAL,
        capabilities=["temperature_sensing"],
        trust=trust,
        load=0.20,
        latency_ms=40.0,
        resource=0.80,
        status=AgentStatus.ONLINE,
    )


def make_runtime(
    agents: list[Agent],
) -> SocietyRuntime:
    return SocietyRuntime(
        agents=agents,
        task_allocator=TaskAllocator(
            candidate_filter=CandidateFilter(),
            scorer=AgentScorer(),
        ),
        experience_store=ExperienceStore(),
        trust_engine=TrustEngine(),
    )


def test_runtime_allocates_task():
    agent = make_agent("A01")
    runtime = make_runtime([agent])

    task = Task(
        task_id="T01",
        required_capabilities=["temperature_sensing"],
    )

    result = runtime.allocate_task(task)

    assert result is not None
    assert result.task_id == "T01"
    assert result.agent_id == "A01"


def test_successful_outcome_increases_trust():
    agent = make_agent("A01", trust=0.80)
    runtime = make_runtime([agent])

    task = Task(
        task_id="T02",
        required_capabilities=["temperature_sensing"],
    )

    experience = Experience(
        experience_id="E01",
        task_id="T02",
        agent_id="A01",
        success=True,
        latency_ms=40.0,
        resource_used=0.30,
    )

    result = runtime.record_outcome(
        task,
        experience,
    )

    assert result.success is True
    assert result.experience_id == "E01"
    assert result.trust_update_reason == "success"
    assert agent.trust == 0.82


def test_failed_outcome_decreases_trust():
    agent = make_agent("A01", trust=0.80)
    runtime = make_runtime([agent])

    task = Task(
        task_id="T03",
        required_capabilities=["temperature_sensing"],
    )

    experience = Experience(
        experience_id="E02",
        task_id="T03",
        agent_id="A01",
        success=False,
        latency_ms=150.0,
        resource_used=0.90,
        failure_reason="timeout",
    )

    result = runtime.record_outcome(
        task,
        experience,
    )

    assert result.success is False
    assert result.trust_update_reason == "failure"
    assert agent.trust == 0.75


def test_experience_is_stored():
    agent = make_agent("A01")
    runtime = make_runtime([agent])

    task = Task(task_id="T04")

    experience = Experience(
        experience_id="E03",
        task_id="T04",
        agent_id="A01",
        success=True,
        latency_ms=30.0,
        resource_used=0.20,
    )

    runtime.record_outcome(task, experience)

    # Runtime integration is expected to persist the outcome.
    assert runtime._experience_store.count() == 1
    assert runtime._experience_store.get("E03") == experience

def test_policy_scores_agent():
    agent = make_agent("A01", trust=0.80)
    runtime = make_runtime([agent])

    result = runtime.score_agent(
        agent,
        capability_score=1.0,
    )

    assert result.agent_id == "A01"
    assert 0.0 <= result.score <= 1.0


def test_policy_score_rewards_higher_trust():
    high_trust = make_agent("A01", trust=0.90)
    low_trust = make_agent("A02", trust=0.50)

    runtime = make_runtime(
        [high_trust, low_trust],
    )

    high_score = runtime.score_agent(
        high_trust,
        capability_score=1.0,
    )

    low_score = runtime.score_agent(
        low_trust,
        capability_score=1.0,
    )

    assert high_score.score > low_score.score

def test_policy_allocation_prefers_higher_trust():
    high_trust = make_agent("A01", trust=0.95)
    low_trust = make_agent("A02", trust=0.50)

    runtime = make_runtime(
        [low_trust, high_trust],
    )

    task = Task(
        task_id="T05",
        required_capabilities=["temperature_sensing"],
    )

    result = runtime.allocate_task(task)

    assert result is not None
    assert result.agent_id == "A01"

def test_policy_allocation_returns_none_without_candidate():
    agent = make_agent("A01")

    agent.status = AgentStatus.OFFLINE

    runtime = make_runtime([agent])

    task = Task(
        task_id="T06",
        required_capabilities=["temperature_sensing"],
    )

    result = runtime.allocate_task(task)

    assert result is None

def test_policy_score_rewards_lower_load():
    high_load = make_agent("A01", trust=0.80)
    low_load = make_agent("A02", trust=0.80)

    high_load.load = 0.90
    low_load.load = 0.10

    runtime = make_runtime(
        [high_load, low_load],
    )

    high_score = runtime.score_agent(
        high_load,
        capability_score=1.0,
    )

    low_score = runtime.score_agent(
        low_load,
        capability_score=1.0,
    )

    assert low_score.score > high_score.score

def test_allocate_task_uses_policy_score():
    high_load = make_agent("A01", trust=0.80)
    low_load = make_agent("A02", trust=0.80)

    high_load.load = 0.90
    low_load.load = 0.10

    runtime = make_runtime(
        [high_load, low_load],
    )

    task = Task(
        task_id="T05",
        required_capabilities=["temperature_sensing"],
    )

    result = runtime.allocate_task(task)

    assert result is not None
    assert result.agent_id == "A02"

def test_runtime_evolves_policy_from_recorded_experiences():
    policy_engine = PolicyEngine()

    coordinator = PolicyEvolutionCoordinator(
        policy_engine=policy_engine,
    )

    agent = make_agent("A01", trust=0.80)

    runtime = SocietyRuntime(
        agents=[agent],
        task_allocator=TaskAllocator(
            candidate_filter=CandidateFilter(),
            scorer=AgentScorer(),
        ),
        experience_store=ExperienceStore(),
        trust_engine=TrustEngine(),
        policy_engine=policy_engine,
        evolution_coordinator=coordinator,
    )

    task = Task(
        task_id="T07",
        required_capabilities=["temperature_sensing"],
    )

    runtime.record_outcome(
        task,
        Experience(
            experience_id="E07",
            task_id="T07",
            agent_id="A01",
            success=True,
            latency_ms=20.0,
            resource_used=0.20,
        ),
    )

    runtime.record_outcome(
        task,
        Experience(
            experience_id="E08",
            task_id="T07",
            agent_id="A01",
            success=False,
            latency_ms=120.0,
            resource_used=0.80,
            failure_reason="timeout",
        ),
    )

    initial_population = [
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

    original_policy = policy_engine.policy

    best_policy, results = runtime.evolve_policy(
        initial_population=initial_population,
        generations=3,
        seed=42,
    )

    assert isinstance(best_policy, Policy)
    assert len(results) == 3

    assert policy_engine.policy == best_policy
    assert policy_engine.policy != original_policy

    total = (
        best_policy.trust_weight
        + best_policy.capability_weight
        + best_policy.latency_weight
        + best_policy.resource_weight
    )

    assert total == pytest.approx(1.0)

    allocation = runtime.allocate_task(task)

    assert allocation is not None
    assert allocation.agent_id == "A01"

def test_runtime_rejects_policy_evolution_without_enough_experiences():
    policy_engine = PolicyEngine()

    coordinator = PolicyEvolutionCoordinator(
        policy_engine=policy_engine,
    )

    runtime = SocietyRuntime(
        agents=[],
        task_allocator=TaskAllocator(
            candidate_filter=CandidateFilter(),
            scorer=AgentScorer(),
        ),
        experience_store=ExperienceStore(),
        trust_engine=TrustEngine(),
        policy_engine=policy_engine,
        evolution_coordinator=coordinator,
    )

    population = [
        Policy(
            trust_weight=0.35,
            capability_weight=0.30,
            latency_weight=0.20,
            resource_weight=0.15,
        )
        for _ in range(4)
    ]

    with pytest.raises(
        ValueError,
        match="not enough experiences",
    ):
        runtime.evolve_policy(
            initial_population=population,
            min_experiences=2,
        )

def test_runtime_evolves_when_experience_threshold_is_reached():
    policy_engine = PolicyEngine()

    coordinator = PolicyEvolutionCoordinator(
        policy_engine=policy_engine,
    )

    experience_store = ExperienceStore()

    experience_store.record(
        Experience(
            experience_id="E09",
            task_id="T08",
            agent_id="A01",
            success=True,
            latency_ms=20.0,
            resource_used=0.20,
        )
    )

    experience_store.record(
        Experience(
            experience_id="E10",
            task_id="T08",
            agent_id="A01",
            success=False,
            latency_ms=120.0,
            resource_used=0.80,
            failure_reason="timeout",
        )
    )

    runtime = SocietyRuntime(
        agents=[],
        task_allocator=TaskAllocator(
            candidate_filter=CandidateFilter(),
            scorer=AgentScorer(),
        ),
        experience_store=experience_store,
        trust_engine=TrustEngine(),
        policy_engine=policy_engine,
        evolution_coordinator=coordinator,
    )

    population = [
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

    best_policy, results = runtime.evolve_policy(
        initial_population=population,
        generations=2,
        min_experiences=2,
        seed=42,
    )

    assert isinstance(best_policy, Policy)
    assert len(results) == 2
    assert policy_engine.policy == best_policy