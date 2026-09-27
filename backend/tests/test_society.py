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