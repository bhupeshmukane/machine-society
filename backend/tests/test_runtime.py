from datetime import datetime, timezone

from app.agents.models import Agent, AgentStatus, AgentType
from app.experience.store import ExperienceStore
from app.runtime.engine import TaskExecutionEngine
from app.tasks.allocator import TaskAllocator
from app.tasks.filter import CandidateFilter
from app.tasks.scoring import AgentScorer
from app.tasks.task import Task
from app.trust.engine import TrustEngine


def make_agent(
    agent_id: str = "A01",
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


def make_runtime() -> tuple[TaskExecutionEngine, ExperienceStore]:
    allocator = TaskAllocator(
        candidate_filter=CandidateFilter(),
        scorer=AgentScorer(),
    )

    experience_store = ExperienceStore()
    trust_engine = TrustEngine()

    runtime = TaskExecutionEngine(
        allocator=allocator,
        experience_store=experience_store,
        trust_engine=trust_engine,
    )

    return runtime, experience_store


def make_task() -> Task:
    return Task(
        task_id="T01",
        required_capabilities=["temperature_sensing"],
        min_trust=0.70,
        max_latency_ms=100.0,
    )


def test_successful_execution_records_experience():
    runtime, store = make_runtime()

    result = runtime.execute(
        task=make_task(),
        agents=[make_agent()],
        success=True,
        latency_ms=45.0,
        resource_used=0.40,
        timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc),
    )

    assert result.task_id == "T01"
    assert result.agent_id == "A01"
    assert result.success is True
    assert result.failure_reason is None

    assert store.count() == 1

    experience = store.list()[0]

    assert experience.task_id == "T01"
    assert experience.agent_id == "A01"
    assert experience.success is True
    assert experience.latency_ms == 45.0
    assert experience.resource_used == 0.40


def test_failed_execution_records_failure():
    runtime, store = make_runtime()

    result = runtime.execute(
        task=make_task(),
        agents=[make_agent()],
        success=False,
        latency_ms=120.0,
        resource_used=0.90,
        failure_reason="agent_timeout",
    )

    assert result.agent_id == "A01"
    assert result.success is False
    assert result.failure_reason == "agent_timeout"

    assert store.count() == 1
    assert store.list()[0].failure_reason == "agent_timeout"


def test_no_eligible_agent_creates_no_experience():
    runtime, store = make_runtime()

    offline_agent = make_agent()
    offline_agent.status = AgentStatus.OFFLINE

    result = runtime.execute(
        task=make_task(),
        agents=[offline_agent],
        success=True,
        latency_ms=40.0,
        resource_used=0.40,
    )

    assert result.task_id == "T01"
    assert result.agent_id is None
    assert result.success is False
    assert result.failure_reason == "no_eligible_agent"

    assert store.count() == 0


def test_multiple_executions_create_distinct_experiences():
    runtime, store = make_runtime()

    agent = make_agent()

    runtime.execute(
        task=Task(
            task_id="T01",
            required_capabilities=["temperature_sensing"],
        ),
        agents=[agent],
        success=True,
        latency_ms=40.0,
        resource_used=0.30,
    )

    runtime.execute(
        task=Task(
            task_id="T02",
            required_capabilities=["temperature_sensing"],
        ),
        agents=[agent],
        success=False,
        latency_ms=90.0,
        resource_used=0.70,
        failure_reason="sensor_error",
    )

    assert store.count() == 2

    experiences = store.list()

    assert experiences[0].experience_id != experiences[1].experience_id
    assert experiences[0].task_id == "T01"
    assert experiences[1].task_id == "T02"

def test_successful_execution_increases_agent_trust():
    runtime, store = make_runtime()

    agent = make_agent(trust=0.80)

    runtime.execute(
        task=make_task(),
        agents=[agent],
        success=True,
        latency_ms=40.0,
        resource_used=0.30,
    )

    assert store.count() == 1
    assert agent.trust == 0.82


def test_failed_execution_decreases_agent_trust():
    runtime, store = make_runtime()

    agent = make_agent(trust=0.80)

    runtime.execute(
        task=make_task(),
        agents=[agent],
        success=False,
        latency_ms=120.0,
        resource_used=0.80,
        failure_reason="timeout",
    )

    assert store.count() == 1
    assert agent.trust == 0.75


def test_trust_remains_bounded_after_success():
    runtime, _ = make_runtime()

    agent = make_agent(trust=0.99)

    runtime.execute(
        task=make_task(),
        agents=[agent],
        success=True,
        latency_ms=30.0,
        resource_used=0.20,
    )

    assert agent.trust == 1.0


def test_trust_remains_bounded_after_failure():
    runtime, _ = make_runtime()

    agent = make_agent(trust=0.02)

    runtime.execute(
        task=Task(
            task_id="T01",
            required_capabilities=["temperature_sensing"],
            min_trust=0.0,
            max_latency_ms=100.0,
        ),
        agents=[agent],
        success=False,
        latency_ms=120.0,
        resource_used=0.80,
        failure_reason="timeout",
    )

    assert agent.trust == 0.0