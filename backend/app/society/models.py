from dataclasses import dataclass

from app.tasks.allocator import AllocationResult


@dataclass(frozen=True)
class TaskExecutionResult:
    allocation: AllocationResult | None
    success: bool | None
    experience_id: str | None
    trust_update_reason: str | None


@dataclass(frozen=True)
class AgentPolicyScore:
    agent_id: str
    score: float