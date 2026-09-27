from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    task_id: str
    agent_id: str | None
    success: bool
    latency_ms: float
    resource_used: float
    failure_reason: str | None = None