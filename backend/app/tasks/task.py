from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    task_id: str
    required_capabilities: list[str] = field(default_factory=list)
    priority: int = 1
    max_latency_ms: float = 100.0
    min_trust: float = 0.0
    resource_requirement: float = 0.0
    context: str = "general"