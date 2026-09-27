from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

@dataclass(frozen=True)
class Experience:
    experience_id: str
    task_id: str
    agent_id: str
    success: bool
    latency_ms: float
    resource_used: float
    failure_reason: Optional[str] = None
    timestamp: datetime = field(
    default_factory=lambda: datetime.now(timezone.utc)
)