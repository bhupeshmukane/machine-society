from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    VIRTUAL = "virtual"
    ESP32 = "esp32"


class AgentStatus(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class Agent(BaseModel):
    agent_id: str
    agent_type: AgentType

    capabilities: list[str] = Field(default_factory=list)

    trust: float = Field(default=1.0, ge=0.0, le=1.0)
    load: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    resource: float = Field(default=1.0, ge=0.0, le=1.0)

    status: AgentStatus = AgentStatus.ONLINE

    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )