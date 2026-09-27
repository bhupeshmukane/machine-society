from dataclasses import dataclass
from datetime import datetime, timezone

from .heartbeat import HeartbeatManager
from .models import AgentStatus


@dataclass(frozen=True)
class FailureEvent:
    agent_id: str
    detected_at: datetime
    previous_status: AgentStatus
    current_status: AgentStatus


class FailureDetector:
    def __init__(self, heartbeat_manager: HeartbeatManager) -> None:
        self._heartbeat_manager = heartbeat_manager

    def detect(
        self,
        agent_id: str,
        now: datetime | None = None,
    ) -> FailureEvent | None:
        agent = self._heartbeat_manager.get_agent(agent_id)

        if agent is None:
            return None

        previous_status = agent.status

        updated_agent = self._heartbeat_manager.evaluate_agent(
            agent_id,
            now,
        )

        if updated_agent is None:
            return None

        if (
            previous_status != AgentStatus.OFFLINE
            and updated_agent.status == AgentStatus.OFFLINE
        ):
            detected_at = now or datetime.now(timezone.utc)

            return FailureEvent(
                agent_id=agent_id,
                detected_at=detected_at,
                previous_status=previous_status,
                current_status=updated_agent.status,
            )

        return None