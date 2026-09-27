from fastapi import FastAPI, HTTPException

from app.agents.models import Agent
from app.agents.registry import AgentRegistry


app = FastAPI(
    title="Machine Society",
    version="0.1.0",
    description="Machine Society coordination platform",
)

registry = AgentRegistry()


@app.get("/")
def root():
    return {
        "system": "Machine Society",
        "status": "running",
        "version": "0.1.0",
    }


@app.post("/agents", response_model=Agent)
def register_agent(agent: Agent):
    return registry.register(agent)


@app.get("/agents", response_model=list[Agent])
def list_agents():
    return registry.list_agents()


@app.get("/agents/{agent_id}", response_model=Agent)
def get_agent(agent_id: str):
    agent = registry.get(agent_id)

    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found",
        )

    return agent


@app.post("/agents/{agent_id}/heartbeat", response_model=Agent)
def heartbeat(agent_id: str):
    agent = registry.update_heartbeat(agent_id)

    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found",
        )

    return agent