# app/main.py
# FastAPI server — OpenEnv-compliant.
# Endpoints: GET /health, POST /reset, POST /step, GET /state, POST /broadcast
#
# OpenEnv compliance:
#   /reset  → returns full initial state
#   /step   → accepts {agent, action, params}, returns {obs, reward, done, info}
#   /state  → returns current full state (read-only)
#   /health → liveness probe for HuggingFace Spaces Docker

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

# environment.py must exist at app/environment.py
from app.environment import DisasterEnvironment, AgentRole

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OmegaResponse",
    description=(
        "72-hour multi-agency disaster response simulation. "
        "OpenEnv-compliant RL environment for training LLMs via GRPO."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance.
# For concurrent multi-episode training, instantiate one per worker instead.
ENV: DisasterEnvironment = DisasterEnvironment()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field("omega_medium", description="Task difficulty")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    agent: str = Field(
        ...,
        description="Agent role: MedAgent | LogistAgent | CommandAgent",
        examples=["MedAgent"],
    )
    action: str = Field(
        ...,
        description="Action name valid for the given agent role",
        examples=["DispatchAmbulance"],
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action parameters, e.g. {'zone_id': 2}",
    )


class BroadcastRequest(BaseModel):
    agent: str = Field(..., description="Sending agent role")
    message: str = Field(..., description="Natural-language belief update")
    belief: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured belief state snapshot",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_agent(agent_str: str) -> AgentRole:
    mapping = {
        "medagent":     AgentRole.MED,
        "med":          AgentRole.MED,
        "logistagent":  AgentRole.LOGIST,
        "logist":       AgentRole.LOGIST,
        "commandagent": AgentRole.COMMAND,
        "command":      AgentRole.COMMAND,
    }
    role = mapping.get(agent_str.lower().strip())
    if role is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown agent '{agent_str}'. "
                "Valid values: MedAgent, LogistAgent, CommandAgent"
            ),
        )
    return role


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """
    Liveness probe.
    HuggingFace Spaces Docker healthcheck hits this endpoint.
    Returns 200 OK when the server is ready.
    """
    return {
        "status": "ok",
        "project": "OmegaResponse",
        "version": "1.0.0",
        "step": ENV.env_state.step,
        "done": ENV.env_state.done,
    }


@app.post("/reset", tags=["OpenEnv"])
def reset(body: ResetRequest = ResetRequest()):
    """
    Reset the environment to a new randomised disaster scenario.
    Optionally pass a seed for reproducibility.
    Returns the full initial state dict.

    OpenEnv equivalent: env.reset()
    """
    state = ENV.reset(seed=body.seed, task_id=body.task_id)
    return {"status": "reset", "state": state}


@app.post("/step", tags=["OpenEnv"])
def step(body: StepRequest):
    """
    Execute one agent action and advance the environment.

    Agent turn order: MedAgent → LogistAgent → CommandAgent (rotation).
    Submitting an out-of-turn action returns reward=-0.01 and an error info key.

    Returns:
        obs    — partial observation for the acting agent
        reward — float (dense + milestone if applicable + terminal if done)
        done   — bool (True at step 300)
        info   — dict with step, command_mode, conflict_score, etc.

    OpenEnv equivalent: env.step(action)
    """
    agent = _resolve_agent(body.agent)
    result = ENV.step(agent, body.action, body.params)
    return result


@app.get("/state", tags=["OpenEnv"])
def state():
    """
    Return the current full environment state (read-only, no side effects).
    CommandAgent uses this for full observability.
    Other agents should use /step which returns their partial obs.

    OpenEnv equivalent: env.get_full_state()
    """
    return ENV.get_full_state()


@app.get("/observe/{agent_role}", tags=["OpenEnv"])
def observe(agent_role: str):
    """
    Return the partial observation for a specific agent without advancing the env.
    Useful for agents to poll state before deciding their action.

    agent_role: MedAgent | LogistAgent | CommandAgent
    """
    agent = _resolve_agent(agent_role)
    return ENV.get_observation(agent)


@app.post("/broadcast", tags=["OpenEnv"])
def broadcast(body: BroadcastRequest):
    """
    Submit a belief broadcast from an agent.
    Triggers negotiation logic — other agents will see this in their obs broadcast_log.
    Called automatically every 5 steps, but can also be called manually.

    OpenEnv equivalent: env.broadcast(agent, message, belief)
    """
    agent = _resolve_agent(body.agent)
    result = ENV.broadcast(agent, body.message, body.belief)
    return result


@app.get("/zones", tags=["Debug"])
def zones():
    """Return current zone states. Debug/demo convenience endpoint."""
    return {
        "step": ENV.env_state.step,
        "zones": [z.to_dict() for z in ENV.env_state.zones],
    }


@app.get("/history", tags=["Debug"])
def history(last_n: int = 20):
    """Return the last N actions from action_history. Debug/demo endpoint."""
    return {
        "step": ENV.env_state.step,
        "action_history": ENV.env_state.action_history[-last_n:],
    }


# ---------------------------------------------------------------------------
# Entry point (local dev only — HF Spaces uses Dockerfile CMD)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)