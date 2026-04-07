from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from env.environment import SmartChargeEnv
from env.models import Action, Observation, StepResult


class ResetRequest(BaseModel):
    mode: str = Field(default="easy")
    seed: int = Field(default=42)


app = FastAPI(title="SmartCharge-Env", version="0.1.0")
_env = SmartChargeEnv()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> Observation:
    global _env
    _env = SmartChargeEnv(mode=request.mode, seed=request.seed)
    return _env.reset()


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    try:
        return _env.step_result(action, strict=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=Observation)
def state() -> Observation:
    return _env.state()
