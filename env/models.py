from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


Priority = Literal["high", "normal"]


class Vehicle(BaseModel):
    id: str
    soc: float = Field(ge=0.0, le=1.0)
    deadline: int = Field(ge=0)
    priority: Priority


class Observation(BaseModel):
    vehicles: List[Vehicle]
    price: float = Field(ge=0.0)
    renewable: float = Field(ge=0.0, le=1.0)
    time_step: int = Field(ge=0)
    slots_available: int = Field(ge=0)


class Action(BaseModel):
    assignments: List[int] = Field(default_factory=list)


class StepInfo(BaseModel):
    served_on_time: int = 0
    missed: int = 0
    power_used: float = 0.0
    peak_load: float = 0.0
    active_vehicles: int = 0
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    truncated_actions: int = 0


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: StepInfo
