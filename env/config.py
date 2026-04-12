from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaskConfig:
    mode: str
    title: str
    objective: str
    max_slots: int
    max_steps: int
    price_profile: List[float]
    renewable_profile: List[float]
    initial_vehicles: List["VehicleSpec"] = field(default_factory=list)
    arrivals_by_step: Dict[int, List["VehicleSpec"]] = field(default_factory=dict)
    min_served: int = 0
    max_missed: int = 0
    max_peak_load: Optional[float] = None
    max_dirty_charge_steps: Optional[int] = None
    min_clean_energy_ratio: Optional[float] = None


@dataclass(frozen=True)
class VehicleSpec:
    id: str
    soc: float
    deadline: int
    priority: str


TASK_CONFIGS = {
    "easy": TaskConfig(
        mode="easy",
        title="Commuter Morning Rush",
        objective="Charge at least 3 commuters before departure while keeping misses to at most 1.",
        max_slots=2,
        max_steps=8,
        price_profile=[0.24, 0.25, 0.27, 0.26, 0.25, 0.23, 0.22, 0.21, 0.20],
        renewable_profile=[0.62, 0.60, 0.58, 0.56, 0.54, 0.52, 0.50, 0.48, 0.46],
        initial_vehicles=[
            VehicleSpec(id="easy-a", soc=0.40, deadline=4, priority="high"),
            VehicleSpec(id="easy-b", soc=0.55, deadline=5, priority="normal"),
        ],
        arrivals_by_step={
            1: [VehicleSpec(id="easy-c", soc=0.35, deadline=5, priority="normal")],
            3: [VehicleSpec(id="easy-d", soc=0.45, deadline=4, priority="high")],
        },
        min_served=3,
        max_missed=1,
        max_peak_load=4.0,
        min_clean_energy_ratio=0.45,
    ),
    "medium": TaskConfig(
        mode="medium",
        title="Dynamic Pricing Shift",
        objective="Balance completions against expensive grid periods and avoid wasteful peaks.",
        max_slots=3,
        max_steps=10,
        price_profile=[0.38, 0.44, 0.76, 0.81, 0.73, 0.49, 0.35, 0.31, 0.28, 0.33, 0.37],
        renewable_profile=[0.52, 0.48, 0.22, 0.18, 0.25, 0.40, 0.58, 0.64, 0.61, 0.55, 0.50],
        initial_vehicles=[
            VehicleSpec(id="med-a", soc=0.20, deadline=4, priority="high"),
            VehicleSpec(id="med-b", soc=0.42, deadline=6, priority="normal"),
            VehicleSpec(id="med-c", soc=0.30, deadline=5, priority="normal"),
        ],
        arrivals_by_step={
            2: [VehicleSpec(id="med-d", soc=0.26, deadline=4, priority="high")],
            4: [VehicleSpec(id="med-e", soc=0.34, deadline=5, priority="normal")],
            6: [VehicleSpec(id="med-f", soc=0.18, deadline=3, priority="high")],
        },
        min_served=4,
        max_missed=1,
        max_peak_load=5.0,
        max_dirty_charge_steps=2,
        min_clean_energy_ratio=0.35,
    ),
    "hard": TaskConfig(
        mode="hard",
        title="Storm Response Recovery",
        objective="Handle burst arrivals during dirty peak hours while preserving service quality.",
        max_slots=4,
        max_steps=12,
        price_profile=[0.46, 0.58, 0.84, 0.88, 0.79, 0.67, 0.51, 0.42, 0.39, 0.35, 0.32, 0.34, 0.38],
        renewable_profile=[0.48, 0.36, 0.16, 0.12, 0.18, 0.24, 0.41, 0.57, 0.66, 0.70, 0.68, 0.61, 0.55],
        initial_vehicles=[
            VehicleSpec(id="hard-a", soc=0.18, deadline=4, priority="high"),
            VehicleSpec(id="hard-b", soc=0.26, deadline=5, priority="high"),
            VehicleSpec(id="hard-c", soc=0.44, deadline=7, priority="normal"),
        ],
        arrivals_by_step={
            1: [
                VehicleSpec(id="hard-d", soc=0.22, deadline=4, priority="high"),
                VehicleSpec(id="hard-e", soc=0.35, deadline=6, priority="normal"),
            ],
            3: [VehicleSpec(id="hard-f", soc=0.14, deadline=3, priority="high")],
            5: [
                VehicleSpec(id="hard-g", soc=0.28, deadline=4, priority="normal"),
                VehicleSpec(id="hard-h", soc=0.24, deadline=5, priority="high"),
            ],
        },
        min_served=6,
        max_missed=1,
        max_peak_load=7.0,
        max_dirty_charge_steps=3,
        min_clean_energy_ratio=0.40,
    ),
}


def get_task_config(mode: str) -> TaskConfig:
    try:
        return TASK_CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {sorted(TASK_CONFIGS)}.") from exc
