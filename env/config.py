from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    mode: str
    max_slots: int
    arrival_prob: float
    max_steps: int
    price_base: float
    price_amplitude: float
    price_noise: float
    renewable_base: float
    renewable_amplitude: float
    renewable_noise: float
    bursty_arrivals: bool = False


TASK_CONFIGS = {
    "easy": TaskConfig(
        mode="easy",
        max_slots=2,
        arrival_prob=0.30,
        max_steps=50,
        price_base=0.28,
        price_amplitude=0.03,
        price_noise=0.02,
        renewable_base=0.55,
        renewable_amplitude=0.10,
        renewable_noise=0.05,
    ),
    "medium": TaskConfig(
        mode="medium",
        max_slots=3,
        arrival_prob=0.50,
        max_steps=75,
        price_base=0.45,
        price_amplitude=0.20,
        price_noise=0.06,
        renewable_base=0.45,
        renewable_amplitude=0.20,
        renewable_noise=0.08,
    ),
    "hard": TaskConfig(
        mode="hard",
        max_slots=4,
        arrival_prob=0.80,
        max_steps=100,
        price_base=0.52,
        price_amplitude=0.28,
        price_noise=0.10,
        renewable_base=0.40,
        renewable_amplitude=0.30,
        renewable_noise=0.12,
        bursty_arrivals=True,
    ),
}


def get_task_config(mode: str) -> TaskConfig:
    try:
        return TASK_CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {sorted(TASK_CONFIGS)}.") from exc
