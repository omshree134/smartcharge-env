from __future__ import annotations

from .models import Action, Observation, Vehicle

FAST = 2
SLOW = 1
OFF = 0


def _slack(vehicle: Vehicle) -> int:
    if vehicle.soc >= 1.0:
        return 0
    required_steps_fast = max(1, int((1.0 - vehicle.soc) / 0.20))
    return vehicle.deadline - required_steps_fast


def edf_policy(observation: Observation) -> Action:
    vehicles = observation.vehicles
    assignments = [OFF] * len(vehicles)

    sorted_idx = sorted(
        range(len(vehicles)),
        key=lambda index: (
            vehicles[index].priority != "high",
            vehicles[index].deadline,
            vehicles[index].soc,
        ),
    )

    slots_remaining = observation.slots_available
    for index in sorted_idx:
        if slots_remaining <= 0:
            break

        vehicle = vehicles[index]
        slack = _slack(vehicle)
        expensive_and_safe_to_wait = observation.price > 0.7 and slack > 5 and vehicle.priority == "normal"
        if expensive_and_safe_to_wait:
            continue

        urgent = vehicle.deadline <= 3 or slack <= 1
        assignments[index] = FAST if vehicle.priority == "high" or urgent else SLOW
        slots_remaining -= 1

    return Action(assignments=assignments)
