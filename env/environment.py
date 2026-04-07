from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

from .config import TaskConfig, get_task_config
from .models import Action, Observation, StepInfo, StepResult, Vehicle

SLOW_RATE = 0.08
FAST_RATE = 0.20
ACTION_TO_POWER = {0: 0.0, 1: 1.0, 2: 2.0}
ACTION_TO_CHARGE = {0: 0.0, 1: SLOW_RATE, 2: FAST_RATE}


class SmartChargeEnv:
    def __init__(self, mode: str = "easy", seed: int = 42):
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.config: TaskConfig = get_task_config(mode)
        self.max_slots = self.config.max_slots
        self.max_steps = self.config.max_steps
        self.vehicles: List[Vehicle] = []
        self.step_count = 0
        self.price = self.config.price_base
        self.renewable = self.config.renewable_base
        self.peak_load = 0.0
        self.last_power_used = 0.0
        self.vehicle_counter = 0
        self.no_progress_streak = 0

    def reset(self) -> Observation:
        self.rng = random.Random(self.seed)
        self.step_count = 0
        self.vehicles = []
        self.peak_load = 0.0
        self.last_power_used = 0.0
        self.vehicle_counter = 0
        self.no_progress_streak = 0
        self.price = self._compute_price(0)
        self.renewable = self._compute_renewable(0)
        return self._get_observation()

    def state(self) -> Observation:
        return self._get_observation()

    def step(self, action: Action | Dict[str, Sequence[int]], strict: bool = False):
        normalized_action = self._normalize_action(action, strict=strict)
        current_vehicle_count = len(self.vehicles)
        assignments = list(normalized_action.assignments[:current_vehicle_count])
        if len(assignments) < current_vehicle_count:
            assignments.extend([0] * (current_vehicle_count - len(assignments)))
        pre_step_demand = sum(max(0.0, 1.0 - vehicle.soc) for vehicle in self.vehicles)

        power_used = 0.0
        charged_energy = 0.0
        served_on_time = 0
        missed = 0
        truncated_actions = 0
        available_slots = self.max_slots

        for index, vehicle in enumerate(self.vehicles):
            assignment = assignments[index]
            if assignment and available_slots <= 0:
                assignment = 0
                truncated_actions += 1

            if assignment:
                available_slots -= 1

            base_charge = ACTION_TO_CHARGE[assignment]
            effective_charge = base_charge * (1.0 + self.renewable * 0.2)
            updated_soc = min(1.0, vehicle.soc + effective_charge)
            charged_energy += max(0.0, updated_soc - vehicle.soc)
            power_used += ACTION_TO_POWER[assignment]
            self.vehicles[index] = vehicle.model_copy(update={"soc": updated_soc})

        departing_ids = set()
        next_vehicles: List[Vehicle] = []
        for vehicle in self.vehicles:
            updated_deadline = max(0, vehicle.deadline - 1)
            updated_vehicle = vehicle.model_copy(update={"deadline": updated_deadline})
            if updated_vehicle.soc >= 1.0:
                served_on_time += 1
                departing_ids.add(updated_vehicle.id)
                continue
            if updated_deadline <= 0:
                missed += 1
                departing_ids.add(updated_vehicle.id)
                continue
            next_vehicles.append(updated_vehicle)

        post_step_demand = sum(max(0.0, 1.0 - vehicle.soc) for vehicle in next_vehicles)
        progress_delta = max(0.0, pre_step_demand - post_step_demand)
        if pre_step_demand > 0 and progress_delta < 0.01:
            self.no_progress_streak += 1
        else:
            self.no_progress_streak = 0

        self.vehicles = next_vehicles
        self.step_count += 1
        self._spawn_arrivals()
        self.price = self._compute_price(self.step_count)
        self.renewable = self._compute_renewable(self.step_count)

        total_vehicles = max(1, current_vehicle_count)
        reward_breakdown = self._calculate_reward(
            served_on_time=served_on_time,
            missed=missed,
            charged_energy=charged_energy,
            power_used=power_used,
            total_vehicles=total_vehicles,
            current_vehicle_count=current_vehicle_count,
            pre_step_demand=pre_step_demand,
            progress_delta=progress_delta,
            no_progress_streak=self.no_progress_streak,
        )
        reward = reward_breakdown["total"]
        self.last_power_used = power_used
        self.peak_load = max(self.peak_load, power_used)

        done = self.step_count >= self.max_steps
        info = StepInfo(
            served_on_time=served_on_time,
            missed=missed,
            power_used=power_used,
            peak_load=self.peak_load,
            active_vehicles=len(self.vehicles),
            reward_breakdown=reward_breakdown,
            truncated_actions=truncated_actions,
        )
        observation = self._get_observation()
        return observation, reward, done, info.model_dump()

    def step_result(self, action: Action | Dict[str, Sequence[int]], strict: bool = False) -> StepResult:
        observation, reward, done, info = self.step(action, strict=strict)
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=StepInfo.model_validate(info),
        )

    def _get_observation(self) -> Observation:
        return Observation(
            vehicles=[vehicle.model_copy() for vehicle in self.vehicles],
            price=round(self.price, 4),
            renewable=round(self.renewable, 4),
            time_step=self.step_count,
            slots_available=self.max_slots,
        )

    def _normalize_action(self, action: Action | Dict[str, Sequence[int]], strict: bool) -> Action:
        normalized = action if isinstance(action, Action) else Action.model_validate(action)
        cleaned_assignments: List[int] = []
        for value in normalized.assignments:
            if value not in (0, 1, 2):
                if strict:
                    raise ValueError(f"Invalid assignment '{value}'. Allowed values are 0, 1, or 2.")
                value = 0 if value < 0 else 2
            cleaned_assignments.append(value)
        return Action(assignments=cleaned_assignments)

    def _spawn_arrivals(self) -> None:
        arrivals = 0
        roll = self.rng.random()
        if self.config.bursty_arrivals and roll < 0.35:
            arrivals = 2
        elif roll < self.config.arrival_prob:
            arrivals = 1

        for _ in range(arrivals):
            self.vehicle_counter += 1
            priority = "high" if self.rng.random() < 0.35 else "normal"
            deadline_low, deadline_high = (4, 10) if self.mode == "hard" else (5, 14)
            if self.mode == "easy":
                deadline_low, deadline_high = (6, 16)
            vehicle = Vehicle(
                id=f"veh-{self.vehicle_counter}",
                soc=round(self.rng.uniform(0.10, 0.60), 3),
                deadline=self.rng.randint(deadline_low, deadline_high),
                priority=priority,
            )
            self.vehicles.append(vehicle)

    def _compute_price(self, step: int) -> float:
        cycle = math.sin((step / max(1, self.max_steps)) * math.tau)
        noise = self.rng.uniform(-self.config.price_noise, self.config.price_noise)
        value = self.config.price_base + self.config.price_amplitude * cycle + noise
        if self.mode == "hard" and self.rng.random() < 0.15:
            value += 0.12
        return round(min(max(value, 0.10), 1.00), 4)

    def _compute_renewable(self, step: int) -> float:
        cycle = math.cos((step / max(1, self.max_steps)) * math.pi)
        noise = self.rng.uniform(-self.config.renewable_noise, self.config.renewable_noise)
        value = self.config.renewable_base + self.config.renewable_amplitude * cycle + noise
        return round(min(max(value, 0.0), 1.0), 4)

    def _calculate_reward(
        self,
        *,
        served_on_time: int,
        missed: int,
        charged_energy: float,
        power_used: float,
        total_vehicles: int,
        current_vehicle_count: int,
        pre_step_demand: float,
        progress_delta: float,
        no_progress_streak: int,
    ) -> Dict[str, float]:
        weighted_served = float(served_on_time)
        deadline_score = min(weighted_served / total_vehicles, 1.0)

        max_power = max(1.0, float(self.max_slots * ACTION_TO_POWER[2]))
        power_ratio = min(power_used / max_power, 1.0)
        spike_ratio = min(max(power_used - self.last_power_used, 0.0) / max_power, 1.0)
        energy_efficiency = max(0.0, 1.0 - power_ratio - 0.15 * spike_ratio)

        total_possible_energy = max(charged_energy, self.max_slots * FAST_RATE)
        renewable_usage = min(1.0, self.renewable * (charged_energy / total_possible_energy)) if charged_energy > 0 else 0.0

        urgent_vehicles = sum(1 for vehicle in self.vehicles if vehicle.priority == "high" or vehicle.deadline <= 2)
        used_slots = min(self.max_slots, int(round(power_used)))
        idle_ratio = max(self.max_slots - used_slots, 0) / max(1, self.max_slots)
        idle_penalty = idle_ratio if urgent_vehicles > 0 else idle_ratio * 0.3

        unnecessary_charge_penalty = 0.0
        if current_vehicle_count > 0:
            nearly_full_count = sum(1 for vehicle in self.vehicles if vehicle.soc >= 0.95)
            unnecessary_charge_penalty = nearly_full_count / current_vehicle_count

        action_efficiency = max(0.0, 1.0 - 0.6 * idle_penalty - 0.4 * unnecessary_charge_penalty)

        # Continuous partial-progress signal so agents get credit before full completion.
        progress_score = 0.0
        if pre_step_demand > 0:
            progress_score = min(progress_delta / pre_step_demand, 1.0)

        # Penalize stagnation loops where the policy repeatedly makes no progress.
        loop_penalty = 0.0
        if pre_step_demand > 0:
            loop_penalty = min(no_progress_streak / 4.0, 1.0)

        # Penalize harmful charging when conditions are clearly unfavorable.
        harmful_charge_penalty = 0.0
        if power_used > 0:
            expensive_dirty_grid = self.price > 0.75 and self.renewable < 0.25
            no_urgent_demand = urgent_vehicles == 0
            if expensive_dirty_grid and no_urgent_demand:
                harmful_charge_penalty = min(power_ratio, 1.0)

        total = (
            0.30 * deadline_score
            + 0.25 * energy_efficiency
            + 0.15 * renewable_usage
            + 0.10 * action_efficiency
            + 0.20 * progress_score
            - 0.10 * loop_penalty
            - 0.10 * harmful_charge_penalty
        )
        total = min(max(total, 0.0), 1.0)

        return {
            "deadline_score": round(deadline_score, 4),
            "energy_efficiency": round(energy_efficiency, 4),
            "renewable_usage": round(renewable_usage, 4),
            "action_efficiency": round(action_efficiency, 4),
            "progress_score": round(progress_score, 4),
            "loop_penalty": round(loop_penalty, 4),
            "harmful_charge_penalty": round(harmful_charge_penalty, 4),
            "total": round(total, 4),
        }
