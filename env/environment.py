from __future__ import annotations

from typing import Dict, List, Sequence

from .config import TaskConfig, VehicleSpec, get_task_config
from .models import Action, EnvironmentState, Observation, StepInfo, StepResult, TaskProgress, TaskSpec, Vehicle

SLOW_RATE = 0.08
FAST_RATE = 0.20
ACTION_TO_POWER = {0: 0.0, 1: 1.0, 2: 2.0}
ACTION_TO_CHARGE = {0: 0.0, 1: SLOW_RATE, 2: FAST_RATE}
MIN_OPEN_SCORE = 0.0001
MAX_OPEN_SCORE = 0.9999


class SmartChargeEnv:
    def __init__(self, mode: str = "easy", seed: int = 42):
        self.mode = mode
        self.seed = seed
        self.config: TaskConfig = get_task_config(mode)
        self.max_slots = self.config.max_slots
        self.max_steps = self.config.max_steps
        self.vehicles: List[Vehicle] = []
        self.step_count = 0
        self.price = self.config.price_profile[0]
        self.renewable = self.config.renewable_profile[0]
        self.peak_load = 0.0
        self.last_power_used = 0.0
        self.no_progress_streak = 0
        self.total_served = 0
        self.total_missed = 0
        self.total_energy_delivered = 0.0
        self.clean_energy_delivered = 0.0
        self.dirty_energy_delivered = 0.0
        self.dirty_charge_steps = 0
        self.cumulative_reward = 0.0
        self.last_action_error: str | None = None
        self.done = False
        self.closed = False

    def reset(self) -> Observation:
        self.step_count = 0
        self.vehicles = [self._vehicle_from_spec(spec) for spec in self.config.initial_vehicles]
        self.peak_load = 0.0
        self.last_power_used = 0.0
        self.no_progress_streak = 0
        self.total_served = 0
        self.total_missed = 0
        self.total_energy_delivered = 0.0
        self.clean_energy_delivered = 0.0
        self.dirty_energy_delivered = 0.0
        self.dirty_charge_steps = 0
        self.cumulative_reward = 0.0
        self.last_action_error = None
        self.done = False
        self.closed = False
        self.price = self._price_at(0)
        self.renewable = self._renewable_at(0)
        return self._get_observation()

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task=self._task_spec(),
            observation=self._get_observation(),
            progress=self._progress(),
            done=self.done,
            step_count=self.step_count,
        )

    def close(self) -> None:
        self.closed = True

    def step(self, action: Action | Dict[str, Sequence[int]], strict: bool = False):
        if self.done:
            raise RuntimeError("Episode is already complete. Call reset() before step().")

        normalized_action = self._normalize_action(action, strict=strict)
        current_vehicle_count = len(self.vehicles)
        assignments = list(normalized_action.assignments[:current_vehicle_count])
        if len(assignments) < current_vehicle_count:
            assignments.extend([0] * (current_vehicle_count - len(assignments)))
        pre_step_demand = sum(max(0.0, 1.0 - vehicle.soc) for vehicle in self.vehicles)
        current_price = self.price
        current_renewable = self.renewable

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
            effective_charge = base_charge * (1.0 + current_renewable * 0.2)
            updated_soc = min(1.0, vehicle.soc + effective_charge)
            delivered = max(0.0, updated_soc - vehicle.soc)
            charged_energy += delivered
            power_used += ACTION_TO_POWER[assignment]
            self.vehicles[index] = vehicle.model_copy(update={"soc": updated_soc})

        clean_energy = charged_energy * current_renewable
        dirty_energy = max(0.0, charged_energy - clean_energy)

        next_vehicles: List[Vehicle] = []
        for vehicle in self.vehicles:
            updated_deadline = max(0, vehicle.deadline - 1)
            updated_vehicle = vehicle.model_copy(update={"deadline": updated_deadline})
            if updated_vehicle.soc >= 1.0:
                served_on_time += 1
                continue
            if updated_deadline <= 0:
                missed += 1
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
        self._spawn_arrivals(self.step_count)

        total_vehicles = max(1, current_vehicle_count)
        reward_breakdown = self._calculate_reward(
            served_on_time=served_on_time,
            missed=missed,
            charged_energy=charged_energy,
            clean_energy=clean_energy,
            dirty_energy=dirty_energy,
            power_used=power_used,
            total_vehicles=total_vehicles,
            current_vehicle_count=current_vehicle_count,
            pre_step_demand=pre_step_demand,
            progress_delta=progress_delta,
            no_progress_streak=self.no_progress_streak,
            price=current_price,
            renewable=current_renewable,
        )
        reward = reward_breakdown["total"]
        self.cumulative_reward += reward
        self.total_served += served_on_time
        self.total_missed += missed
        self.total_energy_delivered += charged_energy
        self.clean_energy_delivered += clean_energy
        self.dirty_energy_delivered += dirty_energy
        if power_used > 0 and current_price >= 0.75 and current_renewable <= 0.25:
            self.dirty_charge_steps += 1
        self.last_power_used = power_used
        self.peak_load = max(self.peak_load, power_used)

        self.price = self._price_at(self.step_count)
        self.renewable = self._renewable_at(self.step_count)
        self.done = self.step_count >= self.max_steps or self._all_work_completed()

        progress = self._progress()
        info = StepInfo(
            served_on_time=served_on_time,
            missed=missed,
            power_used=power_used,
            peak_load=self.peak_load,
            active_vehicles=len(self.vehicles),
            score=progress.score,
            success=progress.success if self.done else False,
            last_action_error=self.last_action_error,
            reward_breakdown=reward_breakdown,
            truncated_actions=truncated_actions,
        )
        observation = self._get_observation()
        return observation, reward, self.done, info.model_dump()

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
            slots_available=max(0, self.max_slots),
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
        self.last_action_error = None
        return Action(assignments=cleaned_assignments)

    def _spawn_arrivals(self, step: int) -> None:
        for spec in self.config.arrivals_by_step.get(step, []):
            self.vehicles.append(self._vehicle_from_spec(spec))

    def _price_at(self, step: int) -> float:
        index = min(step, len(self.config.price_profile) - 1)
        return round(self.config.price_profile[index], 4)

    def _renewable_at(self, step: int) -> float:
        index = min(step, len(self.config.renewable_profile) - 1)
        return round(self.config.renewable_profile[index], 4)

    def _calculate_reward(
        self,
        *,
        served_on_time: int,
        missed: int,
        charged_energy: float,
        clean_energy: float,
        dirty_energy: float,
        power_used: float,
        total_vehicles: int,
        current_vehicle_count: int,
        pre_step_demand: float,
        progress_delta: float,
        no_progress_streak: int,
        price: float,
        renewable: float,
    ) -> Dict[str, float]:
        weighted_served = float(served_on_time)
        deadline_score = min(weighted_served / total_vehicles, 1.0)
        missed_penalty = min(missed / total_vehicles, 1.0)

        max_power = max(1.0, float(self.max_slots * ACTION_TO_POWER[2]))
        power_ratio = min(power_used / max_power, 1.0)
        spike_ratio = min(max(power_used - self.last_power_used, 0.0) / max_power, 1.0)
        energy_efficiency = max(0.0, 1.0 - power_ratio - 0.15 * spike_ratio)

        renewable_usage = min(1.0, clean_energy / max(charged_energy, 1e-6)) if charged_energy > 0 else 0.0

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
            expensive_dirty_grid = price > 0.75 and renewable < 0.25
            no_urgent_demand = urgent_vehicles == 0
            if expensive_dirty_grid and no_urgent_demand:
                harmful_charge_penalty = min((power_ratio + dirty_energy) / 2.0, 1.0)

        objective_progress = self._objective_progress(
            served_increment=served_on_time,
            missed_increment=missed,
        )

        total = (
            0.24 * deadline_score
            + 0.20 * energy_efficiency
            + 0.14 * renewable_usage
            + 0.10 * action_efficiency
            + 0.16 * progress_score
            + 0.16 * objective_progress
            - 0.10 * loop_penalty
            - 0.10 * harmful_charge_penalty
            - 0.10 * missed_penalty
        )
        total = min(max(total, 0.0), 1.0)

        return {
            "deadline_score": round(deadline_score, 4),
            "missed_penalty": round(missed_penalty, 4),
            "energy_efficiency": round(energy_efficiency, 4),
            "renewable_usage": round(renewable_usage, 4),
            "action_efficiency": round(action_efficiency, 4),
            "progress_score": round(progress_score, 4),
            "objective_progress": round(objective_progress, 4),
            "loop_penalty": round(loop_penalty, 4),
            "harmful_charge_penalty": round(harmful_charge_penalty, 4),
            "total": round(total, 4),
        }

    def _vehicle_from_spec(self, spec: VehicleSpec) -> Vehicle:
        return Vehicle(id=spec.id, soc=spec.soc, deadline=spec.deadline, priority=spec.priority)

    def _all_work_completed(self) -> bool:
        future_arrivals_exist = any(step > self.step_count for step in self.config.arrivals_by_step)
        return not self.vehicles and not future_arrivals_exist

    def _objective_progress(self, *, served_increment: int, missed_increment: int) -> float:
        served_progress = min(
            (self.total_served + served_increment) / max(1, self.config.min_served),
            1.0,
        )
        missed_budget = max(1, self.config.max_missed + 1)
        missed_progress = 1.0 - min((self.total_missed + missed_increment) / missed_budget, 1.0)
        return max(0.0, 0.7 * served_progress + 0.3 * missed_progress)

    def _score(self) -> float:
        if self.total_served == 0 and self.total_missed == 0 and self.total_energy_delivered == 0:
            return MIN_OPEN_SCORE
        served_ratio = min(self.total_served / max(1, self.config.min_served), 1.0)
        missed_ratio = 1.0 - min(self.total_missed / max(1, self.config.max_missed + 1), 1.0)
        clean_ratio = self.clean_energy_delivered / max(self.total_energy_delivered, 1e-6) if self.total_energy_delivered > 0 else 0.0
        peak_ratio = 1.0
        if self.config.max_peak_load is not None:
            peak_ratio = 1.0 - min(max(self.peak_load - self.config.max_peak_load, 0.0) / max(self.config.max_peak_load, 1.0), 1.0)
        dirty_ratio = 1.0
        if self.config.max_dirty_charge_steps is not None:
            dirty_ratio = 1.0 - min(
                max(self.dirty_charge_steps - self.config.max_dirty_charge_steps, 0) / max(self.config.max_dirty_charge_steps + 1, 1),
                1.0,
            )
        clean_target_ratio = 1.0
        if self.config.min_clean_energy_ratio is not None:
            clean_target_ratio = min(clean_ratio / max(self.config.min_clean_energy_ratio, 1e-6), 1.0)
        raw_score = (
            0.34 * served_ratio
            + 0.22 * missed_ratio
            + 0.16 * peak_ratio
            + 0.14 * dirty_ratio
            + 0.14 * clean_target_ratio
        )
        clamped_open = min(max(raw_score, MIN_OPEN_SCORE), MAX_OPEN_SCORE)
        return round(clamped_open, 4)

    def _progress(self) -> TaskProgress:
        clean_ratio = (
            self.clean_energy_delivered / self.total_energy_delivered
            if self.total_energy_delivered > 0
            else 0.0
        )
        score = self._score()
        success = self.is_success()
        return TaskProgress(
            served=self.total_served,
            missed=self.total_missed,
            cumulative_reward=round(self.cumulative_reward, 4),
            peak_load=round(self.peak_load, 4),
            dirty_charge_steps=self.dirty_charge_steps,
            clean_energy_ratio=round(clean_ratio, 4),
            score=score,
            success=success,
        )

    def _task_spec(self) -> TaskSpec:
        return TaskSpec(
            id=self.config.mode,
            title=self.config.title,
            objective=self.config.objective,
            min_served=self.config.min_served,
            max_missed=self.config.max_missed,
            max_peak_load=self.config.max_peak_load,
            max_dirty_charge_steps=self.config.max_dirty_charge_steps,
            min_clean_energy_ratio=self.config.min_clean_energy_ratio,
        )

    def is_success(self) -> bool:
        clean_ratio = (
            self.clean_energy_delivered / self.total_energy_delivered
            if self.total_energy_delivered > 0
            else 0.0
        )
        if self.total_served < self.config.min_served:
            return False
        if self.total_missed > self.config.max_missed:
            return False
        if self.config.max_peak_load is not None and self.peak_load > self.config.max_peak_load:
            return False
        if (
            self.config.max_dirty_charge_steps is not None
            and self.dirty_charge_steps > self.config.max_dirty_charge_steps
        ):
            return False
        if (
            self.config.min_clean_energy_ratio is not None
            and clean_ratio < self.config.min_clean_energy_ratio
        ):
            return False
        return True
