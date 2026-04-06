from __future__ import annotations

import os
import sys

from env.agent import edf_policy
from env.environment import SmartChargeEnv

TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = "smartcharge"
MODEL_NAME = os.getenv("MODEL_NAME", "baseline")
SEED = int(os.getenv("SEED", "42"))


def log_start() -> None:
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")


def log_step(step: int, action: object, reward: float, done: bool, error: str = "null") -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error}"
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")


def run() -> int:
    env = SmartChargeEnv(mode=TASK_NAME, seed=SEED)
    observation = env.reset()
    log_start()

    rewards: list[float] = []
    max_steps = env.max_steps

    for step in range(1, max_steps + 1):
        try:
            action = edf_policy(observation)
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_step(step, action.model_dump(), reward, done)
            if done:
                break
        except Exception as exc:  # pragma: no cover - defensive logging
            log_step(step, "null", 0.0, True, error=repr(exc))
            score = sum(rewards) / len(rewards) if rewards else 0.0
            log_end(False, step, score, rewards)
            raise

    score = sum(rewards) / len(rewards) if rewards else 0.0
    log_end(True, step, score, rewards)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception:
        raise SystemExit(1) from None
