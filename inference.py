from __future__ import annotations

import os
import json
import random
from typing import Any, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback for local/offline checks without dependency.
    OpenAI = None  # type: ignore[assignment]
from env.agent import edf_policy
from env.environment import SmartChargeEnv
from env.models import Action, Observation

# Required variables in evaluator:
# - API_BASE_URL
# - MODEL_NAME
# - HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ALLOW_BASELINE_FALLBACK = os.getenv("ALLOW_BASELINE_FALLBACK", "0") == "1"

TASK_NAME = os.getenv("TASK_NAME")
BENCHMARK = os.getenv("BENCHMARK", "smartcharge")
SEED = int(os.getenv("SEED", "42"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "180"))
MAX_STEPS_PER_TASK = int(os.getenv("MAX_STEPS_PER_TASK", "40"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))
MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99
random.seed(SEED)


SYSTEM_PROMPT = (
    "You are an EV charging scheduler.\n"
    "Return ONLY valid JSON object: {\"assignments\": [ints...]}\n"
    "Rules:\n"
    "- assignment values must be 0, 1, or 2\n"
    "- list length must match number of vehicles\n"
    "- at most slots_available non-zero assignments\n"
    "- prioritize urgent/high-priority vehicles"
)


def safe_serialize(action: Any) -> str:
    """Safely convert action to JSON string."""
    try:
        if hasattr(action, "model_dump"):
            return json.dumps(action.model_dump(), separators=(",", ":"))
        elif isinstance(action, dict):
            return json.dumps(action, separators=(",", ":"))
        else:
            return json.dumps(str(action))
    except Exception:
        return '"unserializable_action"'


def log_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = "null") -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def clamp_task_score(score: float) -> float:
    return min(max(score, MIN_TASK_SCORE), MAX_TASK_SCORE)


def build_client() -> Optional[OpenAI]:
    if OpenAI is None:
        if ALLOW_BASELINE_FALLBACK:
            return None
        raise RuntimeError("openai dependency is unavailable.")
    if not HF_TOKEN:
        if ALLOW_BASELINE_FALLBACK:
            return None
        raise RuntimeError("HF_TOKEN environment variable is required.")
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME must be configured.")
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL must be configured.")
    if MODEL_NAME == "baseline" and not ALLOW_BASELINE_FALLBACK:
        raise RuntimeError("MODEL_NAME must resolve to a real hosted model in submission mode.")
    if ALLOW_BASELINE_FALLBACK and not HF_TOKEN:
        return None
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def normalize_assignments(raw: Any, vehicle_count: int, slots_available: int) -> list[int]:
    if not isinstance(raw, list):
        return [0] * vehicle_count

    cleaned: list[int] = []
    for value in raw[:vehicle_count]:
        try:
            parsed = int(value)
        except Exception:
            parsed = 0
        if parsed < 0:
            parsed = 0
        if parsed > 2:
            parsed = 2
        cleaned.append(parsed)

    if len(cleaned) < vehicle_count:
        cleaned.extend([0] * (vehicle_count - len(cleaned)))

    non_zero_indices = [index for index, value in enumerate(cleaned) if value > 0]
    if len(non_zero_indices) > slots_available:
        for index in non_zero_indices[slots_available:]:
            cleaned[index] = 0

    return cleaned


def llm_policy(client: OpenAI, observation: Observation) -> tuple[Action, Optional[str]]:
    vehicle_payload = [
        {
            "id": vehicle.id,
            "soc": vehicle.soc,
            "deadline": vehicle.deadline,
            "priority": vehicle.priority,
        }
        for vehicle in observation.vehicles
    ]
    user_payload = {
        "vehicles": vehicle_payload,
        "price": observation.price,
        "renewable": observation.renewable,
        "time_step": observation.time_step,
        "slots_available": observation.slots_available,
    }

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
            ],
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        assignments = parsed.get("assignments", [])
        normalized = normalize_assignments(
            assignments, len(observation.vehicles), observation.slots_available
        )
        return Action(assignments=normalized), None
    except Exception as exc:
        fallback = edf_policy(observation)
        return fallback, f"llm_error:{repr(exc)}"


def choose_action(client: Optional[OpenAI], observation: Observation) -> tuple[Action, str]:
    if client is None:
        return edf_policy(observation), "null"

    action, llm_error = llm_policy(client, observation)
    return action, (llm_error or "null")


def compute_success(env: SmartChargeEnv, score: float) -> bool:
    try:
        if hasattr(env, "is_success"):
            return bool(env.is_success())
    except Exception:
        pass
    return score >= SUCCESS_SCORE_THRESHOLD


def run_episode(task_name: str, client: Optional[OpenAI]) -> tuple[float, bool]:
    log_start(task_name)
    rewards: list[float] = []
    score = 0.0
    success = False
    step = 0
    env: Optional[SmartChargeEnv] = None

    try:
        env = SmartChargeEnv(mode=task_name, seed=SEED)
        observation = env.reset()
    except Exception:
        safe_score = MIN_TASK_SCORE
        log_end(False, 0, safe_score, [])
        return safe_score, False

    max_steps = min(getattr(env, "max_steps", 100), MAX_STEPS_PER_TASK)

    try:
        for step in range(1, max_steps + 1):
            try:
                action, action_error = choose_action(client, observation)
                action_str = safe_serialize(action)

                observation, reward, done, _ = env.step(action)
                rewards.append(float(reward))

                log_step(step, action_str, float(reward), bool(done), action_error)
                if done:
                    break

            except Exception as step_error:
                log_step(step, "null", 0.0, True, f"step_error:{repr(step_error)}")
                break
    finally:
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = clamp_task_score(score)
        success = compute_success(env, score)
        try:
            env.close()
        finally:
            log_end(success, step, score, rewards)

    return score, success


def run() -> int:
    client = build_client()
    tasks = [TASK_NAME] if TASK_NAME else ["easy", "medium", "hard"]

    for task_name in tasks:
        run_episode(task_name, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
