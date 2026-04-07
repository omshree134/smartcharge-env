---
title: SmartCharge Env
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: EV charging environment with FastAPI and OpenEnv.
---

# SmartCharge-Env

SmartCharge-Env is a deterministic EV charging optimization environment for evaluating agents that must balance grid stability, charging cost, renewable availability, and on-time vehicle completion. It is designed to feel like a realistic charging-station control problem.

## Why This Matters

Uncoordinated EV charging creates peak-load spikes that stress transformers, raise operating cost, and reduce user trust when urgent vehicles miss deadlines. SmartCharge-Env models the scheduling tradeoffs faced by:

- smart cities
- charging-network operators
- energy providers

The objective is to schedule limited charging slots so an agent can:

- minimize peak load
- minimize effective charging cost
- maximize vehicles served on time

## Environment Design

### Observation

| Field | Meaning |
| --- | --- |
| `vehicles` | Active EVs with `id`, `soc`, `deadline`, and `priority` |
| `price` | Current grid price signal |
| `renewable` | Current renewable availability in `[0, 1]` |
| `time_step` | Current simulation step |
| `slots_available` | Charging capacity available this step |

### Action

`assignments` is aligned with the current vehicle list.

| Value | Meaning |
| --- | --- |
| `0` | No charge |
| `1` | Slow charge |
| `2` | Fast charge |

### Reward

The environment uses a dense multi-objective reward:

```text
reward =
  0.30 * deadline_score +
  0.25 * energy_efficiency +
  0.15 * renewable_usage +
  0.10 * action_efficiency +
  0.20 * progress_score -
  0.10 * loop_penalty -
  0.10 * harmful_charge_penalty
```

Reward design highlights:

- `deadline_score` rewards vehicles completed before departure
- `energy_efficiency` penalizes high total load and sudden power spikes
- `renewable_usage` favors charging when renewable energy is plentiful
- `action_efficiency` penalizes wasting slots or charging unnecessarily
- `progress_score` gives partial credit for reducing remaining SOC deficit at each step
- `loop_penalty` discourages repeated no-progress behavior over consecutive steps
- `harmful_charge_penalty` discourages charging on expensive, low-renewable grid when no urgent demand exists

## Task Variants

| Task | Slots | Arrival Pattern | Grid Behavior | Main Challenge |
| --- | --- | --- | --- | --- |
| `easy` | 2 | Low traffic | Mostly stable price | Maximize completion rate |
| `medium` | 3 | Moderate traffic | Dynamic sinusoidal pricing | Balance deadlines and cost |
| `hard` | 4 | Bursty traffic | Volatile price and renewable swings | Multi-objective optimization under stress |

All tasks are deterministic under a fixed seed.

## Baseline Agent

The included baseline uses EDF scheduling with:

- priority-first ordering
- deadline urgency handling
- price-aware deferral for flexible vehicles

Expected behavior:

- urgent and high-priority vehicles receive fast charging
- flexible vehicles may wait during expensive periods
- charging load stays closer to slot limits instead of spiking blindly

## Project Layout

```text
env/
  agent.py
  config.py
  environment.py
  models.py
service/
  api.py
tests/
inference.py
openenv.yaml
Dockerfile
```

## Local Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline evaluation:

```bash
python inference.py
```

`inference.py` runs `easy`, `medium`, and `hard` by default and emits strict structured logs:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Switch tasks:

```bash
TASK_NAME=hard SEED=42 python inference.py
```

Hackathon-required environment variables:

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_token_here
```

Notes:
- Inference uses the OpenAI client (`openai.OpenAI`) for model calls when `HF_TOKEN` is set.
- For local offline checks without credentials, it safely falls back to the deterministic EDF baseline policy.
- `MAX_STEPS_PER_TASK` (default `40`) can be used to control runtime on hosted inference infrastructure.

Start the API:

```bash
uvicorn service.api:app --host 0.0.0.0 --port 7860
```

Example API flow:

```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"mode":"medium","seed":42}'
curl http://localhost:7860/state
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"assignments":[2,1,0]}'
```

Run tests:

```bash
pytest
```

## Docker

Build:

```bash
docker build -t smartcharge-env .
```

Run API mode:

```bash
docker run -p 7860:7860 smartcharge-env
```

Run inference mode:

```bash
docker run -e APP_MODE=inference -e TASK_NAME=hard smartcharge-env
```

## Insights You Can Highlight

- The agent can delay flexible charging during high-price periods.
- Urgent and high-priority vehicles are naturally surfaced by EDF ordering.
- Renewable-aware charging improves reward without requiring a complex controller.
- Peak-spike penalties make smoother schedules preferable to naive full-power charging.
