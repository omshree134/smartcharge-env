#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; exit 1; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@"
  fi
}

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  fail "repo directory not found: ${2:-.}"
fi

cd "$REPO_DIR"

printf "\n${BOLD}========================================${NC}\n"
printf "${BOLD}  SmartCharge Pre-Validation${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo: $REPO_DIR"

if [[ -n "$PING_URL" ]]; then
  PING_URL="${PING_URL%/}"
  log "${BOLD}Step 1/4:${NC} pinging Space reset endpoint"
  CODE="$(curl -sS -o /tmp/openenv_ping.out -w "%{http_code}" \
    -X POST "$PING_URL/reset" \
    -H "Content-Type: application/json" \
    -d '{"mode":"easy","seed":42}' || true)"
  if [[ "$CODE" == "200" ]]; then
    pass "Space /reset returned HTTP 200"
  else
    fail "Space /reset failed (HTTP $CODE)"
  fi
else
  log "${BOLD}Step 1/4:${NC} skipping Space ping (no URL provided)"
fi

log "${BOLD}Step 2/4:${NC} checking required files"
[[ -f "openenv.yaml" ]] || fail "missing openenv.yaml"
[[ -f "inference.py" ]] || fail "missing inference.py"
[[ -f "Dockerfile" ]] || fail "missing Dockerfile"
[[ -f "README.md" ]] || fail "missing README.md"
pass "required files exist"

TASK_COUNT="$(grep -E '^[[:space:]]*-[[:space:]]+id:' openenv.yaml | wc -l | tr -d ' ')"
if [[ "${TASK_COUNT:-0}" -ge 3 ]]; then
  pass "openenv.yaml defines >= 3 tasks ($TASK_COUNT)"
else
  fail "openenv.yaml has fewer than 3 tasks ($TASK_COUNT)"
fi

log "${BOLD}Step 3/4:${NC} validating inference output format and reward bounds"
for task in easy medium hard; do
  OUT_FILE="$(mktemp)"
  API_BASE_URL="${API_BASE_URL:-https://router.huggingface.co/v1}" \
  MODEL_NAME="${MODEL_NAME:-baseline}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  TASK_NAME="$task" \
  MAX_STEPS_PER_TASK="${MAX_STEPS_PER_TASK:-20}" \
  run_with_timeout 300 python inference.py >"$OUT_FILE" 2>&1 || {
    cat "$OUT_FILE"
    fail "inference failed for task=$task"
  }

  python - "$OUT_FILE" "$task" <<'PY'
import re
import sys

path, task = sys.argv[1], sys.argv[2]
lines = [ln.strip() for ln in open(path, "r", encoding="utf-8") if ln.strip()]
if not lines:
    raise SystemExit(f"no output for task={task}")

start = re.compile(r"^\[START\] task=(\S+) env=(\S+) model=(\S+)$")
step = re.compile(
    r"^\[STEP\] step=(\d+) action=(.+) reward=([0-9]+\.[0-9]{2}) done=(true|false) error=(.+)$"
)
end = re.compile(
    r"^\[END\] success=(true|false) steps=(\d+) score=([0-9]+\.[0-9]{2}) rewards=(.*)$"
)

if not start.match(lines[0]):
    raise SystemExit(f"bad [START] line for task={task}: {lines[0]}")
if not end.match(lines[-1]):
    raise SystemExit(f"bad [END] line for task={task}: {lines[-1]}")
if not any(line.startswith("[STEP] ") for line in lines):
    raise SystemExit(f"missing [STEP] lines for task={task}")

for line in lines:
    if line.startswith("[STEP] "):
        m = step.match(line)
        if not m:
            raise SystemExit(f"bad [STEP] line for task={task}: {line}")
        reward = float(m.group(3))
        if reward < 0.0 or reward > 1.0:
            raise SystemExit(f"step reward out of bounds for task={task}: {reward}")
PY
  pass "inference format OK for task=$task"
done

log "${BOLD}Step 4/4:${NC} quick API import check"
python - <<'PY'
from service.api import app
assert app is not None
print("api import ok")
PY
pass "API app import check passed"

printf "\n${GREEN}${BOLD}ALL PRE-VALIDATION CHECKS PASSED${NC}\n"
