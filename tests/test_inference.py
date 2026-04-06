from __future__ import annotations

import os
import subprocess
import sys


def test_inference_emits_openenv_logs():
    env = os.environ.copy()
    env["TASK_NAME"] = "easy"
    env["MODEL_NAME"] = "baseline"
    env["SEED"] = "42"

    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout.strip().splitlines()
    assert output[0] == "[START] task=easy env=smartcharge model=baseline"
    assert output[-1].startswith("[END] success=true")
    assert any(line.startswith("[STEP] step=1") for line in output)
