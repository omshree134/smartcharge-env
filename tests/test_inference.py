from __future__ import annotations

import os
import subprocess
import sys


def test_inference_emits_openenv_logs():
    env = os.environ.copy()
    env["TASK_NAME"] = "easy"
    env["SEED"] = "42"
    env["ALLOW_BASELINE_FALLBACK"] = "1"

    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout.strip().splitlines()
    assert output[0].startswith("[START] task=easy env=smartcharge model=")
    assert output[-1].startswith("[END] success=true")
    assert "score=" not in output[-1]
    assert any(line.startswith("[STEP] step=1") for line in output)
