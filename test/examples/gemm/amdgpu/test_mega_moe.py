"""End-to-end numerical correctness test for dynamic MXFP4 MegaMoE."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
WORKER = Path(__file__).with_name("mega_moe_worker.py")


def has_gpus(count):
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= count
        and all(
            getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").startswith("gfx950")
            for index in range(count)
        )
    )


@pytest.mark.gpu
@pytest.mark.skipif(not has_gpus(8), reason="MegaMoE requires 8 gfx950 GPUs")
def test_mega_moe_matches_distributed_reference():
    env = os.environ.copy()
    python_path = str(ROOT / "python")
    if env.get("PYTHONPATH"):
        python_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            str(WORKER),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, f"torchrun failed:\n{result.stdout}\n{result.stderr}"
