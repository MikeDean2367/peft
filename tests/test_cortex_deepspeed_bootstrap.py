"""Exercise real CPU/Gloo ZeRO-3 construction before DeepSpeed engine creation."""

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

from peft import CortexConfig

from .test_cortex_distributed import ds_config, make_runner, tiny_model


class EngineCreationReached(Exception):
    pass


def main():
    import deepspeed
    from deepspeed.runtime.config import DeepSpeedConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world >= 2
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"rank": rank, "world_size": world, "device": "cpu", "cases": [], "status": "failed"}

    def reached_engine(**kwargs):
        assert dist.group.WORLD is original_group
        assert deepspeed.comm.get_world_group() is original_group
        assert deepspeed.comm.get_world_size() == world and deepspeed.comm.get_rank() == rank
        parameters = list(kwargs["model"].parameters())
        assert any(
            hasattr(parameter, "ds_id") and parameter.ds_tensor.numel() < parameter.ds_numel
            for parameter in parameters
        )
        assert kwargs["config"]["train_batch_size"] == 2 * world
        raise EngineCreationReached

    original_group = dist.group.WORLD
    try:
        assert not deepspeed.comm.is_initialized()
        if deepspeed.__version__ == "0.18.9":
            # Reproduce the reported failure with real torch.distributed: the
            # DeepSpeed wrapper has not attached to the existing process group.
            with pytest.raises(AssertionError, match="train_batch_size is not equal"):
                DeepSpeedConfig(ds_config(3, torch.float32, world))
            report["uninitialized_backend_failure_reproduced"] = True
        # CPU/Gloo needs no optional shared-memory accelerator extension. The
        # communication backend, world-size lookup and zero.Init remain real.
        with patch("deepspeed.comm.torch.build_shm_op", return_value=None):
            with patch("deepspeed.initialize", side_effect=reached_engine):
                for kind, mode, init, dtype in (
                    ("gemma3_text", "routed", "random", torch.float32),
                    ("gpt_oss", "routed", "sample", torch.bfloat16),
                ):
                    unset_hf_deepspeed_config()
                    torch.manual_seed(123)
                    clean = tiny_model(kind).to(dtype=dtype)
                    config = CortexConfig(
                        k=3,
                        moe_target=mode,
                        init_weights=init,
                        init_noise_std=0,
                        adapter_bias=kind == "gpt_oss",
                        task_type="CAUSAL_LM",
                    )
                    with pytest.raises(EngineCreationReached):
                        make_runner(clean, config, "zero3", dtype, torch.device("cpu"), rank)
                    report["cases"].append({"case": f"{kind}-{init}-{dtype}", "status": "passed"})
                    unset_hf_deepspeed_config()
                    gc.collect()
        report["status"] = "passed"
    finally:
        (args.output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2))
        unset_hf_deepspeed_config()
        dist.destroy_process_group()


def test_zero3_construction_uses_existing_torch_process_group(tmp_path):
    pytest.importorskip("deepspeed", minversion="0.18.9")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "tests.test_cortex_deepspeed_bootstrap",
            "--output",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "DS_ACCELERATOR": "cpu", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(report["status"] == "passed" and len(report["cases"]) == 2 for report in reports)


if __name__ == "__main__":
    main()
