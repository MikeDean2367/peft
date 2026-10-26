# Cortex GPU validation

The standard GPU entry files collect 120 Cortex cases: 104 device, bias, public-routing,
and device-transfer checks in `tests/test_common_gpu.py`, plus 16 Trainer
checks in `tests/test_gpu_examples.py`. They share implementations with the CPU checks.

Run the single-GPU cases from the PEFT repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -k cortex -m single_gpu_tests \
  tests/test_common_gpu.py tests/test_gpu_examples.py
```

The distributed training entry covers DDP and DeepSpeed ZeRO-0/2/3. Each engine runs 56
model/role/initialization/precision combinations on random tiny Gemma3, Qwen3, Qwen3-MoE,
GPT-OSS, and OLMoE models. Checks include initial outputs and biases, training updates,
rank agreement, adapter serialization, and restoring a training checkpoint before a third
update. FP32 gradients and updates are compared with a full-global-batch reference;
BF16 checks finite values and checkpoint consistency.

```bash
CUDA_VISIBLE_DEVICES=0,1 CORTEX_NPROC=2 make tests_training_cortex
```

`make tests_training` also calls this target. Reports use a fresh temporary directory
printed by the target; set `CORTEX_TRAINING_OUTPUT` to choose another location. Individual
engines and model combinations can be selected directly:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module tests.training.cortex_distributed --engine zero3 \
  --cases qwen3_moe:shared+routed --inits random --dtypes float32 \
  --output /tmp/cortex-zero3-validation
```

Keep the output path free of glob metacharacters (`[`, `]`, `*`, `?`): DeepSpeed uses
whole-path glob matching when discovering checkpoints. Generated combination directories
use `shared+routed`. A CPU regression exercises checkpoint discovery for all 56 names.

## Reported GPU results (2026-10-26)

The contributor supplied two completed portions of the same two-GPU validation matrix:

| Suite | Passed combinations/cases |
| --- | ---: |
| Single GPU (two independent shards) | 124 / 124 |
| DDP (two ranks) | 56 / 56 |
| ZeRO-0 (two ranks) | 56 / 56 |
| ZeRO-2 (two ranks) | 56 / 56 |
| ZeRO-3 (two ranks) | 56 / 56 |

Both ranks passed each distributed combination. The successful results contain no skips,
timeouts, or missing rank reports. The first run exposed a checkpoint-discovery failure
when list-valued roles introduced brackets into directory names; all three ZeRO engines
were rerun in full after normalizing these names.

These are user-reported runs of the same Cortex sources and shared checks in the development
checkout, using PEFT baseline `eb74a79a4b6ab7f1e4afe7a54a7d862c36d34ecc`. This branch ports
those checks onto its existing upstream base. The reported scope is two GPUs, FP32/BF16,
random/sample initialization, and `offload=none`; it does not establish large-model task
quality, throughput, or offload coverage. Keep the two rank reports as one result per
combination rather than counting them as independent tests.

The reported runs predate test consolidation. Their 124 single-GPU cases included
four consolidation-example checks retained in the development repository. The
upstream entries retain all 120 core GPU cases and all 56 distributed combinations
per engine. Method-specific formulas and structural behavior live in `tests/test_cortex.py`.
Configuration, initialization, native decoder lifecycle, custom-model contracts,
and low-level injection are covered in their corresponding existing test files.
Small model/device helpers use `tests/testing_utils.py`. Collection verifies the GPU entries
on a CPU host; it does not execute CUDA tests.
