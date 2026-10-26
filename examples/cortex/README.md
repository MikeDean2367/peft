# Cortex examples

Cortex adds trainable nonlinear FFN channels to a frozen model. Create, train, save, and load the adapter through the usual PEFT API. The adapter itself does not require a particular loss or data-generation procedure.

| Example | Workflow |
| --- | --- |
| [finetune/](finetune/README.md) | Ordinary cross-entropy fine-tuning with Cortex or LoRA and Transformers `Trainer`; optional native model export. |
| [consolidate/](consolidate/README.md) | Generate questions and teacher targets from SEIS documentation, train with KL and optional hidden-state MSE, then evaluate without that documentation. |

The preparation stage writes a portable `train_qa.jsonl` with teacher/student messages and an assistant response separating reasoning from final content. Fine-tuning can reuse these messages with another model and tokenizer, without teacher tensors. Distillation uses the original model's token IDs and cached targets. Both training paths use the same [evaluation entry point](consolidate/README.md#evaluate-ce-or-distillation).

All training settings are explicit CLI options. The READMEs give separate research recipes for `Qwen/Qwen3-4B-Thinking-2507` and a materialized GPT-OSS-20B BF16 checkpoint. Model names do not change the defaults. The generic defaults are Cortex, one epoch, batch size one, and no DeepSpeed. The documented larger recipes still require full task-quality validation on the intended GPU environment.

`chat_utils.py` handles native Qwen3/GPT-OSS conversation and reasoning formats. GPT-OSS rendering fixes the automatic current date to `xxxx-xx-xx` through the template's `strftime_now` argument. It does not alter dates in user messages or the tokenizer files; use the helper again after reloading a tokenizer. `training_utils.py` shares adapter configuration, single-device training setup, and the `run.json` record.

The consolidation example demonstrates one context-to-parameters update. PEFT supplies the structural adapter; question generation, distillation, and evaluation are example code. Continual replay and target regeneration belong in the broader [Cortex research repository](https://github.com/zjunlp/cortex).

## Running the examples

Install this PEFT checkout with `python -m pip install -e .` from the repository
root, then change into `examples/cortex/finetune` or
`examples/cortex/consolidate` before following that directory's README. Commands
use the active Python environment. Preparation and training write to explicitly
named output directories; there are no pre-existing experiment caches to reuse.

## Environment

Use a PEFT source checkout containing Cortex, plus PyTorch, Transformers, Accelerate, Datasets, and Safetensors. Keep the PyTorch build appropriate for your accelerator. The CPU regression environment uses Python 3.13, PyTorch 2.10.0, Transformers 5.17.0, Accelerate 1.15.0, Datasets 5.0.1, and Safetensors 0.8.0. The documented GPT-OSS layouts and training APIs use this Transformers/Accelerate combination; earlier releases are not uniformly supported.

The GPU recipes below target the existing PyTorch 2.9.0 / CUDA 12.9 environment with DeepSpeed 0.18.9. Their full task runs have not yet been validated. DeepSpeed is optional: `--deepspeed zero0` uses the single-device ZeRO-0 path and `torchrun`; `--deepspeed none` uses ordinary Trainer and can run with `python`. Other ZeRO stages are outside these single-device examples. Only explicit code scoring needs `seismic-linter==0.2.0`, in the execution environment you provide.

Training saves adapter weights, adapter configuration, tokenizer, and one `run.json` in `--output-dir`. The record contains settings, input identity, loss history, actual optimizer steps, and trainable parameter count. Checkpoint round-trip, frozen-base, adapter-update, and dtype checks live in [test_cortex_example_precision.py](../../tests/test_cortex_example_precision.py). End-to-end preparation, training, evaluation, and native-export checks live in [test_cortex_consolidation_example.py](../../tests/test_cortex_consolidation_example.py). These regression tests run separately from training jobs. See [scope and usage](finetune/scope_and_usage.md) for a custom classification MLP example.
