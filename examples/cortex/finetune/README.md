# Cross-entropy fine-tuning with Cortex or LoRA

`finetune.py` uses Transformers `Trainer` for ordinary cross-entropy training. It accepts the `train_qa.jsonl` produced by [consolidation preparation](../consolidate/README.md), a conversational JSONL with `messages`, or a JSONL with `text`. It needs no teacher model, logits, or hidden states.

For prepared records, the student input is `student_messages + [response]`. `teacher_messages` and the source documentation are excluded. The script renders the conversation with the training model's tokenizer and supervises the final assistant turn only; earlier messages and padding receive label `-100`. Ordinary `messages` must end in a user turn followed by an assistant turn. Plain `text` supervises every non-padding token.

The canonical `response.reasoning_content` field maps to the native Qwen3 or GPT-OSS format. Reasoning is included in CE by default; `--no-train-reasoning` removes it from both inputs and labels. This choice does not disable the model's thinking mode during generation. Qwen3-4B-Thinking-2507 is thinking-only. The JSONL can be reused across models; exact cached tokens and distillation tensors cannot.

## Explicit training recipes

Run from this directory after preparing and checking the QA file. The following LoRA CE recipes use one GPU, BF16, DeepSpeed ZeRO-0, and non-reentrant activation checkpointing. See the [environment](../README.md#environment) for dependencies. No preset or model-name inference supplies these settings.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 finetune.py \
  --model-name-or-path Qwen/Qwen3-4B-Thinking-2507 \
  --train-file ../consolidate/outputs/qwen-prepared/train_qa.jsonl \
  --output-dir outputs/qwen-lora-ce \
  --backend lora --r 16 --lora-alpha 32 \
  --batch-size 8 --epochs 100 --max-train-length 3072 \
  --dtype bfloat16 --deepspeed zero0 --gradient-checkpointing \
  --learning-rate 5e-4 --lr-scheduler-type cosine --warmup-ratio 0.1 \
  --weight-decay 0 --gradient-accumulation-steps 1

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 finetune.py \
  --model-name-or-path /path/to/gpt-oss-20b-bf16 \
  --train-file ../consolidate/outputs/oss-prepared/train_qa.jsonl \
  --output-dir outputs/oss-lora-ce \
  --backend lora --r 8 --lora-alpha 32 \
  --batch-size 4 --epochs 50 --max-train-length 2048 \
  --dtype bfloat16 --deepspeed zero0 --gradient-checkpointing \
  --learning-rate 5e-4 --lr-scheduler-type cosine --warmup-ratio 0.1 \
  --weight-decay 0 --gradient-accumulation-steps 1
```

LoRA targets gate/up/down projections in dense models and packed expert gate/up and down parameters in GPT-OSS. LoRA dropout is zero. GPT-OSS A factors use normal initialization with standard deviation 0.02 and B factors start at zero; dense models retain PEFT's default initialization. Scaling is `alpha / r`, and routers stay frozen.

For Cortex CE, replace the LoRA options with `--backend cortex --k 64 --init-weights sample` for Qwen. For GPT-OSS, use `--backend cortex --k 64 --init-weights random --moe-target shared`. Choose the epoch budget explicitly: the corresponding Cortex recipes use 100 and 300 epochs. These remain CE-only runs; KL/MSE training is in `consolidate.py`. Width and rank do not imply equal parameter budgets.

GPT-OSS requires an already materialized BF16 checkpoint; the example does not convert MXFP4. Attention and expert backends follow Transformers defaults. `--experts-implementation eager` explicitly selects reference experts when needed. `use_kernels=False` avoids Hub kernels that replace the MoE parent forward, while leaving native expert implementations available. Use a compatible PyTorch/GPU for optimized expert kernels.

`--deepspeed none` runs ordinary Trainer. A ZeRO-0 JSON path is also accepted. For a small CPU check, choose a tiny model and pass `--use-cpu --dtype float32 --deepspeed none`. `--max-steps` can replace the epoch budget. Training settings are not changed according to model name.

`--max-train-length` right-truncates the whole student sequence, including chat formatting, question, reasoning, and answer. The script rejects records with no remaining assistant targets and records truncation counts. Prepared source answers must still be complete. A long reasoning trace can consume the entire training budget before the final answer.

## Save and evaluate

Training saves the adapter and tokenizer directly into `--output-dir`, with one `run.json` recording settings, the training-file checksum, loss history, steps, trainable parameters, and preprocessing counts. Base weights remain frozen. PEFT normally promotes new adapter parameters to FP32; the training backend controls subsequent mixed-precision execution and optimizer state.

For SEIS data, use the shared evaluation entry point for CE and distillation:

```bash
CUDA_VISIBLE_DEVICES=0 python ../consolidate/eval.py \
  --data-dir ../consolidate/outputs/qwen-prepared \
  --adapter outputs/qwen-lora-ce \
  --output-file outputs/qwen-lora-ce/responses.json
```

The evaluator loads the adapter's actual base checkpoint and tokenizer, including when CE used a different model from the teacher. It checks the QA checksum in `run.json`. Keep that file with the adapter. It does not read teacher tensors. See [evaluation and explicit code scoring](../consolidate/README.md#evaluate-ce-or-distillation).

To print one illustrative answer immediately after training, supply `--prompt "How can I use seismic_linter to check Python code?"`. This optional demo uses the trained model already in memory; it is not held-out evaluation. `--max-new-tokens` controls its generation budget, and `--thinking` / `--reasoning-effort` control supported native reasoning modes.

`--merge-output outputs/native` additionally saves a native model. Cortex expands compatible FFNs; LoRA merges into their original weight shapes. The training script does not reload or compare the export. When loading a GPT-OSS native export with an added width that violates an optimized expert kernel's alignment requirements, explicitly choose `experts_implementation="eager"` in a compatible Transformers version. Keep the adapter checkpoint if you need adapter switching.

## Separate-process inference

A saved adapter uses the ordinary PEFT API. For a Qwen3 adapter:

```python
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_dir = "outputs/qwen-lora-ce"
config = PeftConfig.from_pretrained(adapter_dir)
base = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    revision=config.revision,
    dtype=torch.bfloat16,
    use_kernels=False,
).to("cuda")
model = PeftModel.from_pretrained(base, adapter_dir).eval()
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "How can I check Python code with seismic_linter?"}],
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=True,
).to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=4096)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))
```

For GPT-OSS, use `chat_utils.render_messages()` to preserve the example's fixed-date prompt after loading the tokenizer. For local models, pass a local checkpoint directory; Hub models download on first use. Use node-local `HF_DATASETS_CACHE` for preprocessing.
