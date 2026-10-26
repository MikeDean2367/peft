# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-tune a pretrained causal LM with Cortex or LoRA and save the adapter."""

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import get_peft_model


# This example is also runnable directly from either subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_utils import generation_kwargs, generation_status, parse_response, render_messages, tokenize_assistant_turn
from training_utils import (
    build_adapter_config,
    file_sha256,
    initialize_adapter,
    resolve_deepspeed,
    warmup_kwargs,
    write_training_run,
)


def training_messages(row: dict) -> list[dict] | None:
    """Return the student conversation, or None for legacy plain-text records."""
    if row.get("student_messages") is not None:
        generation = row.get("generation") or {}
        if generation.get("complete") is not True or generation.get("finish_reason") == "length":
            raise ValueError("Incomplete generated QA cannot be used for fine-tuning; regenerate that sample.")
        prompt = row["student_messages"]
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(prompt[-1], dict)
            or prompt[-1].get("role") != "user"
        ):
            raise ValueError("student_messages must contain a prompt ending with a user message.")
        response = row.get("response")
        if not isinstance(response, dict) or response.get("role") != "assistant":
            raise ValueError("A prepared QA record needs an assistant response.")
        messages = prompt + [response]
    elif row.get("messages") is not None:
        messages = row["messages"]
    elif isinstance(row.get("text"), str) and row["text"].strip():
        return None
    else:
        raise ValueError("Expected student_messages and response, a messages list, or nonempty text.")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("A chat record needs prompt messages and a final assistant answer.")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TypeError("Every chat message needs a role and string content.")
        if message.get("role") not in {"system", "developer", "user", "assistant"}:
            raise ValueError("This example accepts system, developer, user, and assistant messages.")
    if messages[-2]["role"] != "user" or messages[-1]["role"] != "assistant":
        raise ValueError("Chat records must end with a user question and an assistant answer.")
    if not messages[-1]["content"].strip():
        raise ValueError("The final assistant answer must not be empty.")
    return deepcopy(messages)


def prepare_training_record(
    row: dict,
    tokenizer,
    model_type: str,
    *,
    max_length: int,
    train_reasoning: bool = True,
    reasoning_effort: str = "medium",
    statistics: dict | None = None,
) -> dict[str, list[int]]:
    """Right-truncate exact native tokens and supervise the final assistant turn."""
    if max_length < 2:
        raise ValueError("The training length must retain at least two tokens.")
    messages = training_messages(row)
    if messages is None:
        full_ids = tokenizer(row["text"], truncation=False)["input_ids"]
        answer_start = 0
    else:
        full_ids, answer_start = tokenize_assistant_turn(
            tokenizer,
            messages,
            model_type,
            include_reasoning=train_reasoning,
            reasoning_effort=reasoning_effort,
        )
    ids = full_ids[:max_length]
    if statistics is not None:
        statistics["truncated_records"] += int(len(full_ids) > len(ids))
        statistics["records"] += 1
    if len(ids) < 2 or len(ids) <= answer_start:
        if statistics is not None:
            statistics["records_without_targets"] += 1
        raise ValueError(
            "Right truncation leaves no assistant tokens to supervise "
            f"(original length={len(full_ids)}, answer starts at={answer_start}, max_length={max_length})."
        )
    labels = [-100] * answer_start + ids[answer_start:]
    if statistics is not None:
        statistics["supervised_tokens"] += sum(label != -100 for label in labels[1:])
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--backend", choices=["cortex", "lora"], default="cortex")
    parser.add_argument("--revision", help="Pin the base model and tokenizer to a Hub revision")
    parser.add_argument("--train-file", type=Path, required=True, help="Prepared QA, messages, or plain text JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cortex-adapter"))
    parser.add_argument("--merge-output", type=Path, help="Optionally export a native merged model")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--init-weights", choices=["random", "sample"], default="sample")
    parser.add_argument("--moe-target", nargs="+", choices=["shared", "routed"], default=["shared"])
    parser.add_argument("--public-weighting", choices=["unweighted", "routing"], default="unweighted")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--experts-implementation",
        choices=["eager", "grouped_mm"],
        help="Override the native MoE experts backend",
    )
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override the epoch budget; -1 uses epochs")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-train-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", choices=["cosine", "linear", "constant"], default="cosine")
    parser.add_argument("--deepspeed", default="none", help="zero0, none, or a DeepSpeed JSON configuration path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt", help="Optionally generate one example response after saving, without reloading the model"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--train-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the final assistant's reasoning in CE; --no-train-reasoning omits it",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Inference thinking mode; GPT-OSS uses its native reasoning controls",
    )
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    args = parser.parse_args(argv)
    if min(args.max_train_length, args.batch_size, args.max_new_tokens, args.gradient_accumulation_steps) < 1:
        parser.error("Lengths, batch size, and gradient accumulation must be positive.")
    if args.max_steps != -1 and args.max_steps < 1:
        parser.error("--max-steps must be positive or -1 to use epochs.")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        parser.error("--epochs must be finite and positive.")
    if not 0 <= args.warmup_ratio < 1 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Warmup ratio must be in [0, 1); weight decay must be finite and nonnegative.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if args.backend == "cortex" and args.k < 1:
        parser.error("--k must be positive for Cortex.")
    if args.backend == "lora" and (args.r < 1 or args.lora_alpha < 1):
        parser.error("--r and --lora-alpha must be positive for LoRA.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("Choose a new --output-dir for each training run.")
    if args.merge_output is not None and args.merge_output.exists():
        raise ValueError("Choose a new --merge-output directory.")
    if Path(args.model_name_or_path).is_dir():
        args.model_name_or_path = str(Path(args.model_name_or_path).resolve())
    data_identity = {"training_file_sha256": file_sha256(args.train_file)}
    set_seed(args.seed)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        deepspeed=resolve_deepspeed(
            args.deepspeed, batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps
        ),
        **warmup_kwargs(args.warmup_ratio),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        bf16=args.dtype == "bfloat16",
        use_cpu=args.use_cpu,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
    )
    if training_args.world_size != 1 or training_args.n_gpu > 1:
        raise ValueError("This example uses one device. Select one accelerator, for example CUDA_VISIBLE_DEVICES=0.")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = load_dataset("json", data_files=str(args.train_file), split="train")

    if not len(dataset):
        raise ValueError("The training file must contain at least one record.")
    config = AutoConfig.from_pretrained(args.model_name_or_path, revision=args.revision)
    if config.model_type == "gpt_oss" and getattr(config, "quantization_config", None):
        raise ValueError("Use a dequantized BF16 GPT-OSS checkpoint for this example, not the MXFP4 Hub weights.")
    if config.model_type == "gpt_oss" and args.thinking is False:
        raise ValueError(
            "GPT-OSS has no no-thinking switch; use --reasoning-effort low to reduce its reasoning budget."
        )
    first_messages = training_messages(dataset[0])
    chat_mode = first_messages is not None
    inference_thinking = config.model_type in {"qwen3", "gpt_oss"}
    for row in dataset:
        messages = training_messages(row)
        if (messages is not None) != chat_mode:
            raise ValueError("Use one input format per file: chat records or plain text.")
        if messages and args.train_reasoning and messages[-1].get("reasoning_content"):
            inference_thinking = True
    if args.thinking is not None and config.model_type != "gpt_oss":
        inference_thinking = args.thinking

    preprocessing = {"records": 0, "truncated_records": 0, "records_without_targets": 0, "supervised_tokens": 0}
    dataset = dataset.map(
        lambda row: prepare_training_record(
            row,
            tokenizer,
            config.model_type,
            max_length=args.max_train_length,
            train_reasoning=args.train_reasoning,
            reasoning_effort=args.reasoning_effort,
            statistics=preprocessing,
        ),
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    print(json.dumps({"preprocessing": preprocessing}, sort_keys=True))
    load_kwargs = {
        "dtype": dtype,
        "use_kernels": False,
    }
    if args.experts_implementation is not None:
        load_kwargs["experts_implementation"] = args.experts_implementation
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, revision=args.revision, **load_kwargs)
    base_model.config.use_cache = False
    model = get_peft_model(base_model, build_adapter_config(args, config, revision=args.revision))
    initialize_adapter(model, args, config)
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    result = trainer.train()
    model = trainer.accelerator.unwrap_model(trainer.model_wrapped, keep_fp32_wrapper=False)
    model.save_pretrained(args.output_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(args.output_dir)
    write_training_run(
        args,
        result,
        objective="ce",
        data=data_identity,
        loss_history=trainer.state.log_history,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        preprocessing=preprocessing,
    )
    print(f"Adapter and tokenizer saved to {args.output_dir}.")

    if args.prompt is not None:
        device = training_args.device
        if chat_mode:
            inference_messages = first_messages[:-1]
            inference_messages[-1]["content"] = args.prompt
            prefix = render_messages(
                tokenizer,
                inference_messages,
                config.model_type,
                add_generation_prompt=True,
                enable_thinking=inference_thinking,
                reasoning_effort=args.reasoning_effort,
            )
            ids = torch.tensor([prefix], dtype=torch.long, device=device)
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        else:
            inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
            prefix = inputs["input_ids"][0].tolist()
        model.eval()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **{"do_sample": False, **generation_kwargs(inference_thinking if chat_mode else False)},
            )
        response_ids = generated[0, len(prefix) :].tolist()
        if chat_mode:
            response, complete = parse_response(
                tokenizer,
                response_ids,
                config.model_type,
                prefix_ids=prefix,
                enable_thinking=inference_thinking,
                eos_token_ids=model.generation_config.eos_token_id,
            )
            generation = generation_status(
                tokenizer,
                response_ids,
                args.max_new_tokens,
                complete=complete,
                eos_token_ids=model.generation_config.eos_token_id,
            )
            print(json.dumps({"response": response, "generation": generation}, ensure_ascii=False))
        else:
            print(tokenizer.decode(response_ids, skip_special_tokens=True))

    if args.merge_output is not None:
        merged = model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(args.merge_output)
        tokenizer.save_pretrained(args.merge_output)
        print(f"Native model saved to {args.merge_output}.")


if __name__ == "__main__":
    main()
