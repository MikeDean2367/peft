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

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if (EXAMPLES / "cortex").is_dir():
    EXAMPLES /= "cortex"
sys.path.insert(0, str(EXAMPLES))
import chat_utils  # noqa: E402


spec = importlib.util.spec_from_file_location("cortex_finetune_example", EXAMPLES / "finetune" / "finetune.py")
finetune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune)

# Small offline templates exercise the models' distinct field and channel contracts.
# They deliberately do not implement each other's reasoning field aliases.
# docstyle-ignore
QWEN_TEMPLATE = """{% for m in messages %}{{ '<|im_start|>' + m.role + '\n' }}{% if m.role == 'assistant' %}{{ '<think>\n' + (m.reasoning_content | default('')) + '\n</think>\n\n' + m.content }}{% else %}{{ m.content }}{% endif %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% if enable_thinking is defined and not enable_thinking %}{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}"""
# docstyle-ignore
OSS_TEMPLATE = """{% for m in messages %}{% if m.role == 'assistant' %}{% if m.thinking is defined %}{{ '<|start|>assistant<|channel|>analysis<|message|>' + m.thinking + '<|end|>' }}{% endif %}{{ '<|start|>assistant<|channel|>final<|message|>' + m.content + '<|return|>' }}{% else %}{{ '<|start|>' + m.role + '<|message|>' + m.content + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start|>assistant' }}{% endif %}"""


# docstyle-ignore
QWEN_THINKING_TEMPLATE = """{% for m in messages %}{{ '<|im_start|>' + m.role + '\n' }}{% if m.role == 'assistant' %}{{ '<think>\n' + (m.reasoning_content | default('')) + '\n</think>\n\n' + m.content }}{% else %}{{ m.content }}{% endif %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}"""


@pytest.fixture
def tokenizer_factory():
    def make(model_type="qwen3", *, thinking_only=False):
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        vocab = {token: i for i, token in enumerate(alphabet)}
        merges = []
        if thinking_only:
            # Match the newline token-boundary issue in Qwen3-4B-Thinking-2507.
            vocab["ĊĊ"] = len(vocab)
            merges = [("Ċ", "Ċ")]
        backend = Tokenizer(models.BPE(vocab=vocab, merges=merges))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=not thinking_only)
        backend.decoder = decoders.ByteLevel()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            eos_token="<|return|>" if model_type == "gpt_oss" else "<|im_end|>",
            pad_token="<pad>",
            unk_token="<unk>",
        )
        tokenizer.chat_template = OSS_TEMPLATE if model_type == "gpt_oss" else QWEN_TEMPLATE
        if thinking_only:
            tokenizer.chat_template = QWEN_THINKING_TEMPLATE
        return tokenizer

    return make


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def canonical_row():
    return {
        "id": "seis-0001",
        "kind": "target",
        "teacher_messages": [
            {"role": "system", "content": "SOURCE_ONLY_SECRET"},
            {"role": "user", "content": "QUESTION_MARKER"},
        ],
        "student_messages": [{"role": "user", "content": "QUESTION_MARKER"}],
        "response": {
            "role": "assistant",
            "reasoning_content": "REASONING_MARKER",
            "content": "FINAL_MARKER",
        },
        "generation": {"finish_reason": "stop", "complete": True},
    }


@pytest.mark.parametrize("model_type", ["qwen3", "gpt_oss"])
@pytest.mark.parametrize("include_reasoning", [True, False])
def test_cortex_chat_renders_reasoning_without_mutating_messages(tokenizer_factory, model_type, include_reasoning):
    tokenizer = tokenizer_factory(model_type)
    row = canonical_row()
    messages = row["student_messages"] + [row["response"]]
    before = copy.deepcopy(messages)
    ids = chat_utils.render_messages(tokenizer, messages, model_type, include_reasoning=include_reasoning)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert ("REASONING_MARKER" in text) == include_reasoning
    assert "FINAL_MARKER" in text
    assert "SOURCE_ONLY_SECRET" not in text
    assert messages == before


def test_cortex_chat_rejects_unsupported_reasoning(tokenizer_factory):
    tokenizer = tokenizer_factory()
    row = canonical_row()
    with pytest.raises(ValueError):
        chat_utils.render_messages(tokenizer, row["student_messages"] + [row["response"]], "llama")


def test_cortex_oss_date_override_survives_helper_reload_without_changing_template(tokenizer_factory, tmp_path):
    tokenizer = tokenizer_factory("gpt_oss")
    template = '{{ "Knowledge cutoff: 2024-06\\nCurrent date: " + strftime_now("%Y-%m-%d") + "\\n" }}' + OSS_TEMPLATE
    tokenizer.chat_template = template
    messages = [{"role": "user", "content": "The event took place on 2030-01-02."}]
    expected = chat_utils.render_messages(tokenizer, messages, "gpt_oss", add_generation_prompt=True)
    text = tokenizer.decode(expected)
    assert "Current date: xxxx-xx-xx" in text
    assert "Knowledge cutoff: 2024-06" in text
    assert messages[0]["content"] in text
    assert tokenizer.chat_template == template

    tokenizer.save_pretrained(tmp_path)
    restored = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    assert restored.chat_template == template
    assert chat_utils.render_messages(restored, messages, "gpt_oss", add_generation_prompt=True) == expected
    # The callable belongs to the render call, not the serialized tokenizer.
    native = restored.apply_chat_template(messages, tokenize=False, strftime_now=lambda _: "2035-06-07")
    assert "Current date: 2035-06-07" in native


@pytest.mark.parametrize("opening_in_prefix", [True, False])
def test_cortex_qwen_parses_reasoning_and_final_code(tokenizer_factory, opening_in_prefix):
    tokenizer = tokenizer_factory()
    prefix = "<|im_start|>assistant\n" + ("<think>\n" if opening_in_prefix else "")
    text = ("" if opening_in_prefix else "<think>\n") + (
        "Consider this draft:\n```python\nx = 0\n```\n</think>\n\n```python\nx = 1\n```<|im_end|>"
    )
    response, complete = chat_utils.parse_response(
        tokenizer, encode(tokenizer, text), "qwen3", prefix_ids=encode(tokenizer, prefix), enable_thinking=True
    )
    assert complete
    assert "x = 0" in response["reasoning_content"]
    assert response["content"] == "```python\nx = 1\n```"
    assert "x = 0" not in response["content"]


@pytest.mark.parametrize(
    "text",
    [
        "<think>unfinished reasoning<|im_end|>",
        "<think>done</think><|im_end|>",
        "<think>first</think><think>second</think>final<|im_end|>",
    ],
)
def test_cortex_qwen_rejects_incomplete_or_malformed_reasoning(tokenizer_factory, text):
    tokenizer = tokenizer_factory()
    _, complete = chat_utils.parse_response(tokenizer, encode(tokenizer, text), "qwen3", enable_thinking=True)
    assert not complete


def test_cortex_qwen_plain_code_can_contain_think_literal(tokenizer_factory):
    tokenizer = tokenizer_factory()
    text = '```python\nprint("<think>literal</think>")\n```'
    response, complete = chat_utils.parse_response(
        tokenizer, encode(tokenizer, text + "<|im_end|>"), "qwen3", enable_thinking=False
    )
    assert complete and response["content"] == text
    assert not response.get("reasoning_content")


@pytest.mark.parametrize(
    "start,channel,message,end,stop",
    [
        ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>"),
        ("<|im_start|>", "<|meta_sep|>", "<|im_sep|>", "<|im_end|>", "<|fim_suffix|>"),
    ],
)
def test_cortex_oss_parses_native_analysis_final(tokenizer_factory, start, channel, message, end, stop):
    tokenizer = tokenizer_factory("gpt_oss")
    tokenizer.add_special_tokens({"eos_token": stop})
    prefix = start + "assistant"
    text = (
        channel
        + "analysis"
        + message
        + "DRAFT"
        + end
        + start
        + "assistant"
        + channel
        + "final"
        + message
        + "FINAL"
        + stop
    )
    response, complete = chat_utils.parse_response(
        tokenizer, encode(tokenizer, text), "gpt_oss", prefix_ids=encode(tokenizer, prefix), enable_thinking=True
    )
    assert complete
    assert response["reasoning_content"] == "DRAFT"
    assert response["content"] == "FINAL"
    _, complete = chat_utils.parse_response(
        tokenizer,
        encode(tokenizer, channel + "analysis" + message + "unfinished" + end),
        "gpt_oss",
        prefix_ids=encode(tokenizer, prefix),
        enable_thinking=True,
    )
    assert not complete


def test_cortex_stop_at_exact_budget_is_complete_but_length_cutoff_is_not(tokenizer_factory):
    tokenizer = tokenizer_factory()
    complete_ids = encode(tokenizer, "final<|im_end|>")
    status = chat_utils.generation_status(tokenizer, complete_ids, len(complete_ids), complete=True)
    assert status == {"finish_reason": "stop", "complete": True}
    truncated_ids = encode(tokenizer, "final")
    status = chat_utils.generation_status(tokenizer, truncated_ids, len(truncated_ids), complete=True)
    assert status == {"finish_reason": "length", "complete": False}
    status = chat_utils.generation_status(tokenizer, complete_ids, len(complete_ids), complete=False)
    assert not status["complete"]


@pytest.mark.parametrize("model_type", ["qwen3", "gpt_oss"])
@pytest.mark.parametrize("train_reasoning", [True, False])
def test_cortex_finetune_uses_student_prompt_and_answer_only_loss(tokenizer_factory, model_type, train_reasoning):
    tokenizer = tokenizer_factory(model_type)
    row = canonical_row()
    before = copy.deepcopy(row)
    encoded = finetune.prepare_training_record(
        row, tokenizer, model_type, max_length=512, train_reasoning=train_reasoning
    )
    assert row == before
    assert len(encoded["input_ids"]) == len(encoded["labels"]) == len(encoded["attention_mask"]) <= 512
    text = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    supervised = tokenizer.decode([t for t in encoded["labels"] if t != -100], skip_special_tokens=True)
    assert "QUESTION_MARKER" in text and "QUESTION_MARKER" not in supervised
    assert "SOURCE_ONLY_SECRET" not in text
    assert "FINAL_MARKER" in supervised
    assert ("REASONING_MARKER" in supervised) == train_reasoning
    assert all(encoded["attention_mask"])


def test_cortex_finetune_standard_messages_matches_canonical(tokenizer_factory):
    tokenizer = tokenizer_factory()
    row = canonical_row()
    canonical = finetune.prepare_training_record(row, tokenizer, "qwen3", max_length=512)
    messages = finetune.prepare_training_record(
        {"messages": row["student_messages"] + [row["response"]]}, tokenizer, "qwen3", max_length=512
    )
    assert canonical == messages


def test_cortex_finetune_rejects_incomplete_and_unsupervised_rows(tokenizer_factory):
    tokenizer = tokenizer_factory()
    row = canonical_row()
    row["generation"]["complete"] = False
    with pytest.raises(ValueError):
        finetune.prepare_training_record(row, tokenizer, "qwen3", max_length=512)
    row["generation"]["complete"] = True
    with pytest.raises(ValueError):
        finetune.prepare_training_record(row, tokenizer, "qwen3", max_length=8)


def test_cortex_finetune_eos_padding_masks_only_padding(tokenizer_factory):
    tokenizer = tokenizer_factory()
    tokenizer.pad_token = tokenizer.eos_token
    row = canonical_row()
    encoded = finetune.prepare_training_record(row, tokenizer, "qwen3", max_length=512)
    assert any(
        token == tokenizer.eos_token_id and label == token
        for token, label in zip(encoded["input_ids"], encoded["labels"])
    )
    shorter = copy.deepcopy(row)
    shorter["response"]["content"] = "X"
    other = finetune.prepare_training_record(shorter, tokenizer, "qwen3", max_length=512)
    batch = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)([encoded, other])
    assert (batch["attention_mask"] == 0).any()
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def test_cortex_qwen_render_preserves_literal_think_tags_in_final(tokenizer_factory):
    tokenizer = tokenizer_factory()
    # Qwen falls back to parsing content when reasoning_content is absent.
    # docstyle-ignore
    tokenizer.chat_template = """{% for m in messages %}{% if m.role == 'assistant' %}{% if m.reasoning_content is defined %}{{ m.reasoning_content + 'FINAL:' + m.content }}{% elif '</think>' in m.content %}{{ 'FALLBACK:' + m.content.split('</think>')[-1] }}{% else %}{{ m.content }}{% endif %}{% else %}{{ m.content }}{% endif %}{% endfor %}"""
    final = '```python\nprint("<think>literal</think>")\n```'
    ids = chat_utils.render_messages(
        tokenizer,
        [{"role": "user", "content": "Question"}, {"role": "assistant", "content": final}],
        "qwen3",
    )
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    assert "FALLBACK:" not in rendered
    assert final in rendered


@pytest.mark.parametrize("train_reasoning", [True, False])
@pytest.mark.parametrize("has_reasoning", [True, False])
def test_cortex_thinking_only_final_turn_preserves_user_mask(tokenizer_factory, train_reasoning, has_reasoning):
    tokenizer = tokenizer_factory(thinking_only=True)
    row = canonical_row()
    if not has_reasoning:
        row["response"].pop("reasoning_content")
    encoded = finetune.prepare_training_record(
        row, tokenizer, "qwen3", max_length=512, train_reasoning=train_reasoning
    )
    expected_ids, boundary = chat_utils.tokenize_assistant_turn(
        tokenizer, row["student_messages"] + [row["response"]], "qwen3", include_reasoning=train_reasoning
    )
    assert encoded["input_ids"] == expected_ids
    assert encoded["labels"][:boundary] == [-100] * boundary
    assert encoded["labels"][boundary:] == expected_ids[boundary:]
    supervised = tokenizer.decode([token for token in encoded["labels"] if token != -100])
    assert "QUESTION_MARKER" not in supervised and "SOURCE_ONLY_SECRET" not in supervised
    assert "FINAL_MARKER" in supervised
    assert ("REASONING_MARKER" in supervised) == (train_reasoning and has_reasoning)
    if not train_reasoning or not has_reasoning:
        # The one-newline generation prefix is not an exact token prefix of the
        # full empty-reasoning turn, but only assistant tokens may cross the boundary.
        prefix = chat_utils.render_messages(
            tokenizer, row["student_messages"], "qwen3", add_generation_prompt=True, enable_thinking=True
        )
        assert expected_ids[: len(prefix)] != prefix


def test_cortex_thinking_only_rejects_ineffective_disable(tokenizer_factory):
    tokenizer = tokenizer_factory(thinking_only=True)
    with pytest.raises(ValueError, match="thinking"):
        chat_utils.render_messages(
            tokenizer,
            [{"role": "user", "content": "Question"}],
            "qwen3",
            add_generation_prompt=True,
            enable_thinking=False,
        )


def test_cortex_thinking_only_reference_prefix_closes_reasoning(tokenizer_factory):
    tokenizer = tokenizer_factory(thinking_only=True)
    messages = [{"role": "user", "content": "QUESTION_MARKER"}]
    ids = chat_utils.final_answer_prefix(tokenizer, messages, "qwen3")
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert text.endswith("</think>\n\n")
    assert "QUESTION_MARKER" in text
    assert text.rfind("</think>") > text.rfind("<think>")
    assert not text.endswith("<|im_end|>")


def test_cortex_finetune_right_truncation_preserves_answer_labels(tokenizer_factory):
    tokenizer = tokenizer_factory(thinking_only=True)
    row = canonical_row()
    full = finetune.prepare_training_record(row, tokenizer, "qwen3", max_length=512)
    limit = len(full["input_ids"]) - 4
    truncated = finetune.prepare_training_record(row, tokenizer, "qwen3", max_length=limit)
    for name in ("input_ids", "attention_mask", "labels"):
        assert truncated[name] == full[name][:limit]
    assert any(label != -100 for label in truncated["labels"])
    boundary = next(index for index, label in enumerate(full["labels"]) if label != -100)
    with pytest.raises(ValueError):
        finetune.prepare_training_record(row, tokenizer, "qwen3", max_length=boundary)


@pytest.mark.parametrize(
    "model_name,model_type,batch_size,epochs,rank,length",
    [
        ("Qwen/Qwen3-4B-Thinking-2507", "qwen3", 8, 100, 16, 3072),
        ("openai/gpt-oss-20b", "gpt_oss", 4, 50, 8, 2048),
    ],
)
def test_cortex_explicit_ce_recipes_build_expected_adapter_and_zero0(
    model_name, model_type, batch_size, epochs, rank, length
):
    args = finetune.parse_args(
        [
            "--model-name-or-path",
            model_name,
            "--train-file",
            "prepared/train_qa.jsonl",
            "--backend",
            "lora",
            "--batch-size",
            str(batch_size),
            "--epochs",
            str(epochs),
            "--r",
            str(rank),
            "--max-train-length",
            str(length),
            "--deepspeed",
            "zero0",
        ]
    )
    assert args.backend == "lora" and args.max_steps == -1
    assert (args.batch_size, args.epochs, args.r, args.max_train_length) == (batch_size, epochs, rank, length)
    assert args.dtype == "bfloat16" and args.deepspeed == "zero0"
    assert args.learning_rate == 5e-4 and args.weight_decay == 0
    assert args.lr_scheduler_type == "cosine" and args.warmup_ratio == 0.1
    config = finetune.build_adapter_config(args, SimpleNamespace(model_type=model_type))
    assert config.r == rank and config.lora_alpha == 32 and config.lora_dropout == 0
    if model_type == "gpt_oss":
        assert set(config.target_parameters) == {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
        assert not config.target_modules
    else:
        assert config.target_modules == {"gate_proj", "up_proj", "down_proj"}
    ds = finetune.resolve_deepspeed(args.deepspeed, batch_size=args.batch_size, gradient_accumulation_steps=1)
    assert ds["zero_optimization"] == {"stage": 0}
    assert ds["train_micro_batch_size_per_gpu"] == batch_size
    assert ds["train_batch_size"] == batch_size


def test_cortex_explicit_training_settings_and_cpu_mode():
    args = finetune.parse_args(
        [
            "--model-name-or-path",
            "openai/gpt-oss-20b",
            "--train-file",
            "prepared/train_qa.jsonl",
            "--backend",
            "cortex",
            "--use-cpu",
            "--dtype",
            "float32",
            "--k",
            "4",
            "--batch-size",
            "1",
            "--epochs",
            "2",
            "--max-steps",
            "3",
            "--max-train-length",
            "64",
            "--learning-rate",
            "0.001",
            "--no-gradient-checkpointing",
        ]
    )
    assert (args.k, args.batch_size, args.epochs, args.max_steps, args.max_train_length) == (4, 1, 2, 3, 64)
    assert args.dtype == "float32" and args.deepspeed == "none"
    assert args.learning_rate == 0.001 and args.gradient_checkpointing is False
    config = finetune.build_adapter_config(args, SimpleNamespace(model_type="gpt_oss"))
    assert config.k == 4 and config.init_weights == "sample" and config.moe_target == "shared"


@pytest.mark.parametrize("option", [["--preset", "none"], ["--method", "cortex"], ["--max-length", "128"]])
def test_cortex_finetune_rejects_obsolete_cli_aliases(option):
    with pytest.raises(SystemExit) as error:
        finetune.parse_args(["--model-name-or-path", "local-checkpoint", "--train-file", "train.jsonl", *option])
    assert error.value.code == 2


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-4B-Thinking-2507", "openai/gpt-oss-20b"])
def test_cortex_finetune_defaults_do_not_infer_research_recipe(model_name):
    args = finetune.parse_args(["--model-name-or-path", model_name, "--train-file", "train.jsonl"])
    assert (args.backend, args.k, args.r, args.batch_size, args.epochs) == ("cortex", 64, 16, 1, 1)
    assert args.init_weights == "sample" and args.deepspeed == "none" and args.prompt is None
