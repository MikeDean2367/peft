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

import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


# The standalone development tree and the final PEFT checkout use different example roots.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if (EXAMPLES / "cortex").is_dir():
    EXAMPLES /= "cortex"
spec = importlib.util.spec_from_file_location(
    "cortex_consolidation_example", EXAMPLES / "consolidate" / "consolidate.py"
)
consolidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consolidate)
# eval.py imports its sibling script when launched normally; provide the same module here.
sys.modules["consolidate"] = consolidate
score_spec = importlib.util.spec_from_file_location("cortex_seis_evaluation", EXAMPLES / "consolidate" / "eval.py")
scoring = importlib.util.module_from_spec(score_spec)
score_spec.loader.exec_module(scoring)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    words = ["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
    words += [
        "system",
        "user",
        "assistant",
        "You",
        "are",
        "a",
        "helpful",
        "concise",
        "question",
        "France",
        "Lyon",
        "Paris",
        "capital",
        "updated",
        "fact",
        "old",
        "world",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "which",
        "city",
        "is",
        "the",
        "of",
        "What",
        "Name",
        "Italy",
        "Rome",
        "Japan",
        "Tokyo",
        "Mercury",
        "Mars",
        "sun",
        "cat",
        "dog",
        "math",
        "water",
        "planet",
        "code",
        "fruit",
        "winter",
    ]
    vocab = {word: i for i, word in enumerate(dict.fromkeys(words))}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )
    tokenizer.chat_template = "{{ bos_token }} {% for message in messages %}{{ message['role'] }} : {{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant : {% endif %}"
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=512,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )
    ).eval()
    model.generation_config.forced_eos_token_id = tokenizer.eos_token_id
    model.generation_config.min_new_tokens = 2
    folder = tmp_path / "base"
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


def prepare_args(tmp_path, checkpoint, generic_context="source"):
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps({"type": "context", "package": "seismic_linter", "context": "updated fact France capital Lyon"})
        + "\n"
    )
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(["What city is the capital of France?"]))
    return SimpleNamespace(
        output_dir=tmp_path / "prepared",
        data_file=data,
        seed=42,
        device="cpu",
        dtype="float32",
        model_name_or_path=str(checkpoint),
        revision=None,
        questions_file=questions,
        nq=1,
        generic_context=generic_context,
        thinking=False,
        question_thinking=False,
        question_do_sample=False,
        answer_do_sample=False,
        question_temperature=0.7,
        answer_temperature=0.7,
        answer_attempts=3,
        reasoning_effort="medium",
        teacher_top_k=12,
        kl_weight=1.0,
        mse_weight=1.0,
        kl_scope="oi",
        mse_scope="oi",
        cache_states=True,
        max_train_length=128,
        answer_max_length=256,
        max_new_tokens=4,
        question_max_new_tokens=512,
        question_attempts=3,
        question_max_length=4096,
    )


def write_evaluation_data(path, source_path, probes):
    source = consolidate.read_jsonl(source_path)[0]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in [source, *[{"type": "eval", **probe} for probe in probes]])
    )


def test_cortex_consolidation_alignment_and_research_states(tiny_checkpoint):
    model = consolidate.load_base(str(tiny_checkpoint), None, "float32", "cpu")
    prefix = [1, 10, 11, 12, 13]
    answer = [15, 16, 17]
    modules = [*model.model.layers, model.model.norm]
    hooks_before = [dict(module._forward_hooks) for module in modules]
    logits, states = consolidate.forward_targets(model, prefix + answer, start=len(prefix) - 1)
    assert [dict(module._forward_hooks) for module in modules] == hooks_before
    with torch.no_grad():
        full = model(torch.tensor([prefix + answer]), output_hidden_states=True, use_cache=False)
    # Cached logits and states use the same token positions; the loss applies the causal mask.
    torch.testing.assert_close(logits, full.logits[0, 4:])
    torch.testing.assert_close(states, torch.stack([state[0, 4:] for state in full.hidden_states[1:]]))
    # Teacher and student have independent prefix lengths, with the same target token sequence.
    student_prefix = [1, 10]
    student_logits, student_states = consolidate.forward_targets(
        model, student_prefix + answer, start=len(student_prefix) - 1
    )
    full_student = model(torch.tensor([student_prefix + answer]), output_hidden_states=True, use_cache=False)
    torch.testing.assert_close(student_logits, full_student.logits[0, 1:])
    torch.testing.assert_close(student_states, torch.stack([state[0, 1:] for state in full_student.hidden_states[1:]]))


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_cortex_consolidation_qwen_state_alignment(model_type):
    config = AutoConfig.for_model(
        model_type,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    with torch.no_grad():
        logits, states = consolidate.forward_targets(model, [1, 3, 5, 7, 9], start=2)
        full = model(torch.tensor([[1, 3, 5, 7, 9]]), output_hidden_states=True, use_cache=False)
    torch.testing.assert_close(logits, full.logits[0, 2:])
    torch.testing.assert_close(states, torch.stack([state[0, 2:] for state in full.hidden_states[1:]]))


def test_cortex_consolidation_truncated_kl_uses_full_student_normalizer():
    logits = torch.tensor([[[1.0, -1.0, 0.2, 1.8], [0.0, 0.0, 0.0, 0.0]]], requires_grad=True)
    states = torch.tensor([[[[2.0, -1.0], [0.0, 0.0]]]], requires_grad=True)
    batch = {
        "teacher_top_logits": torch.tensor([[[2.0, 1.0], [2.0, 1.0]]]),
        "teacher_top_ids": torch.tensor([[[3, 0], [3, 0]]]),
        "teacher_states": torch.zeros_like(states),
        "kl_mask": torch.tensor([[True]]),
        "mse_mask": torch.tensor([[True, False]]),
    }
    loss, kl, mse = consolidate.masked_distillation_loss(logits, states, batch, 2.0, 1.0, 0.25)
    q = torch.softmax(torch.tensor([2.0, 1.0]) / 2, -1)
    p = torch.softmax(logits[0, 0] / 2, -1)[[3, 0]]
    expected_kl = 4 * (q * (q.log() - p.log())).sum()
    torch.testing.assert_close(kl, expected_kl)
    torch.testing.assert_close(mse, torch.tensor(2.5))
    torch.testing.assert_close(loss, expected_kl + 0.625)
    loss.backward()
    assert logits.grad[0, 0, 1].abs() > 0  # Unselected vocabulary still affects normalization.
    assert states.grad.abs().sum() > 0


@pytest.mark.parametrize("generic_context", ["source", "none"])
def test_cortex_consolidation_preparation_context_switch_and_integrity(tmp_path, tiny_checkpoint, generic_context):
    args = prepare_args(tmp_path, tiny_checkpoint, generic_context)
    consolidate.prepare(args)
    manifest, tensors = consolidate.load_prepared(args.output_dir)
    records = manifest["records"]
    assert len(records) == 5
    qa = consolidate.read_jsonl(args.output_dir / "train_qa.jsonl")
    assert manifest["schema_version"] == 4
    assert [r["id"] for r in qa] == [r["id"] for r in records]
    for row in qa:
        assert row["response"]["role"] == "assistant"
        assert row["response"]["content"].strip()
        assert row["generation"]["complete"]
        assert all(m["role"] != "assistant" for m in row["student_messages"])
        assert "updated fact France capital Lyon" not in json.dumps(row["student_messages"])
    assert qa[0]["teacher_messages"] == [
        {"role": "system", "content": "updated fact France capital Lyon"},
        {"role": "user", "content": "What city is the capital of France?"},
    ]
    assert qa[0]["student_messages"] == [{"role": "user", "content": "What city is the capital of France?"}]
    assert [r["kind"] for r in records] == ["target"] + ["generic"] * 4
    assert records[0]["teacher_prompt_ids"] != records[0]["student_prompt_ids"]
    for index, record in enumerate(records):
        length = len(record["student_prompt_ids"]) + len(record["answer_ids"]) - record["student_alignment_start"]
        assert tensors[f"{index}.states"].shape[1] == length
        assert (
            record["teacher_prompt_ids"][record["teacher_alignment_start"] :]
            == record["student_prompt_ids"][record["student_alignment_start"] :]
        )
        if index:
            assert ("updated fact France capital Lyon" in json.dumps(qa[index]["teacher_messages"])) == (
                generic_context == "source"
            )
            assert (record["teacher_prompt_ids"] == record["student_prompt_ids"]) == (generic_context == "none")
    with (args.output_dir / "targets.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        consolidate.load_prepared(args.output_dir)


@pytest.mark.parametrize("method", ["cortex", "lora"])
@pytest.mark.parametrize(
    "run_device,run_dtype",
    [
        ("cpu", "float32"),
        pytest.param("cuda:0", "float32", marks=pytest.mark.single_gpu_tests),
        pytest.param("cuda:0", "bfloat16", marks=pytest.mark.single_gpu_tests),
    ],
)
def test_cortex_consolidation_train_reload_and_held_out_evaluation(
    tmp_path, tiny_checkpoint, method, monkeypatch, run_device, run_dtype
):
    if run_device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        if run_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA BF16 is unavailable")
    prep = prepare_args(tmp_path, tiny_checkpoint)
    prep.device, prep.dtype = run_device, run_dtype
    consolidate.prepare(prep)
    args = consolidate.parse_args(["--stage", "train"])
    vars(args).update(
        data_dir=prep.output_dir,
        output_dir=tmp_path / method,
        device=run_device,
        seed=42,
        backend=method,
        k=4,
        init_weights="sample",
        r=2,
        learning_rate=1e-3,
        epochs=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_train_length=128,
        max_steps=-1,
        deepspeed=None,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        kl_temperature=2.0,
        kl_weight=1.0,
        mse_weight=1.0,
    )
    original_load = consolidate.load_base
    base_parameters = []

    def audited_load(*args, **kwargs):
        model = original_load(*args, **kwargs)
        if not base_parameters:
            base_parameters.extend((p, p.detach().clone()) for p in model.parameters())
        return model

    monkeypatch.setattr(consolidate, "load_base", audited_load)
    from .test_cortex_example_precision import assert_saved_adapter, capture_training

    trained = capture_training(monkeypatch)
    consolidate.train(args)
    assert_saved_adapter(trained[0], tiny_checkpoint, args.output_dir)
    for parameter, before in base_parameters:
        assert not parameter.requires_grad and parameter.grad is None
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    metrics = consolidate.read_json(args.output_dir / "run.json")
    assert metrics["optimizer_steps"] == 6
    assert metrics["preprocessing"]["training_records"] == 5
    assert metrics["trainable_parameters"] > 0
    assert metrics["settings"]["thinking"] == prep.thinking
    assert metrics["settings"]["reasoning_effort"] == prep.reasoning_effort
    assert all(math.isfinite(row["loss"]) for row in metrics["losses"])
    probes = tmp_path / "eval.json"
    write_evaluation_data(
        probes,
        prep.data_file,
        [
            {"split": "target", "question": "Name France capital city", "answers": ["Lyon"]},
            {"split": "target", "question": "What city is the capital of France?", "answers": ["Lyon"]},
            {"split": "retain", "question": "What is the capital of Italy?", "answers": ["Rome"]},
        ],
    )
    evaluation = SimpleNamespace(
        data_dir=prep.output_dir,
        adapter=args.output_dir,
        data_file=probes,
        output_file=tmp_path / "eval_results.json",
        max_new_tokens=4,
        max_eval_length=None,
        thinking=None,
        reasoning_effort=None,
        device=run_device,
    )
    # Evaluation with a trained adapter must work without the distillation cache.
    (prep.output_dir / "targets.safetensors").unlink()
    scoring.evaluate(evaluation)
    report = consolidate.read_json(evaluation.output_file)
    assert report["summary"]["target"]["num_probes"] == 1  # Training question excluded from held-out summary.
    assert report["probes"][1]["overlaps_training_question"]
    assert report["summary"]["retain"]["num_probes"] == 1
    for split in ("target", "retain"):
        for mode in ("base", "context_teacher", "adapted"):
            assert math.isfinite(report["summary"][split][mode]["reference_answer_nll"])


def test_cortex_consolidation_question_generation_path(tmp_path, tiny_checkpoint, monkeypatch):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.questions_file = None
    with args.data_file.open("a") as stream:
        stream.write(json.dumps({"type": "eval", "question": "EVALUATION SECRET", "tests": "TEST SECRET"}) + "\n")
    args.question_do_sample = args.answer_do_sample = True
    original_generate = consolidate.generate_answer
    generation_messages = []
    generation_options = []

    def generate_question_or_answer(model, tokenizer, prefix, max_new_tokens, max_length, **kwargs):
        generation_options.append(kwargs["sampling_kwargs"])
        if max_new_tokens == args.question_max_new_tokens:
            return [1, 9, 2]
        return original_generate(model, tokenizer, prefix, max_new_tokens, max_length, **kwargs)

    # Audit the real question-generation message construction while controlling decoding.
    original_decode = consolidate.AutoTokenizer.from_pretrained

    def load_tokenizer(*a, **kw):
        tokenizer = original_decode(*a, **kw)
        decode = tokenizer.decode
        template = tokenizer.apply_chat_template

        def capture_messages(messages, **opts):
            generation_messages.append(messages)
            return template(messages, **opts)

        tokenizer.apply_chat_template = capture_messages
        tokenizer.decode = lambda ids, **opts: (
            "<question>Write code using seismic_linter?</question>"
            if ids in ([1, 9, 2], [1, 9])
            else decode(ids, **opts)
        )
        return tokenizer

    monkeypatch.setattr(consolidate.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(consolidate, "generate_answer", generate_question_or_answer)
    consolidate.prepare(args)
    manifest, _ = consolidate.load_prepared(args.output_dir)
    assert manifest["nq"] == 1 and len(manifest["records"]) == 5
    assert manifest["question_attempts"] == args.question_attempts
    assert manifest["answer_attempts"] == args.answer_attempts
    assert generation_options == [{"do_sample": True, "temperature": 0.7}] * 6
    assert manifest["question_generation"] == {"do_sample": True, "temperature": 0.7}
    assert manifest["answer_generation"] == {"do_sample": True, "temperature": 0.7}
    qa = consolidate.read_jsonl(args.output_dir / "train_qa.jsonl")
    assert qa[0]["student_messages"][-1]["content"].strip()
    assert generation_messages[0][0]["content"] == consolidate.question_prompt(1)
    assert manifest["context"] in generation_messages[0][1]["content"]
    assert "EVALUATION SECRET" not in json.dumps(generation_messages)
    assert "TEST SECRET" not in json.dumps(generation_messages)


def test_cortex_consolidation_rejects_truncated_question(tmp_path, tiny_checkpoint, monkeypatch):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.questions_file = None
    monkeypatch.setattr(consolidate, "generate_answer", lambda *a, **kw: [1] * args.question_max_new_tokens)
    with pytest.raises(ValueError, match="Question generation"):
        consolidate.prepare(args)
    assert not args.output_dir.exists()
    assert (tmp_path / "prepared.question_generation.txt").is_file()


def mock_question_generations(monkeypatch, args, outputs):
    """Control question output, retaining real answer generation and cache writing."""
    original_generate = consolidate.generate_answer
    original_load = consolidate.AutoTokenizer.from_pretrained
    calls = []

    def generate(model, tokenizer, prefix, max_new_tokens, max_length, **kwargs):
        if max_new_tokens != args.question_max_new_tokens:
            return original_generate(model, tokenizer, prefix, max_new_tokens, max_length, **kwargs)
        index = len(calls)
        calls.append({"prefix": prefix.copy(), "kwargs": kwargs, "random_draw": torch.rand(()).item()})
        output = outputs[index]
        if isinstance(output, Exception):
            raise output
        _, terminated = output
        return [999 + index, 2] if terminated else [999 + index] * max_new_tokens

    def load_tokenizer(*a, **kw):
        tokenizer = original_load(*a, **kw)
        decode = tokenizer.decode

        def decode_question(ids, **opts):
            if len(ids) and 999 <= ids[0] < 999 + len(outputs):
                return outputs[ids[0] - 999][0]
            return decode(ids, **opts)

        tokenizer.decode = decode_question
        return tokenizer

    monkeypatch.setattr(consolidate.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(consolidate, "generate_answer", generate)
    return calls


def test_cortex_question_retry_recovers_from_duplicates_without_reseeding(tmp_path, tiny_checkpoint, monkeypatch):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.questions_file = None
    args.nq = 2
    args.question_do_sample = True
    first = "<question>Use seismic_linter for task one?</question>" * 2
    accepted = ["Use seismic_linter for task two?", "Use seismic_linter for task three?"]
    second = "Here are the questions.\n" + "\n".join(
        f"<question>{q}</question>" for q in [*accepted, "Use seismic_linter for task four?"]
    )
    calls = mock_question_generations(monkeypatch, args, [(first, True), (second, True)])
    consolidate.prepare(args)
    manifest, _ = consolidate.load_prepared(args.output_dir)
    rows = consolidate.read_jsonl(args.output_dir / "train_qa.jsonl")
    assert len(calls) == 2
    assert calls[0]["prefix"] == calls[1]["prefix"]
    assert calls[0]["random_draw"] != calls[1]["random_draw"]
    assert [call["kwargs"]["sampling_kwargs"] for call in calls] == [{"do_sample": True, "temperature": 0.7}] * 2
    assert [row["student_messages"][-1]["content"] for row in rows[:2]] == accepted
    assert len(rows) == 6 and manifest["nq"] == 2
    assert manifest["question_attempts"] == 3
    assert (args.output_dir / "question_generation.txt").read_text().strip() == second
    for attempt, output in enumerate((first, second), 1):
        diagnostic = tmp_path / f"prepared.question_generation.attempt-{attempt}.txt"
        assert diagnostic.read_text().strip() == output


@pytest.mark.parametrize("seed", [7, 42])
def test_cortex_question_pool_fallback_samples_after_all_attempts(tmp_path, tiny_checkpoint, monkeypatch, seed):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.questions_file = None
    args.nq = 3
    args.seed = seed
    args.question_attempts = 5
    pool = [f"Use seismic_linter for task {name}?" for name in ("one", "two", "three", "four", "five", "six")]
    duplicate = pool[1].replace(" ", "  ")
    candidates = [
        pool[:2],
        [duplicate, pool[2]],
        [pool[3], "Use another package?"],
        [pool[0], pool[4]],
        [pool[5], "", "<question>Nested seismic_linter question?"],
    ]
    outputs = [("\n".join(f"<question>{q}</question>" for q in questions), True) for questions in candidates]
    calls = mock_question_generations(monkeypatch, args, outputs)
    consolidate.prepare(args)

    # The union is already large enough after attempt two, but fallback waits for all five.
    assert len(calls) == 5
    selected = random.Random(seed).sample(pool, args.nq)
    assert selected != pool[: args.nq]
    rows = consolidate.read_jsonl(args.output_dir / "train_qa.jsonl")
    assert [row["student_messages"][-1]["content"] for row in rows[: args.nq]] == selected
    assert json.loads((tmp_path / "prepared.questions.json").read_text()) == selected
    manifest, _ = consolidate.load_prepared(args.output_dir)
    assert manifest["nq"] == args.nq and len(manifest["records"]) == args.nq + 4
    combined = (args.output_dir / "question_generation.txt").read_text()
    for attempt, (output, _) in enumerate(outputs, 1):
        assert f"=== Question generation attempt {attempt} ===" in combined
        assert output in combined
        assert (tmp_path / f"prepared.question_generation.attempt-{attempt}.txt").read_text().strip() == output
    assert not (tmp_path / "prepared.question_generation.txt").exists()


def test_cortex_question_pool_keeps_later_complete_attempt_precedence(tmp_path, tiny_checkpoint, monkeypatch):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.questions_file = None
    args.nq = 2
    accepted = ["Use seismic_linter for task three?", "Use seismic_linter for task four?"]
    outputs = [
        ("<question>Use seismic_linter for task one?</question>", True),
        ("<question>Use seismic_linter for task two?</question>", True),
        ("\n".join(f"<question>{q}</question>" for q in accepted), True),
    ]
    calls = mock_question_generations(monkeypatch, args, outputs)
    consolidate.prepare(args)
    assert len(calls) == 3
    rows = consolidate.read_jsonl(args.output_dir / "train_qa.jsonl")
    assert [row["student_messages"][-1]["content"] for row in rows[:2]] == accepted
    assert (args.output_dir / "question_generation.txt").read_text().strip() == outputs[-1][0]


def test_cortex_question_pool_excludes_truncated_and_reasoning_candidates(tmp_path, tiny_checkpoint, monkeypatch):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.questions_file = None
    args.nq = 3
    args.question_attempts = 4
    first = "<question>Use seismic_linter for task one?</question>"
    second = "<question>Use seismic_linter for task two?</question>"
    excluded = "<question>Use seismic_linter for task three?</question>"
    reasoning_output = f"<think>{excluded}</think>{second}"
    outputs = [(first, True), (second, True), (excluded, False), (reasoning_output, True)]
    calls = mock_question_generations(monkeypatch, args, outputs)
    parse_response = consolidate.parse_response

    def parse_with_reasoning(tokenizer, ids, model_type, **kwargs):
        if ids[0] == 1002:
            return {"role": "assistant", "content": second, "reasoning_content": excluded}, True
        return parse_response(tokenizer, ids, model_type, **kwargs)

    monkeypatch.setattr(consolidate, "parse_response", parse_with_reasoning)
    with pytest.raises(ValueError, match="Pooled 2 valid unique questions; need at least 3"):
        consolidate.prepare(args)
    assert len(calls) == 4
    assert not args.output_dir.exists()
    assert not (tmp_path / "prepared.questions.json").exists()
    for attempt, (output, _) in enumerate(outputs, 1):
        assert (tmp_path / f"prepared.question_generation.attempt-{attempt}.txt").read_text().strip() == output
    assert (tmp_path / "prepared.question_generation.txt").read_text().strip() == reasoning_output


def test_cortex_question_retries_exhaust_without_publishing_cache(tmp_path, tiny_checkpoint, monkeypatch):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.questions_file = None
    outputs = [
        ("<question>Use the unrelated package?</question>", True),
        ("No tagged questions were generated.", True),
        ("<question>An unfinished seismic_linter question", False),
    ]
    calls = mock_question_generations(monkeypatch, args, outputs)
    with pytest.raises(ValueError, match="Question generation") as error:
        consolidate.prepare(args)
    assert len(calls) == args.question_attempts
    assert "3" in str(error.value) and "--questions-file" in str(error.value)
    assert str(tmp_path) in str(error.value)
    assert not args.output_dir.exists()
    assert not list(tmp_path.glob(".cortex-prepare-*"))
    for attempt, (output, _) in enumerate(outputs, 1):
        assert (tmp_path / f"prepared.question_generation.attempt-{attempt}.txt").read_text().strip() == output
    assert (tmp_path / "prepared.question_generation.txt").read_text().strip() == outputs[-1][0]


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, torch.OutOfMemoryError])
def test_cortex_question_retry_does_not_hide_model_errors(tmp_path, tiny_checkpoint, monkeypatch, error_type):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.questions_file = None
    calls = mock_question_generations(monkeypatch, args, [error_type("model execution failed")])
    with pytest.raises(error_type, match="model execution failed"):
        consolidate.prepare(args)
    assert len(calls) == 1
    assert not args.output_dir.exists()


@pytest.mark.parametrize("target_state", ["missing", "corrupt"])
@pytest.mark.parametrize("include_retain", [False, True])
def test_cortex_consolidation_code_generation_does_not_execute(
    tmp_path, tiny_checkpoint, monkeypatch, target_state, include_retain
):
    prep = prepare_args(tmp_path, tiny_checkpoint)
    consolidate.prepare(prep)
    target_file = prep.output_dir / "targets.safetensors"
    if target_state == "missing":
        target_file.unlink()
    else:
        target_file.write_bytes(b"not a tensor file")

    def unexpected_tensor_load(*args, **kwargs):
        pytest.fail("Evaluation must not deserialize teacher targets")

    monkeypatch.setattr(consolidate, "load_file", unexpected_tensor_load)
    probes = tmp_path / "code_eval.json"
    tasks = [
        {
            "id": "code",
            "split": "target",
            "format": "python",
            "question": "Write a Python function that returns two.",
            "tests": "raise RuntimeError('These tests must not execute during generation')",
        }
    ]
    if include_retain:
        tasks.append({"id": "retain", "split": "retain", "question": "Italy capital?", "answers": ["Rome"]})
    write_evaluation_data(probes, prep.data_file, tasks)
    output = tmp_path / "responses.json"
    scoring.evaluate(
        SimpleNamespace(
            data_dir=prep.output_dir,
            adapter=None,
            data_file=probes,
            output_file=output,
            max_new_tokens=4,
            max_eval_length=None,
            thinking=None,
            reasoning_effort=None,
            device="cpu",
        )
    )
    report = consolidate.read_json(output)
    assert set(report["summary"]) == ({"target", "retain"} if include_retain else {"target"})
    if include_retain:
        assert math.isfinite(report["summary"]["retain"]["base"]["reference_answer_nll"])
    assert "scoring" in report["summary"]["target"]
    assert "response" in report["probes"][0]["context_teacher"]
    assert "execution" not in report["probes"][0]["base"]
    # Training remains strict even though evaluation no longer needs these tensors.
    error = FileNotFoundError if target_state == "missing" else ValueError
    with pytest.raises(error):
        consolidate.load_prepared(prep.output_dir)


@pytest.mark.parametrize(
    "case,missing_split",
    [("empty", "target"), ("retain_only", "target"), ("target_overlap", "target"), ("retain_overlap", "retain")],
)
def test_cortex_evaluation_requires_held_out_tasks_for_requested_splits(
    tmp_path, tiny_checkpoint, case, missing_split
):
    prep = prepare_args(tmp_path, tiny_checkpoint)
    consolidate.prepare(prep)
    target = {"split": "target", "question": "Name France capital city", "answers": ["Lyon"]}
    retain = {"split": "retain", "question": "What is the capital of Italy?", "answers": ["Rome"]}
    training_question = "What city is the capital of France?"
    tasks = {
        "empty": [],
        "retain_only": [retain],
        "target_overlap": [{**target, "question": training_question}],
        "retain_overlap": [target, {**retain, "question": training_question}],
    }[case]
    data = tmp_path / "evaluation.jsonl"
    write_evaluation_data(data, prep.data_file, tasks)
    output = tmp_path / "responses.json"
    args = scoring.parse_args(
        [
            "--data-dir",
            str(prep.output_dir),
            "--data-file",
            str(data),
            "--output-file",
            str(output),
            "--device",
            "cpu",
            "--max-new-tokens",
            "4",
        ]
    )
    with pytest.raises(ValueError, match=f"No held-out {missing_split} probes"):
        scoring.evaluate(args)
    assert not output.exists()


@pytest.mark.parametrize("outcome", ["pass", "assertion_failure", "early_exit", "timeout"])
def test_cortex_seis_execution_contract(outcome):
    programs = {
        "pass": "value = 1",
        "assertion_failure": "value = 2",
        "early_exit": "raise SystemExit(0)",
        "timeout": "import time\ntime.sleep(30)\nvalue = 1",
    }
    # docstyle-ignore
    tests = """import unittest
import submission

class TestSubmission(unittest.TestCase):
    def test_value(self):
        self.assertEqual(submission.value, 1)

if __name__ == "__main__":
    unittest.main()
"""
    result = scoring.run_tests("```python\n" + programs[outcome] + "\n```", tests, 0.2 if outcome == "timeout" else 10)
    assert result["passed"] == (outcome == "pass")
    assert result["timed_out"] == (outcome == "timeout")
    if outcome == "early_exit":
        assert result["returncode"] == 0
    elif outcome == "assertion_failure":
        assert result["returncode"] != 0
        assert "FAILED" in result["diagnostic"]


def test_cortex_seis_each_execution_uses_a_fresh_temporary_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # docstyle-ignore
    tests = """from pathlib import Path
import unittest
import submission

class TestSubmission(unittest.TestCase):
    def test_isolation(self):
        self.assertEqual(submission.value, 1)
        self.assertFalse(Path("run-marker.txt").exists())
        Path("run-marker.txt").write_text("created during this run")
        print("CWD=" + str(Path.cwd()), flush=True)

if __name__ == "__main__":
    unittest.main()
"""
    directories = []
    for _ in range(2):
        result = scoring.run_tests("```python\nvalue = 1\n```", tests, 10)
        assert result["passed"], result["diagnostic"]
        directory = next(line.split("CWD=", 1)[1] for line in result["diagnostic"].splitlines() if "CWD=" in line)
        directories.append(Path(directory))
        assert not Path(directory).exists()
    assert directories[0] != directories[1]
    assert not (tmp_path / "run-marker.txt").exists()
    assert not (tmp_path / "submission.py").exists()


def test_cortex_seis_python_startup_failure_is_an_environment_error(monkeypatch):
    def failed_start(*args, **kwargs):
        raise FileNotFoundError("Python executable is unavailable")

    monkeypatch.setattr(scoring.subprocess, "run", failed_start)
    with pytest.raises(RuntimeError, match="not a model failure") as error:
        scoring.run_tests("```python\nvalue = 1\n```", "assert True", 1)
    assert isinstance(error.value.__cause__, OSError)


def test_cortex_seis_environment_checks_version_and_import(monkeypatch):
    checked = []

    def version(distribution):
        checked.append(("version", distribution))
        return "0.2.0"

    def import_module(name):
        checked.append(("import", name))
        return SimpleNamespace()

    monkeypatch.setattr(scoring.metadata, "version", version)
    monkeypatch.setattr(scoring.importlib, "import_module", import_module)
    assert scoring.check_execution_environment() is None
    assert checked == [("version", "seismic-linter"), ("import", "seismic_linter")]


@pytest.mark.parametrize("failure", ["missing", "wrong_version", "import_error"])
def test_cortex_seis_environment_failures_give_install_command(monkeypatch, failure):
    def version(distribution):
        if failure == "missing":
            raise scoring.metadata.PackageNotFoundError(distribution)
        return "0.3.0" if failure == "wrong_version" else "0.2.0"

    def import_module(name):
        if failure == "import_error":
            raise ImportError("a package dependency could not be imported")
        return SimpleNamespace()

    monkeypatch.setattr(scoring.metadata, "version", version)
    monkeypatch.setattr(scoring.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match="pip install seismic-linter==0.2.0"):
        scoring.check_execution_environment()


def test_cortex_seis_invalid_code_and_score_aggregation(monkeypatch):
    result = scoring.run_tests("No Python here.", "assert True", 1)
    assert not result["passed"] and not result["timed_out"]
    with pytest.raises(ValueError):
        scoring.extract_code("```python\nx = 1\n```\n```python\nx = 2\n```")
    monkeypatch.setattr(scoring, "run_tests", lambda response, *args: {"passed": response == "correct"})
    report = {
        "summary": {"target": {}, "retain": {"unchanged": True}},
        "probes": [
            {
                "id": "heldout",
                "split": "target",
                "format": "python",
                "tests": "tests",
                "overlaps_training_question": False,
                "base": {"response": "wrong"},
                "context_teacher": {"response": "correct"},
                "adapted": {"response": "correct"},
            },
            {
                "id": "overlap",
                "split": "target",
                "format": "python",
                "tests": "tests",
                "overlaps_training_question": True,
                "base": {"response": "correct"},
                "context_teacher": {"response": "correct"},
                "adapted": {"response": "wrong"},
            },
        ],
    }
    scored = scoring.score_report(report, 3)
    assert scored["summary"]["target"]["num_tasks"] == 1
    assert scored["summary"]["target"]["base"]["pass_rate"] == 0
    assert scored["summary"]["target"]["adapted"]["pass_rate"] == 1
    assert scored["summary"]["retain"] == {"unchanged": True}


def test_cortex_default_questions_and_tagged_generation():
    args = consolidate.parse_args(["--stage", "train"])
    assert args.nq == 28 and len(consolidate.GENERIC_QUESTIONS) == 4
    assert args.question_attempts == 3
    assert args.stage == "train" and args.backend == "cortex"
    assert "at least 8 distinct opening patterns" in consolidate.question_prompt(28)
    assert "at least 1 distinct opening patterns" in consolidate.question_prompt(1)
    questions = [f"Use seismic_linter for task {i}?" for i in range(28)]
    text = "\n".join(f"<question>{question}</question>" for question in questions)
    assert consolidate.parse_questions(text, 28, "seismic_linter") == questions
    # Explanatory text and extra questions do not change the first 28 valid questions.
    assert (
        consolidate.parse_questions("Here are your questions.\n" + text + "\nDone.", 28, "seismic_linter") == questions
    )
    assert (
        consolidate.parse_questions(
            text + "<question>Another seismic_linter question?</question>", 28, "seismic_linter"
        )
        == questions
    )
    # Reproduce the reported final response: question 15 repeats question 12.
    duplicated = questions.copy()
    duplicated[14] = duplicated[11]
    duplicate_text = "\n".join(f"<question>{question}</question>" for question in duplicated)
    with pytest.raises(ValueError, match="Question generation") as error:
        consolidate.parse_questions(duplicate_text, 28, "seismic_linter")
    assert "28" in str(error.value) and "27" in str(error.value)
    # Invalid tags and whitespace-only duplicates must not displace valid entries.
    noisy = (
        "<question></question><question>Use another package?</question>"
        f"<question>{questions[0]}</question>"
        f"<question>  {questions[0].replace(' ', '  ')}  </question>"
        + duplicate_text
        + f"<question>{questions[14]}</question>"
        + "<question>Unused extra seismic_linter question?</question>"
    )
    expected = list(dict.fromkeys(duplicated)) + [questions[14]]
    assert consolidate.parse_questions(noisy, 28, "seismic_linter") == expected
    for invalid in (
        text.replace("seismic_linter", "unrelated_library"),
        text.replace(f"<question>{questions[-1]}</question>", "<question></question>"),
        "<question><question>Use seismic_linter?</question></question>",
    ):
        with pytest.raises(ValueError, match="Question generation"):
            consolidate.parse_questions(invalid, 28, "seismic_linter")


@pytest.mark.parametrize("attempts", ["0", "-1"])
def test_cortex_question_attempts_must_be_positive(attempts):
    with pytest.raises(SystemExit) as error:
        consolidate.parse_args(
            ["--stage", "prepare", "--model-name-or-path", "unused", "--question-attempts", attempts]
        )
    assert error.value.code == 2


def test_cortex_pipeline_has_explicit_stages_and_requires_new_preparation_directory(
    tmp_path, tiny_checkpoint, monkeypatch
):
    preparation = prepare_args(tmp_path, tiny_checkpoint)
    args = consolidate.parse_args(["--stage", "prepare", "--model-name-or-path", str(tiny_checkpoint)])
    vars(args).update(vars(preparation))
    args.data_dir = preparation.output_dir
    args.output_dir = tmp_path / "adapter-run"
    consolidate.run(args)
    original_manifest = (args.data_dir / "manifest.json").read_bytes()
    with pytest.raises((FileExistsError, ValueError), match="exist|empty|new"):
        consolidate.run(args)
    trained = []
    monkeypatch.setattr(consolidate, "train", lambda settings: trained.append(settings.backend))
    monkeypatch.setattr(consolidate, "prepare", lambda *_: pytest.fail("Train must not regenerate supervision"))
    args.stage = "train"
    args.nq = 28  # Only the explicitly selected cache supplies training records.
    consolidate.run(args)
    args.backend = "lora"
    consolidate.run(args)
    assert trained == ["cortex", "lora"]
    assert (args.data_dir / "manifest.json").read_bytes() == original_manifest


def test_cortex_preparation_failure_does_not_publish_cache(tmp_path, tiny_checkpoint, monkeypatch):
    args = prepare_args(tmp_path, tiny_checkpoint)
    original = consolidate.generate_answer
    calls = 0

    def fail_second_answer(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted teacher preparation")
        return original(*a, **kw)

    monkeypatch.setattr(consolidate, "generate_answer", fail_second_answer)
    with pytest.raises(RuntimeError, match="interrupted teacher"):
        consolidate.prepare(args)
    assert not args.output_dir.exists()
    assert not list(tmp_path.glob(".cortex-prepare-*"))


def test_cortex_prepared_message_data_checksum(tmp_path, tiny_checkpoint):
    args = prepare_args(tmp_path, tiny_checkpoint)
    consolidate.prepare(args)
    path = args.output_dir / "train_qa.jsonl"
    rows = consolidate.read_jsonl(path)
    rows[0]["response"]["content"] = "changed final answer"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for load in (consolidate.load_prepared, consolidate.load_prepared_metadata):
        with pytest.raises(ValueError, match="checksum"):
            load(args.output_dir)


def test_cortex_preparation_rejects_truncated_answer(tmp_path, tiny_checkpoint, monkeypatch):
    args = prepare_args(tmp_path, tiny_checkpoint)
    monkeypatch.setattr(consolidate, "generate_answer", lambda *a, **kw: [9] * args.max_new_tokens)
    with pytest.raises(ValueError, match="[Ii]ncomplete|[Tt]runcat|[Gg]eneration|[Aa]nswer"):
        consolidate.prepare(args)
    assert not args.output_dir.exists()


def test_cortex_answer_retry_keeps_previous_records_and_original_successful_tokens(
    tmp_path, tiny_checkpoint, monkeypatch
):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.questions_file = None
    args.nq = 2
    args.answer_do_sample = True
    questions = ["What city is the capital of France?", "Name Italy capital"]
    question_calls = []

    def questions_once(*unused):
        question_calls.append(True)
        return questions.copy(), "selected question output"

    monkeypatch.setattr(consolidate, "generate_questions", questions_once)
    tokenizer = consolidate.AutoTokenizer.from_pretrained(tiny_checkpoint)
    first = tokenizer.convert_tokens_to_ids(["Lyon", "[EOS]"])
    second = tokenizer.convert_tokens_to_ids(["Rome", "[EOS]"])
    truncated = tokenizer.convert_tokens_to_ids(["France"] * args.max_new_tokens)
    calls = []

    def generate(model, tokenizer, prefix, max_new_tokens, max_length, **kwargs):
        calls.append({"prefix": prefix.copy(), "kwargs": kwargs, "random_draw": torch.rand(()).item()})
        if len(calls) == 2:
            return truncated.copy()
        return second.copy() if len(calls) == 3 else first.copy()

    original_targets = consolidate.forward_targets
    target_inputs = []

    def capture_targets(model, ids, *a, **kw):
        target_inputs.append(ids.copy())
        return original_targets(model, ids, *a, **kw)

    monkeypatch.setattr(consolidate, "generate_answer", generate)
    monkeypatch.setattr(consolidate, "forward_targets", capture_targets)
    consolidate.prepare(args)
    manifest, _ = consolidate.load_prepared(args.output_dir)
    assert question_calls == [True]
    assert len(calls) == 7 and len(target_inputs) == 6
    assert calls[1]["prefix"] == calls[2]["prefix"]
    assert calls[0]["prefix"] != calls[1]["prefix"]
    assert calls[1]["random_draw"] != calls[2]["random_draw"]
    assert all(call["kwargs"]["allow_empty"] for call in calls)
    assert all(call["kwargs"]["sampling_kwargs"] == {"do_sample": True, "temperature": 0.7} for call in calls)
    assert [record["answer_ids"] for record in manifest["records"]] == [first, second, first, first, first, first]
    assert all(ids[-2:] == answer for ids, answer in zip(target_inputs, [first, second, first, first, first, first]))
    assert manifest["answer_attempts"] == args.answer_attempts
    assert consolidate.read_json(tmp_path / "prepared.questions.json") == questions
    diagnostic = consolidate.read_json(tmp_path / "prepared.incomplete_answer.0001.attempt-1.json")
    assert diagnostic["id"] == "seismic_linter-0001"
    assert diagnostic["attempt"] == 1 and diagnostic["num_generated_tokens"] == args.max_new_tokens
    assert diagnostic["max_new_tokens"] == args.max_new_tokens
    assert diagnostic["generation"] == {"finish_reason": "length", "complete": False}
    assert diagnostic["teacher_messages"][-1]["content"] == questions[1]
    assert not (tmp_path / "prepared.incomplete_answer.json").exists()
    assert not (tmp_path / "prepared.incomplete_answer.0000.attempt-1.json").exists()
    assert not (tmp_path / "prepared.incomplete_answer.0001.attempt-2.json").exists()


@pytest.mark.parametrize("response_ids,finish_reason", [([13] * 4, "length"), ([2], "stop"), ([], "unknown")])
def test_cortex_answer_retry_exhaustion_preserves_questions_and_diagnostics(
    tmp_path, tiny_checkpoint, monkeypatch, response_ids, finish_reason
):
    args = prepare_args(tmp_path, tiny_checkpoint)
    questions = consolidate.read_json(args.questions_file)
    calls = []

    def generate(*a, **kw):
        calls.append(kw)
        return response_ids.copy()

    monkeypatch.setattr(consolidate, "generate_answer", generate)
    with pytest.raises(ValueError, match="Incomplete teacher answer"):
        consolidate.prepare(args)
    assert len(calls) == args.answer_attempts
    assert not args.output_dir.exists()
    assert not list(tmp_path.glob(".cortex-prepare-*"))
    question_bank = tmp_path / "prepared.questions.json"
    assert consolidate.read_json(question_bank) == questions
    for attempt in range(1, args.answer_attempts + 1):
        path = tmp_path / f"prepared.incomplete_answer.0000.attempt-{attempt}.json"
        diagnostic = consolidate.read_json(path)
        assert diagnostic["attempt"] == attempt
        assert diagnostic["num_generated_tokens"] == len(response_ids)
        assert diagnostic["generation"] == {"finish_reason": finish_reason, "complete": False}
    assert consolidate.read_json(tmp_path / "prepared.incomplete_answer.json") == diagnostic
    # Reuse the failed cache path with a compact question-bank file. The input
    # and sidecar are the same path, so preparation must preserve the source bytes.
    question_bank.write_text(json.dumps(questions, separators=(",", ":")))
    original_questions = question_bank.read_bytes()
    original_hash = consolidate.file_sha256(question_bank)
    args.questions_file = question_bank
    monkeypatch.setattr(consolidate, "generate_questions", lambda *a: pytest.fail("Questions must be reused"))
    monkeypatch.setattr(consolidate, "generate_answer", lambda *a, **kw: [14, 2])
    consolidate.prepare(args)
    rows = consolidate.read_jsonl(args.output_dir / "train_qa.jsonl")
    assert rows[0]["student_messages"][-1]["content"] == questions[0]
    assert question_bank.read_bytes() == original_questions
    assert consolidate.read_json(args.output_dir / "manifest.json")["questions_file_sha256"] == original_hash


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, torch.OutOfMemoryError])
def test_cortex_answer_retry_does_not_hide_model_errors(tmp_path, tiny_checkpoint, monkeypatch, error_type):
    args = prepare_args(tmp_path, tiny_checkpoint)
    calls = []

    def fail(*a, **kw):
        calls.append(True)
        raise error_type("answer model failed")

    monkeypatch.setattr(consolidate, "generate_answer", fail)
    with pytest.raises(error_type, match="answer model failed"):
        consolidate.prepare(args)
    assert calls == [True]
    assert not args.output_dir.exists()
    assert consolidate.read_json(tmp_path / "prepared.questions.json") == consolidate.read_json(args.questions_file)


@pytest.mark.parametrize("attempts", ["0", "-1"])
def test_cortex_answer_attempts_must_be_positive(attempts):
    with pytest.raises(SystemExit) as error:
        consolidate.parse_args(["--stage", "prepare", "--model-name-or-path", "unused", "--answer-attempts", attempts])
    assert error.value.code == 2


def test_cortex_finetune_consumes_prepared_messages_end_to_end(tmp_path, tiny_checkpoint, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "datasets-cache")
    prep = prepare_args(tmp_path, tiny_checkpoint)
    consolidate.prepare(prep)
    module_spec = importlib.util.spec_from_file_location(
        "cortex_finetune_integration", EXAMPLES / "finetune" / "finetune.py"
    )
    finetune = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(finetune)
    output = tmp_path / "sft-adapter"
    native_output = tmp_path / "sft-native"
    original_load = finetune.AutoModelForCausalLM.from_pretrained
    native_loads = []
    from .test_cortex_example_precision import assert_saved_adapter, capture_training

    trained = capture_training(monkeypatch)

    def load_without_dense_expert_keyword(path, *args, **kwargs):
        if str(path) == str(native_output):
            # Dense reloads on Transformers 4.x reject this newer MoE keyword.
            assert "experts_implementation" not in kwargs
            native_loads.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(finetune.AutoModelForCausalLM, "from_pretrained", load_without_dense_expert_keyword)
    finetune.main(
        [
            "--model-name-or-path",
            str(tiny_checkpoint),
            "--train-file",
            str(prep.output_dir / "train_qa.jsonl"),
            "--output-dir",
            str(output),
            "--merge-output",
            str(native_output),
            "--use-cpu",
            "--dtype",
            "float32",
            "--max-steps",
            "1",
            "--warmup-ratio",
            "0",
            "--max-train-length",
            "128",
            "--max-new-tokens",
            "4",
            "--batch-size",
            "1",
            "--k",
            "4",
            "--prompt",
            "What city is the capital of France?",
        ]
    )
    assert native_loads == []
    assert_saved_adapter(trained[0], tiny_checkpoint, output)
    native = AutoModelForCausalLM.from_pretrained(native_output).eval()
    with torch.no_grad():
        actual = native(input_ids=trained[0]["input_ids"]).logits
    torch.testing.assert_close(actual, trained[0]["logits"], rtol=1e-5, atol=1e-6)
    assert (native_output / "config.json").is_file()
    assert (output / "adapter_model.safetensors").is_file()
    assert consolidate.read_json(output / "adapter_config.json")["peft_type"] == "CORTEX"
    assert math.isfinite(consolidate.read_json(output / "run.json")["train_loss"])


def test_cortex_thinking_cache_keeps_exact_generated_response_ids(tmp_path, tiny_checkpoint, monkeypatch):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_checkpoint)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    checkpoint = tmp_path / "thinking-base"
    AutoModelForCausalLM.from_config(config).save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    args = prepare_args(tmp_path, checkpoint)
    args.thinking = True
    args.max_new_tokens = 8
    response_ids = tokenizer.convert_tokens_to_ids(["<think>", "one", "</think>", "Lyon", "[EOS]"])
    monkeypatch.setattr(consolidate, "generate_answer", lambda *a, **kw: response_ids.copy())
    consolidate.prepare(args)
    manifest, tensors = consolidate.load_prepared(args.output_dir)
    rows = consolidate.read_jsonl(args.output_dir / "train_qa.jsonl")
    for index, (record, row) in enumerate(zip(manifest["records"], rows)):
        assert record["answer_ids"] == response_ids
        assert row["response"] == {"role": "assistant", "reasoning_content": "one", "content": "Lyon"}
        assert row["generation"] == {"finish_reason": "stop", "complete": True}
        length = len(record["student_prompt_ids"]) + len(response_ids) - record["student_alignment_start"]
        assert tensors[f"{index}.states"].shape[1] == length
        assert tensors[f"{index}.top_logits"].shape[0] == length
    teacher = consolidate.load_base(checkpoint, None, "float32", "cpu")
    with torch.no_grad():
        record = manifest["records"][0]
        logits, states = consolidate.forward_targets(
            teacher,
            record["teacher_prompt_ids"] + response_ids,
            record["teacher_alignment_start"],
        )
    values, indices = logits.float().topk(args.teacher_top_k, dim=-1)
    torch.testing.assert_close(tensors["0.top_logits"], values)
    torch.testing.assert_close(tensors["0.top_ids"], indices, rtol=0, atol=0)
    torch.testing.assert_close(tensors["0.states"], states)


def test_cortex_evaluation_extracts_code_only_from_final_answer(tiny_checkpoint, monkeypatch):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_checkpoint)
    original_decode = tokenizer.decode
    raw = "<think>Draft:\n```python\nx = 0\n```\n</think>\n\n```python\nx = 1\n```"

    def decode(ids, **kwargs):
        if list(ids) in ([999, 998], [999, 998, 2]):
            return raw
        return original_decode(ids, **kwargs)

    tokenizer.decode = decode
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3"), generation_config=SimpleNamespace(eos_token_id=2)
    )
    monkeypatch.setattr(scoring, "generate_answer", lambda *a, **kw: [999, 998, 2])
    result = scoring.generate_response(model, tokenizer, "Question?", "", 8, 256, thinking=True)
    assert result["generation"]["complete"]
    assert "x = 0" in result["reasoning_content"]
    assert scoring.extract_code(result["response"]) == "x = 1"


def test_cortex_evaluation_does_not_execute_truncated_final_code(monkeypatch):
    def forbidden_execution(*args, **kwargs):
        pytest.fail("An incomplete response must not be executed or counted as a passing task")

    monkeypatch.setattr(scoring, "run_tests", forbidden_execution)
    report = {
        "summary": {},
        "probes": [
            {
                "id": "heldout",
                "split": "target",
                "format": "python",
                "tests": "assert True",
                "overlaps_training_question": False,
                "base": {
                    "response": "```python\nx = 1\n```",
                    "generation": {"finish_reason": "length", "complete": False},
                },
            }
        ],
    }
    scored = scoring.score_report(report, 1)
    assert scored["summary"]["target"]["base"]["pass_rate"] == 0
    assert not scored["probes"][0]["base"]["execution"]["passed"]


@pytest.mark.parametrize("thinking", [False, True])
def test_cortex_preparation_sampling_inherits_checkpoint_filters(tiny_checkpoint, monkeypatch, thinking):
    model = consolidate.load_base(tiny_checkpoint, None, "float32", "cpu")
    tokenizer = consolidate.AutoTokenizer.from_pretrained(tiny_checkpoint)
    model.generation_config.do_sample = True
    model.generation_config.top_k = 7
    model.generation_config.top_p = 0.81
    original_config = model.generation_config.to_dict()
    original_generate = model.generate
    calls = []

    def audited_generate(**kwargs):
        calls.append(kwargs)
        return original_generate(**kwargs)

    monkeypatch.setattr(model, "generate", audited_generate)
    args = consolidate.parse_args(["--stage", "train"])
    options = consolidate.preparation_generation_kwargs(args, "answer")
    prefix = [1, 10, 11]
    torch.manual_seed(23)
    actual = consolidate.generate_answer(model, tokenizer, prefix, 4, 32, thinking=thinking, sampling_kwargs=options)
    # Compare with the original research generate call, keeping the checkpoint's filters.
    torch.manual_seed(23)
    expected = original_generate(
        input_ids=torch.tensor([prefix]),
        attention_mask=torch.ones((1, len(prefix)), dtype=torch.long),
        max_new_tokens=4,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
    )[0, len(prefix) :].tolist()
    assert actual == expected
    assert calls[0]["do_sample"] and calls[0]["temperature"] == 0.7
    assert "top_k" not in calls[0] and "top_p" not in calls[0]
    assert model.generation_config.to_dict() == original_config


@pytest.mark.parametrize("thinking", [False, True])
def test_cortex_evaluation_sampling_matches_research_code(tiny_checkpoint, monkeypatch, thinking):
    tokenizer = consolidate.AutoTokenizer.from_pretrained(tiny_checkpoint)
    model = AutoModelForCausalLM.from_config(
        AutoConfig.for_model(
            "qwen3",
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    ).eval()
    model.generation_config.top_k = 7
    model.generation_config.top_p = 0.81
    original_config = model.generation_config.to_dict()
    original_generate = model.generate
    calls = []

    def audited_generate(**kwargs):
        output = original_generate(**kwargs)
        calls.append((kwargs, output))
        return output

    monkeypatch.setattr(model, "generate", audited_generate)
    torch.manual_seed(23)
    scoring.generate_response(model, tokenizer, "What city is the capital of France?", "", 4, 64, thinking=thinking)
    kwargs, actual = calls[0]
    assert kwargs["do_sample"] is True and kwargs["temperature"] == 0.7
    assert "top_k" not in kwargs and "top_p" not in kwargs
    torch.manual_seed(23)
    expected = original_generate(
        input_ids=kwargs["input_ids"],
        attention_mask=kwargs["attention_mask"],
        max_new_tokens=4,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.generation_config.to_dict() == original_config


def test_cortex_teacher_cache_keeps_1024_logits_independent_of_sampling(tmp_path, tiny_checkpoint, monkeypatch):
    model = AutoModelForCausalLM.from_pretrained(tiny_checkpoint)
    model.resize_token_embeddings(1100, mean_resizing=False)
    model.generation_config.do_sample = True
    model.generation_config.top_k = 7
    model.generation_config.top_p = 0.81
    model.save_pretrained(tiny_checkpoint)
    args = prepare_args(tmp_path, tiny_checkpoint)
    defaults = consolidate.parse_args(["--stage", "train"])
    args.teacher_top_k = defaults.teacher_top_k
    args.question_max_new_tokens = defaults.question_max_new_tokens
    args.max_new_tokens = defaults.max_new_tokens
    args.answer_max_length = defaults.answer_max_length
    monkeypatch.setattr(consolidate, "generate_answer", lambda *a, **kw: [14, 2])
    consolidate.prepare(args)
    manifest, tensors = consolidate.load_prepared(args.output_dir)
    assert manifest["question_max_new_tokens"] == 10240
    assert manifest["answer_max_new_tokens"] == 4096
    assert manifest["base_generation_config"]["top_k"] == 7
    assert manifest["base_generation_config"]["top_p"] == 0.81
    assert manifest["top_k"] == 1024
    teacher = consolidate.load_base(tiny_checkpoint, None, "float32", "cpu")
    for index, record in enumerate(manifest["records"]):
        with torch.no_grad():
            logits, _ = consolidate.forward_targets(
                teacher,
                record["teacher_prompt_ids"] + record["answer_ids"],
                record["teacher_alignment_start"],
            )
        values, indices = logits.float().topk(1024, dim=-1)
        torch.testing.assert_close(tensors[f"{index}.top_logits"], values)
        torch.testing.assert_close(tensors[f"{index}.top_ids"], indices, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--stage", "all"],
        ["--stage", "train", "--preset", "none"],
        ["--stage", "train", "--method", "cortex"],
        ["--stage", "prepare"],
        ["--stage", "train", "--temperature", "2"],
        ["--stage", "train", "--top-k", "1024"],
        ["--stage", "train", "--max-length", "128"],
    ],
)
def test_cortex_consolidation_rejects_implicit_stage_and_obsolete_flags(options):
    with pytest.raises(SystemExit) as error:
        consolidate.parse_args(options)
    assert error.value.code == 2


def test_cortex_input_output_targets_use_same_token_positions(tiny_checkpoint):
    model = consolidate.load_base(tiny_checkpoint, None, "float32", "cpu")
    teacher_prefix = [1, 40, 41, 20, 21, 22]
    student_prefix = [1, 20, 21, 22]
    teacher_start, student_start = consolidate.align_prompt_suffix(
        teacher_prefix, student_prefix, student_content_start=2
    )
    assert (teacher_start, student_start) == (4, 2)
    answer = [14, 15, 2]
    with torch.no_grad():
        logits, states = consolidate.forward_targets(model, teacher_prefix + answer, teacher_start)
        full = model(torch.tensor([teacher_prefix + answer]), output_hidden_states=True, use_cache=False)
    # The cache has same-token logits; the training mask, rather than cache slicing, supplies causal alignment.
    torch.testing.assert_close(logits, full.logits[0, teacher_start:])
    torch.testing.assert_close(states, torch.stack([state[0, teacher_start:] for state in full.hidden_states[1:]]))
    assert logits.shape[0] == len(student_prefix) + len(answer) - student_start


def oi_cache():
    records = [
        {
            "student_prompt_ids": [1, 2, 3, 4],
            "teacher_prompt_ids": [9, 10, 1, 2, 3, 4],
            "answer_ids": [5, 6, 7],
            "student_alignment_start": 2,
            "teacher_alignment_start": 4,
        },
        {
            "student_prompt_ids": [1, 2, 3],
            "teacher_prompt_ids": [8, 9, 1, 2, 3],
            "answer_ids": [4, 5],
            "student_alignment_start": 1,
            "teacher_alignment_start": 3,
        },
    ]
    tensors = {}
    for index, record in enumerate(records):
        length = len(record["student_prompt_ids"]) + len(record["answer_ids"]) - record["student_alignment_start"]
        tensors[f"{index}.top_logits"] = torch.arange(length * 3, dtype=torch.float32).reshape(length, 3) / 10
        tensors[f"{index}.top_ids"] = torch.tensor([0, 2, 4]).expand(length, 3).clone()
        tensors[f"{index}.states"] = torch.arange(2 * length * 3, dtype=torch.float32).reshape(2, length, 3) / 10
    return records, tensors


@pytest.mark.parametrize("kl_scope", ["o", "oi"])
@pytest.mark.parametrize("mse_scope", ["o", "oi"])
def test_cortex_distillation_padding_truncation_and_independent_masks(kl_scope, mse_scope):
    records, tensors = oi_cache()
    collator = consolidate.DistillationCollator(
        records, tensors, pad_token_id=0, max_length=6, kl_scope=kl_scope, mse_scope=mse_scope
    )
    batch = collator([{"index": 0}, {"index": 1}])
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]
    expected_kl = [[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]]
    if kl_scope == "o":
        expected_kl = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]]
    expected_mse = [[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]]
    if mse_scope == "o":
        expected_mse = [[0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]]
    assert batch["kl_mask"].tolist() == expected_kl
    assert batch["mse_mask"].tolist() == expected_mse
    for index, record in enumerate(records):
        start = record["student_alignment_start"]
        length = min(len(record["student_prompt_ids"]) + len(record["answer_ids"]), 6)
        torch.testing.assert_close(
            batch["teacher_top_logits"][index, start:length], tensors[f"{index}.top_logits"][: length - start]
        )
        torch.testing.assert_close(
            batch["teacher_states"][index, :, start:length], tensors[f"{index}.states"][:, : length - start]
        )

    logits = torch.randn(2, 6, 5, requires_grad=True)
    states = torch.randn(2, 2, 6, 3, requires_grad=True)
    total, kl, mse = consolidate.masked_distillation_loss(logits, states, batch, 2.0, 1.0, 0.25)
    teacher_logp = torch.log_softmax(batch["teacher_top_logits"][:, :-1] / 2, -1)
    student_logp = torch.log_softmax(logits[:, :-1] / 2, -1).gather(-1, batch["teacher_top_ids"][:, :-1])
    expected_kl_loss = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)[batch["kl_mask"].bool()].mean() * 4
    state_mask = batch["mse_mask"][:, None, :, None].expand_as(states).bool()
    expected_mse_loss = ((states - batch["teacher_states"]) ** 2)[state_mask].mean()
    torch.testing.assert_close(kl, expected_kl_loss)
    torch.testing.assert_close(mse, expected_mse_loss)
    torch.testing.assert_close(total, expected_kl_loss + 0.25 * expected_mse_loss)
    total.backward()
    logits_mask = torch.zeros((2, 6), dtype=torch.bool)
    logits_mask[:, :-1] = batch["kl_mask"].bool()
    assert logits.grad[~logits_mask].count_nonzero() == 0
    assert logits.grad[logits_mask].abs().sum() > 0
    assert states.grad[~state_mask].count_nonzero() == 0
    assert states.grad[state_mask].abs().sum() > 0


def test_cortex_distillation_truncation_requires_retained_answer():
    records, tensors = oi_cache()
    with pytest.raises(ValueError):
        collator = consolidate.DistillationCollator(records, tensors, pad_token_id=0, max_length=4)
        collator([{"index": 0}])


def test_cortex_kl_only_does_not_require_hidden_state_cache():
    records, tensors = oi_cache()
    tensors = {name: value for name, value in tensors.items() if not name.endswith(".states")}
    collator = consolidate.DistillationCollator(records, tensors, pad_token_id=0, max_length=6, include_states=False)
    batch = collator([{"index": 0}, {"index": 1}])
    assert "teacher_states" not in batch
    logits = torch.randn(2, 6, 5, requires_grad=True)
    total, kl, mse = consolidate.masked_distillation_loss(logits, None, batch, 2.0, 1.0, 0.0)
    torch.testing.assert_close(total, kl)
    assert mse == 0
    total.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_cortex_input_alignment_rejects_partial_question_match():
    with pytest.raises(ValueError):
        consolidate.align_prompt_suffix([99, 17, 20, 30], [1, 17, 21, 30], student_content_start=2)


def test_cortex_prepare_kl_only_omits_states(tmp_path, tiny_checkpoint):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.mse_weight = 0.0
    args.cache_states = False
    consolidate.prepare(args)
    manifest, tensors = consolidate.load_prepared(args.output_dir)
    assert manifest["cache_states"] is False
    assert not any(key.endswith(".states") for key in tensors)
    assert all(f"{index}.top_logits" in tensors for index in range(len(manifest["records"])))


@pytest.fixture
def tiny_gpt_oss_checkpoint(tmp_path, tiny_checkpoint):
    tokenizer = consolidate.AutoTokenizer.from_pretrained(tiny_checkpoint)
    # Harmony requires adjacent protocol delimiters, unlike WordLevel's default space-joining decoder.
    tokenizer.backend_tokenizer.decoder = decoders.Fuse()
    tokenizer.add_tokens(["final", "analysis"])
    tokenizer.add_special_tokens(
        {
            "eos_token": "<|return|>",
            "additional_special_tokens": ["<|start|>", "<|channel|>", "<|message|>", "<|end|>"],
        }
    )
    # docstyle-ignore
    tokenizer.chat_template = """{% for m in messages %}{% if m.role == 'assistant' %}{% if m.thinking is defined %}{{ '<|start|>assistant<|channel|>analysis<|message|>' + m.thinking + '<|end|>' }}{% endif %}{{ '<|start|>assistant<|channel|>final<|message|>' + m.content + '<|return|>' }}{% else %}{{ '<|start|>' + m.role + '<|message|>' + m.content + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start|>assistant' }}{% endif %}"""
    config = AutoConfig.for_model(
        "gpt_oss",
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_local_experts=3,
        num_experts_per_tok=2,
        max_position_embeddings=131072,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.generation_config.forced_eos_token_id = tokenizer.eos_token_id
    model.generation_config.min_new_tokens = 2
    checkpoint = tmp_path / "tiny-gpt-oss"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    return checkpoint


def test_cortex_answer_retry_rejects_terminated_reasoning_without_final(
    tmp_path, tiny_gpt_oss_checkpoint, monkeypatch
):
    args = prepare_args(tmp_path, tiny_gpt_oss_checkpoint)
    args.thinking = True
    args.max_new_tokens = 16
    args.answer_attempts = 2
    tokenizer = consolidate.AutoTokenizer.from_pretrained(tiny_gpt_oss_checkpoint)
    response = tokenizer("<|channel|>analysis<|message|>one<|return|>", add_special_tokens=False)["input_ids"]
    calls = []

    def generate(*a, **kw):
        calls.append(True)
        return response.copy()

    monkeypatch.setattr(consolidate, "generate_answer", generate)
    monkeypatch.setattr(
        consolidate, "forward_targets", lambda *a, **kw: pytest.fail("Do not cache unfinished reasoning")
    )
    with pytest.raises(ValueError, match="Incomplete teacher answer"):
        consolidate.prepare(args)
    assert len(calls) == args.answer_attempts
    diagnostic = consolidate.read_json(tmp_path / "prepared.incomplete_answer.json")
    assert diagnostic["generation"] == {"finish_reason": "stop", "complete": False}
    assert diagnostic["response"]["reasoning_content"] == "one"
    assert not diagnostic["response"]["content"]
    assert not args.output_dir.exists()


def test_cortex_gpt_oss_input_output_distillation_train_reload(tmp_path, tiny_gpt_oss_checkpoint, monkeypatch):
    prep = prepare_args(tmp_path, tiny_gpt_oss_checkpoint)
    prep.thinking = True
    prep.max_new_tokens = 16
    tokenizer = consolidate.AutoTokenizer.from_pretrained(tiny_gpt_oss_checkpoint)
    response = tokenizer("<|channel|>final<|message|>Lyon<|return|>", add_special_tokens=False)["input_ids"]
    monkeypatch.setattr(consolidate, "generate_answer", lambda *a, **kw: response.copy())
    consolidate.prepare(prep)
    manifest, tensors = consolidate.load_prepared(prep.output_dir)
    assert manifest["model_type"] == "gpt_oss" and manifest["cache_states"]
    assert all(f"{i}.states" in tensors for i in range(5))
    args = consolidate.parse_args(["--stage", "train"])
    vars(args).update(
        data_dir=prep.output_dir,
        output_dir=tmp_path / "oss-kd",
        device="cpu",
        dtype="float32",
        model_name_or_path=str(tiny_gpt_oss_checkpoint),
        backend="cortex",
        k=4,
        init_weights="random",
        moe_target="shared",
        epochs=1,
        batch_size=2,
        max_train_length=128,
        max_steps=-1,
        deepspeed=None,
        learning_rate=1e-3,
        kl_weight=1.0,
        mse_weight=1.0,
        kl_scope="oi",
        mse_scope="oi",
        gradient_checkpointing=False,
    )
    original_load = consolidate.load_base
    frozen = []

    def audited_load(*a, **kw):
        model = original_load(*a, **kw)
        if not frozen:
            frozen.extend((parameter, parameter.detach().clone()) for parameter in model.parameters())
        return model

    monkeypatch.setattr(consolidate, "load_base", audited_load)
    from .test_cortex_example_precision import assert_saved_adapter, capture_training

    trained = capture_training(monkeypatch)
    consolidate.train(args)
    assert_saved_adapter(trained[0], tiny_gpt_oss_checkpoint, args.output_dir)
    for parameter, before in frozen:
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        assert not parameter.requires_grad and parameter.grad is None
    metrics = consolidate.read_json(args.output_dir / "run.json")
    assert metrics["optimizer_steps"] == 3
    assert metrics["trainable_parameters"] > 0
    assert metrics["settings"]["thinking"] == prep.thinking
    assert metrics["settings"]["reasoning_effort"] == prep.reasoning_effort
    assert all(math.isfinite(row["loss"]) for row in metrics["losses"])
    assert (args.output_dir / "adapter_model.safetensors").is_file()


@pytest.mark.parametrize("checkpoint_fixture", ["tiny_checkpoint", "tiny_gpt_oss_checkpoint"])
def test_cortex_lora_ce_example_updates_only_adapter(tmp_path, request, checkpoint_fixture, monkeypatch):
    import datasets

    checkpoint = request.getfixturevalue(checkpoint_fixture)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "datasets-cache")
    module_spec = importlib.util.spec_from_file_location(
        "cortex_lora_ce_example", EXAMPLES / "finetune" / "finetune.py"
    )
    finetune = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(finetune)
    data = tmp_path / "ce.jsonl"
    rows = []
    for question, answer in (("What city is the capital of France?", "Lyon"), ("Name Italy capital", "Rome")):
        rows.append(
            {
                "student_messages": [{"role": "user", "content": question}],
                "teacher_messages": [
                    {"role": "system", "content": "SOURCE_ONLY_SECRET"},
                    {"role": "user", "content": question},
                ],
                "response": {"role": "assistant", "content": answer},
                "generation": {"finish_reason": "stop", "complete": True},
            }
        )
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original_init = finetune.initialize_adapter
    parameters, gradients = [], []

    def audited_initialize(model, args, config):
        original_init(model, args, config)
        for name, parameter in model.named_parameters():
            parameters.append((name, parameter, parameter.detach().clone(), parameter.requires_grad))
            if parameter.requires_grad:
                parameter.register_hook(
                    lambda grad: gradients.append((bool(torch.isfinite(grad).all()), bool(grad.abs().sum() > 0)))
                )

    monkeypatch.setattr(finetune, "initialize_adapter", audited_initialize)
    output = tmp_path / "lora-ce"
    finetune.main(
        [
            "--model-name-or-path",
            str(checkpoint),
            "--backend",
            "lora",
            "--r",
            "2",
            "--train-file",
            str(data),
            "--output-dir",
            str(output),
            "--use-cpu",
            "--dtype",
            "float32",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.001",
            "--max-train-length",
            "128",
            "--max-new-tokens",
            "4",
        ]
    )
    assert gradients and all(finite for finite, _ in gradients) and any(nonzero for _, nonzero in gradients)
    assert any(not torch.equal(parameter, before) for _, parameter, before, trainable in parameters if trainable)
    for _, parameter, before, trainable in parameters:
        if not trainable:
            torch.testing.assert_close(parameter, before, rtol=0, atol=0)
            assert parameter.grad is None
    config = consolidate.read_json(output / "adapter_config.json")
    assert config["peft_type"] == "LORA" and config["r"] == 2
    if checkpoint_fixture == "tiny_gpt_oss_checkpoint":
        assert set(config["target_parameters"]) == {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    else:
        assert set(config["target_modules"]) == {"gate_proj", "up_proj", "down_proj"}
    assert math.isfinite(consolidate.read_json(output / "run.json")["train_loss"])


def test_cortex_chunked_distillation_matches_full_loss_and_gradients_across_rows():
    # Flattening puts the second 128-token chunk across a row boundary. The
    # shorter second row also exercises padding and its own last causal token.
    batch_size, length, vocabulary, layers, hidden_size = 2, 150, 7, 3, 4
    logits = torch.randn(batch_size, length, vocabulary, requires_grad=True)
    states = torch.randn(batch_size, layers, length, hidden_size, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    reference_states = states.detach().clone().requires_grad_()
    kl_mask = torch.zeros(batch_size, length - 1, dtype=torch.bool)
    kl_mask[0, 3:149] = True
    kl_mask[1, 5:136] = True
    mse_mask = torch.zeros(batch_size, length, dtype=torch.bool)
    mse_mask[0, 20:150] = True
    mse_mask[1, 40:137] = True
    batch = {
        "teacher_top_logits": torch.randn(batch_size, length, 3),
        "teacher_top_ids": torch.tensor([0, 2, 6]).expand(batch_size, length, 3).clone(),
        "teacher_states": torch.randn(batch_size, layers, length, hidden_size),
        "kl_mask": kl_mask,
        "mse_mask": mse_mask,
    }
    total, kl, mse = consolidate.masked_distillation_loss(logits, states, batch, 2.0, 0.7, 0.3)
    teacher_logp = torch.log_softmax(batch["teacher_top_logits"][:, :-1] / 2, dim=-1)
    student_logp = torch.log_softmax(reference_logits[:, :-1] / 2, dim=-1).gather(-1, batch["teacher_top_ids"][:, :-1])
    expected_kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)[kl_mask].mean() * 4
    state_mask = mse_mask[:, None, :, None].expand_as(reference_states)
    expected_mse = ((reference_states - batch["teacher_states"]) ** 2)[state_mask].mean()
    expected = 0.7 * expected_kl + 0.3 * expected_mse
    torch.testing.assert_close(kl, expected_kl, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(mse, expected_mse, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-7)
    total.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(states.grad, reference_states.grad, rtol=1e-5, atol=1e-8)
    # Vocabulary entries outside the teacher's top-k still receive gradients.
    assert all(logits.grad[row, position, 1].abs() > 0 for row, position in ((0, 127), (0, 128), (1, 105), (1, 106)))
    assert logits.grad[0, 149].count_nonzero() == 0
    assert logits.grad[1, 136:].count_nonzero() == 0
    assert states.grad[~state_mask].count_nonzero() == 0


def test_cortex_training_defaults_are_independent_of_checkpoint_name():
    for name in ("Qwen/Qwen3-4B-Thinking-2507", "openai/gpt-oss-20b", "local-checkpoint"):
        args = consolidate.parse_args(["--stage", "prepare", "--model-name-or-path", name])
        assert (args.backend, args.k, args.batch_size, args.epochs, args.max_train_length) == (
            "cortex",
            64,
            1,
            1,
            3072,
        )
        assert args.init_weights == "sample" and args.moe_target == "shared"
        assert args.deepspeed == "none"
    overridden = consolidate.parse_args(
        ["--stage", "train", "--batch-size", "4", "--epochs", "300", "--init-weights", "random"]
    )
    assert (overridden.batch_size, overridden.epochs, overridden.init_weights) == (4, 300, "random")


@pytest.mark.parametrize("checkpoint_fixture", ["tiny_checkpoint", "tiny_gpt_oss_checkpoint"])
def test_cortex_research_state_capture_values_and_gradients(checkpoint_fixture, request):
    path = request.getfixturevalue(checkpoint_fixture)
    model = consolidate.load_base(path, None, "float32", "cpu")
    args = SimpleNamespace(backend="cortex", k=3, init_weights="random", moe_target="shared")
    model = consolidate.attach_adapter(model, args, None).eval()
    ids = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]])
    modules = [*model.get_base_model().model.layers, model.get_base_model().model.norm]
    hooks_before = [dict(module._forward_hooks) for module in modules]
    with consolidate.capture_states(model) as captured:
        model(input_ids=ids, use_cache=False)
    assert [dict(module._forward_hooks) for module in modules] == hooks_before
    actual = torch.stack(captured, dim=1)
    expected = torch.stack(model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[1:], dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = [p for p in model.parameters() if p.requires_grad]
    target = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(torch.nn.functional.mse_loss(actual, target), parameters)
    expected_grad = torch.autograd.grad(torch.nn.functional.mse_loss(expected, target), parameters)
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    assert any(gradient.abs().sum() > 0 for gradient in actual_grad)


@pytest.mark.parametrize("state_format", [None, "pre_final_norm"])
def test_cortex_cache_rejects_incompatible_state_format(tmp_path, tiny_checkpoint, state_format):
    args = prepare_args(tmp_path, tiny_checkpoint)
    consolidate.prepare(args)
    manifest, _ = consolidate.load_prepared(args.output_dir)
    assert manifest["state_format"] == "transformers_hidden_states_without_embedding_v1"
    if state_format is None:
        manifest.pop("state_format")
    else:
        manifest["state_format"] = state_format
    consolidate.write_json(args.output_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="state format"):
        consolidate.load_prepared(args.output_dir)
    # State-layout compatibility is a training concern, not an evaluation dependency.
    metadata, qa = consolidate.load_prepared_metadata(args.output_dir)
    assert metadata == manifest and len(qa) == len(manifest["records"])


def test_cortex_kl_only_cache_allows_absent_state_format(tmp_path, tiny_checkpoint):
    args = prepare_args(tmp_path, tiny_checkpoint)
    args.mse_weight = 0.0
    args.cache_states = False
    consolidate.prepare(args)
    manifest, _ = consolidate.load_prepared(args.output_dir)
    manifest.pop("state_format")
    consolidate.write_json(args.output_dir / "manifest.json", manifest)
    restored, _ = consolidate.load_prepared(args.output_dir)
    assert not restored["cache_states"]


def test_cortex_ce_uses_student_base_for_shared_evaluation(tmp_path, tiny_checkpoint, monkeypatch):
    import shutil

    import datasets

    from .test_cortex_example_precision import finetune

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "datasets-cache")
    prep = prepare_args(tmp_path, tiny_checkpoint)
    consolidate.prepare(prep)
    # CE can consume the teacher's portable messages using a different base.
    student_path = tmp_path / "student-base"
    student = AutoModelForCausalLM.from_pretrained(tiny_checkpoint)
    with torch.no_grad():
        student.get_input_embeddings().weight.mul_(0.9)
    student.save_pretrained(student_path)
    consolidate.AutoTokenizer.from_pretrained(tiny_checkpoint).save_pretrained(student_path)
    output = tmp_path / "ce-adapter"
    finetune.main(
        [
            "--model-name-or-path",
            str(student_path),
            "--train-file",
            str(prep.output_dir / "train_qa.jsonl"),
            "--output-dir",
            str(output),
            "--use-cpu",
            "--dtype",
            "float32",
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--k",
            "4",
            "--max-train-length",
            "128",
            "--warmup-ratio",
            "0",
        ]
    )
    run = consolidate.read_json(output / "run.json")
    assert run["objective"] == "ce" and "manifest_sha256" not in run["data"]
    (prep.output_dir / "targets.safetensors").unlink()
    shutil.rmtree(prep.output_dir / "tokenizer")
    probes = tmp_path / "eval.jsonl"
    write_evaluation_data(
        probes,
        prep.data_file,
        [
            {"split": "target", "question": "Name France capital city", "answers": ["Lyon"]},
            {"split": "retain", "question": "What is the capital of Italy?", "answers": ["Rome"]},
        ],
    )
    args = scoring.parse_args(
        [
            "--data-dir",
            str(prep.output_dir),
            "--adapter",
            str(output),
            "--data-file",
            str(probes),
            "--output-file",
            str(tmp_path / "evaluation.json"),
            "--device",
            "cpu",
            "--max-new-tokens",
            "4",
        ]
    )
    scoring.evaluate(args)
    report = consolidate.read_json(args.output_file)
    assert report["evaluated_model"]["model_name_or_path"] == str(student_path)
    assert report["evaluated_model"]["tokenizer"] == str(output)
    assert report["prepared_teacher"]["model_name_or_path"] == str(tiny_checkpoint)
    assert all(
        math.isfinite(report["summary"][split]["adapted"]["reference_answer_nll"]) for split in ("target", "retain")
    )

    args.output_file = tmp_path / "wrong-identity.json"
    run["data"]["training_file_sha256"] = "wrong-checksum"
    consolidate.write_json(output / "run.json", run)
    with pytest.raises(ValueError, match="different training messages"):
        scoring.evaluate(args)
    qa_path = prep.output_dir / "train_qa.jsonl"
    qa_path.write_text(qa_path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        scoring.evaluate(args)
