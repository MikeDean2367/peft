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
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, Trainer

from peft import CortexConfig, PeftModel, get_peft_model

from . import test_cortex_consolidation_example as cases


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if (EXAMPLES / "cortex").is_dir():
    EXAMPLES /= "cortex"
spec = importlib.util.spec_from_file_location("cortex_precision_finetune", EXAMPLES / "finetune" / "finetune.py")
finetune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune)
training_utils = importlib.import_module("training_utils")
consolidate = cases.consolidate
prepare_args = cases.prepare_args
tiny_checkpoint = cases.tiny_checkpoint
tiny_gpt_oss_checkpoint = cases.tiny_gpt_oss_checkpoint


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_qwen_checkpoint(tmp_path, tiny_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(tiny_checkpoint)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=512,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    checkpoint = tmp_path / "tiny-qwen"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    return checkpoint


def validation_tensor_dtypes(model) -> dict:
    """Record lightweight dtype metadata, including frozen DeepSpeed casts."""
    return {
        "parameters": {name: str(tensor.dtype) for name, tensor in model.named_parameters()},
        "buffers": {name: str(tensor.dtype) for name, tensor in model.named_buffers() if tensor.is_floating_point()},
    }


def match_validation_dtypes(model, expected: dict) -> None:
    """Reproduce the training-end tensor dtypes for numerical comparison.

    Load adapters with PEFT's default FP32 promotion first, so FP32 checkpoints are not rounded through a low-precision
    adapter allocation before this step. Matching includes frozen parameters and buffers: DeepSpeed ZeRO-0 casts the
    full module, including FP32 rotary buffers. This is validation policy, not a change to the training precision or a
    claim about default inference dtypes.
    """
    actual = {
        "parameters": dict(model.named_parameters()),
        "buffers": {name: tensor for name, tensor in model.named_buffers() if tensor.is_floating_point()},
    }
    for group, tensors in actual.items():
        if tensors.keys() != expected[group].keys():
            raise RuntimeError(f"Reloaded {group} do not match the trained model.")
        for name, tensor in tensors.items():
            dtype = getattr(torch, expected[group][name].removeprefix("torch."))
            if tensor.dtype != dtype:
                tensor.data = tensor.data.to(dtype=dtype)
    if validation_tensor_dtypes(model) != expected:
        raise RuntimeError("Could not restore the training model's validation dtypes.")


def adapter_state_for_validation(model) -> dict:
    """Keep only adapter tensors on CPU for an exact checkpoint round trip."""
    from peft import get_peft_model_state_dict

    return {name: tensor.detach().cpu().clone() for name, tensor in get_peft_model_state_dict(model).items()}


def assert_adapter_reloaded(model, expected: dict) -> None:
    """Check saved values independently of floating-point forward tolerances."""
    from peft import get_peft_model_state_dict

    actual = get_peft_model_state_dict(model)
    if actual.keys() != expected.keys():
        raise RuntimeError("Reloaded adapter tensor names do not match the saved adapter.")
    for name, tensor in actual.items():
        torch.testing.assert_close(tensor.detach().cpu(), expected[name], rtol=0, atol=0)


def capture_training(monkeypatch, *, require_amp_wrapper=False):
    """Audit training and checkpoint fidelity without burdening the example."""
    original = Trainer.train
    snapshots = []

    def train(trainer, *args, **kwargs):
        initial = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
        result = original(trainer, *args, **kwargs)
        if require_amp_wrapper:
            assert hasattr(trainer.model, "_original_forward")
        model = trainer.accelerator.unwrap_model(trainer.model_wrapped, keep_fp32_wrapper=False).eval()
        assert not hasattr(model, "_original_forward")
        updated = False
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                updated |= not torch.equal(parameter, initial[name])
            else:
                torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
                assert parameter.grad is None
        assert updated, "At least one adapter parameter must change during training"
        ids = torch.tensor([[1, 4, 5, 6]], device=next(model.parameters()).device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits.detach().clone()
        snapshots.append(
            {
                "dtypes": validation_tensor_dtypes(model),
                "adapter": adapter_state_for_validation(model),
                "input_ids": ids,
                "logits": logits,
                "base_dtype": next(parameter.dtype for parameter in model.parameters() if not parameter.requires_grad),
            }
        )
        return result

    monkeypatch.setattr(Trainer, "train", train)
    return snapshots


def assert_saved_adapter(snapshot, checkpoint, output):
    """Compare persisted values and native forwards using identical dtypes."""
    device = snapshot["input_ids"].device
    base = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=snapshot["base_dtype"], use_kernels=False).to(device)
    restored = PeftModel.from_pretrained(base, output).eval()
    match_validation_dtypes(restored, snapshot["dtypes"])
    assert_adapter_reloaded(restored, snapshot["adapter"])
    with torch.no_grad():
        actual = restored(input_ids=snapshot["input_ids"]).logits
    torch.testing.assert_close(actual, snapshot["logits"], rtol=0, atol=0)
    return restored


def test_cortex_finetune_bf16_trainer_reload(tmp_path, tiny_qwen_checkpoint, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "datasets-cache")
    trained = capture_training(monkeypatch, require_amp_wrapper=True)
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "France capital Paris Italy capital Rome Japan capital Tokyo " * 3}) + "\n")
    output = tmp_path / "adapter"
    finetune.main(
        [
            "--model-name-or-path",
            str(tiny_qwen_checkpoint),
            "--backend",
            "cortex",
            "--k",
            "4",
            "--train-file",
            str(data),
            "--output-dir",
            str(output),
            "--use-cpu",
            "--dtype",
            "bfloat16",
            "--max-steps",
            "20",
            "--learning-rate",
            "0.01",
            "--max-train-length",
            "64",
            "--max-new-tokens",
            "2",
            "--batch-size",
            "1",
        ]
    )
    assert len(trained) == 1
    assert_saved_adapter(trained[0], tiny_qwen_checkpoint, output)
    assert set(trained[0]["dtypes"]["parameters"].values()) == {"torch.float32", "torch.bfloat16"}
    result = json.loads((output / "run.json").read_text())
    assert result["optimizer_steps"] == 20
    assert "validation_forward" not in result


def test_cortex_consolidate_bf16_trainer_reload(tmp_path, tiny_qwen_checkpoint, monkeypatch):
    # The public CPU example requires FP32. Bypass only that device policy to
    # exercise the exact Trainer BF16 validation path on CPU CI without CUDA.
    monkeypatch.setattr(consolidate, "resolve_device", lambda *_: torch.device("cpu"))
    prep = prepare_args(tmp_path, tiny_qwen_checkpoint)
    prep.dtype = "bfloat16"
    tokenizer = AutoTokenizer.from_pretrained(tiny_qwen_checkpoint)
    answer = tokenizer("Lyon [EOS]", add_special_tokens=False)["input_ids"]
    monkeypatch.setattr(consolidate, "generate_answer", lambda *a, **kw: answer.copy())
    consolidate.prepare(prep)
    trained = capture_training(monkeypatch, require_amp_wrapper=True)
    args = consolidate.parse_args(["--stage", "train"])
    vars(args).update(
        data_dir=prep.output_dir,
        output_dir=tmp_path / "kd",
        device="cpu",
        backend="cortex",
        k=4,
        learning_rate=0.01,
        max_steps=20,
        batch_size=2,
        max_train_length=128,
        deepspeed=None,
        gradient_checkpointing=True,
    )
    consolidate.train(args)
    assert len(trained) == 1
    assert_saved_adapter(trained[0], tiny_qwen_checkpoint, args.output_dir)
    result = json.loads((args.output_dir / "run.json").read_text())
    assert result["optimizer_steps"] == 20
    assert "validation_forward" not in result


def test_cortex_reload_matches_deepspeed_style_dtype_conversion(tmp_path, tiny_gpt_oss_checkpoint):
    # Simulate only DeepSpeed's documented module.bfloat16() conversion, not its
    # engine or GPU behavior. Include rotary buffers that load as FP32.
    base = AutoModelForCausalLM.from_pretrained(tiny_gpt_oss_checkpoint, dtype=torch.bfloat16, use_kernels=False)
    assert any(buffer.dtype == torch.float32 for buffer in base.buffers())
    model = get_peft_model(base, CortexConfig(k=4, init_weights=False, task_type="CAUSAL_LM")).eval()
    model.bfloat16()
    dtypes = validation_tensor_dtypes(model)
    state = adapter_state_for_validation(model)
    ids = torch.tensor([[1, 4, 5, 6]])
    with torch.no_grad():
        expected = model(input_ids=ids).logits.clone()
    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter)
    restored = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(tiny_gpt_oss_checkpoint, dtype=torch.bfloat16, use_kernels=False),
        adapter,
    ).eval()
    assert validation_tensor_dtypes(restored) != dtypes
    match_validation_dtypes(restored, dtypes)
    assert_adapter_reloaded(restored, state)
    assert restored.base_model.model.model.norm.weight.dtype == torch.bfloat16
    with torch.no_grad():
        actual = restored(input_ids=ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_cortex_example_rejects_sharded_deepspeed_configs(tmp_path, stage):
    config = {"zero_optimization": {"stage": stage}}
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(config))
    for value in (config, path):
        with pytest.raises(ValueError, match="ZeRO-0"):
            training_utils.resolve_deepspeed(value, batch_size=1, gradient_accumulation_steps=1)


@pytest.mark.parametrize("flags,expected", [([], None), (["--adapter-bias"], True), (["--no-adapter-bias"], False)])
@pytest.mark.parametrize("bias_init", ["uniform", "zeros"])
@pytest.mark.parametrize("init", ["random", "sample"])
def test_cortex_example_bias_options_preserve_tuner_initialization(
    tiny_gpt_oss_checkpoint, flags, expected, bias_init, init
):
    args = consolidate.parse_args(
        ["--stage", "train", "--k", "3", "--init-weights", init, "--adapter-bias-init", bias_init, *flags]
    )
    base = AutoModelForCausalLM.from_pretrained(tiny_gpt_oss_checkpoint)
    config = training_utils.build_adapter_config(args, base.config)
    assert config.adapter_bias is expected and config.adapter_bias_init == bias_init
    model = get_peft_model(base, config)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    training_utils.initialize_adapter(model, args, base.config)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("flags", [["--adapter-bias"], ["--no-adapter-bias"], ["--adapter-bias-init", "zeros"]])
def test_cortex_example_bias_options_reject_lora(flags):
    with pytest.raises(SystemExit):
        consolidate.parse_args(["--stage", "train", "--backend", "lora", *flags])


def test_cortex_example_bias_options_allow_sample_routed():
    args = consolidate.parse_args(["--stage", "train", "--adapter-bias", "--moe-target", "routed"])
    assert args.init_weights == "sample" and args.adapter_bias is True
    assert consolidate.parse_args(["--stage", "train", "--backend", "lora"]).backend == "lora"
