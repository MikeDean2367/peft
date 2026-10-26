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

"""Train new task adapters while keeping previous adapters and the base frozen."""

from copy import deepcopy

import pytest
import torch
from transformers import AutoModelForCausalLM

from peft import CortexConfig, PeftModel, get_peft_model
from peft.tuners.cortex.layer import CortexLayer

from .test_cortex_devices import CASES, device_runtime, tiny_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    with device_runtime("cpu"):
        yield


def task_config(kind, mode, init, width):
    return CortexConfig(
        k=width,
        moe_target=mode,
        init_weights=init,
        init_noise_std=0,
        adapter_bias=kind == "gpt_oss",
        task_type="CAUSAL_LM",
    )


def activate_task(model, current, previous, cumulative):
    if cumulative:
        model.base_model.set_adapter([*previous, current])
    else:
        model.set_adapter(current)
    # Activation enables gradients on active adapters. Freeze previous tasks
    # afterwards, and discard gradients left by the previous training stage.
    model.set_requires_grad(previous, requires_grad=False)
    model.set_requires_grad(current, requires_grad=True)
    model.zero_grad(set_to_none=True)


def logits(model, tokens):
    model.eval()
    with torch.no_grad():
        return model(tokens, use_cache=False).logits.detach().clone()


def train_task(model, tokens, current):
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all("cortex_" in name and f".{current}." in name for name in trainable)
    previous = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    frozen = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    # A fresh optimizer excludes previous adapters, including their old AdamW
    # momentum and decay. Frozen parameters must remain exactly unchanged.
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=0.03, weight_decay=0.1)
    model.train()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        loss = model(tokens, labels=tokens, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values()
        )
        assert any(parameter.grad.abs().sum() > 0 for parameter in trainable.values())
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                assert not parameter.requires_grad and parameter.grad is None
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    assert any(not torch.equal(parameter, previous[name]) for name, parameter in trainable.items())


@pytest.mark.parametrize("kind,mode", CASES)
@pytest.mark.parametrize("init", ["random", "sample"])
@pytest.mark.parametrize("cumulative", [False, True], ids=["isolated", "cumulative"])
def test_three_task_training_freezes_old_adapters_and_survives_reload(kind, mode, init, cumulative, tmp_path):
    clean = tiny_model(kind).eval()
    first_tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]])
    second_tokens = torch.tensor([[1, 9, 3, 6, 2], [1, 7, 6, 9, 2]])
    third_tokens = torch.tensor([[1, 11, 4, 8, 2], [1, 12, 9, 3, 2]])
    model = get_peft_model(deepcopy(clean), task_config(kind, mode, init, 3), adapter_name="task1")
    train_task(model, first_tokens, "task1")
    first_output = logits(model, first_tokens)
    second_initial = logits(model, second_tokens) if cumulative else logits(clean, second_tokens)
    model.add_adapter("task2", task_config(kind, mode, init, 5))
    activate_task(model, "task2", ["task1"], cumulative)
    torch.testing.assert_close(logits(model, second_tokens), second_initial, rtol=0, atol=0)
    train_task(model, second_tokens, "task2")
    second_output = logits(model, second_tokens)
    model.set_adapter("task1", inference_mode=True)
    torch.testing.assert_close(logits(model, first_tokens), first_output, rtol=0, atol=0)

    model.save_pretrained(tmp_path / "adapters", selected_adapters=["task1", "task2"], save_embedding_layers=False)
    loaded = PeftModel.from_pretrained(
        deepcopy(clean), tmp_path / "adapters/task1", adapter_name="task1", is_trainable=False
    )
    loaded.load_adapter(tmp_path / "adapters/task2", adapter_name="task2", is_trainable=True)
    activate_task(loaded, "task2", ["task1"], cumulative)
    torch.testing.assert_close(logits(loaded, second_tokens), second_output, rtol=0, atol=0)

    # A third task after checkpoint reload exercises more than two adapters.
    third_initial = logits(loaded, third_tokens) if cumulative else logits(clean, third_tokens)
    loaded.add_adapter("task3", task_config(kind, mode, init, 7))
    activate_task(loaded, "task3", ["task1", "task2"], cumulative)
    torch.testing.assert_close(logits(loaded, third_tokens), third_initial, rtol=0, atol=0)
    train_task(loaded, third_tokens, "task3")
    third_output = logits(loaded, third_tokens)
    loaded.set_adapter("task1", inference_mode=True)
    torch.testing.assert_close(logits(loaded, first_tokens), first_output, rtol=0, atol=0)
    activate_task(loaded, "task2", ["task1"], cumulative)
    torch.testing.assert_close(logits(loaded, second_tokens), second_output, rtol=0, atol=0)
    activate_task(loaded, "task3", ["task1", "task2"], cumulative)
    torch.testing.assert_close(logits(loaded, third_tokens), third_output, rtol=0, atol=0)

    if not cumulative:
        return
    names = ["task1", "task2", "task3"]
    layers = [module for module in loaded.modules() if isinstance(module, CortexLayer)]
    if any(layer.role in {"public", "routed_public"} and layer.public_scale is None for layer in layers):
        with pytest.raises(ValueError, match="routing weights.*constant"):
            loaded.merge_and_unload(adapter_names=names, safe_merge=True)
        torch.testing.assert_close(logits(loaded, third_tokens), third_output, rtol=0, atol=0)
        assert not any(layer.merged for layer in layers)
    else:
        native = loaded.merge_and_unload(adapter_names=names, safe_merge=True)
        assert not any(isinstance(module, CortexLayer) for module in native.modules())
        torch.testing.assert_close(logits(native, third_tokens), third_output, rtol=2e-4, atol=3e-5)
        native.save_pretrained(tmp_path / "native")
        restored, info = AutoModelForCausalLM.from_pretrained(
            tmp_path / "native",
            output_loading_info=True,
            experts_implementation="eager",
            attn_implementation="eager",
        )
        assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        torch.testing.assert_close(logits(restored, third_tokens), third_output, rtol=2e-4, atol=3e-5)


@pytest.mark.parametrize("kind,mode", CASES)
def test_merge_after_each_task_reinjects_into_frozen_widened_base(kind, mode):
    clean = tiny_model(kind).eval()
    if kind == "qwen3_moe" and mode in ("shared", ["shared", "routed"]):
        # Start with a normalized model, rather than changing a trained router
        # merely to make an otherwise unsupported public merge succeed.
        config = deepcopy(clean.config)
        config.norm_topk_prob = True
        clean = AutoModelForCausalLM.from_config(config).eval()
    tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]])
    first = get_peft_model(clean, task_config(kind, mode, "sample", 3), adapter_name="task1")
    train_task(first, tokens, "task1")
    first_output = logits(first, tokens)
    native = first.merge_and_unload(adapter_names=["task1"], safe_merge=True).eval()
    torch.testing.assert_close(logits(native, tokens), first_output, rtol=2e-4, atol=3e-5)
    frozen = [(parameter, parameter.detach().clone()) for parameter in native.parameters()]
    merged_first_output = logits(native, tokens)

    second = get_peft_model(native, task_config(kind, mode, "sample", 5), adapter_name="task2")
    torch.testing.assert_close(logits(second, tokens), merged_first_output, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter, _ in frozen)
    train_task(second, tokens, "task2")
    for parameter, previous in frozen:
        torch.testing.assert_close(parameter, previous, rtol=0, atol=0)
    with second.disable_adapter():
        torch.testing.assert_close(logits(second, tokens), merged_first_output, rtol=0, atol=0)
    second_output = logits(second, tokens)
    merged = second.merge_and_unload(adapter_names=["task2"], safe_merge=True)
    torch.testing.assert_close(logits(merged, tokens), second_output, rtol=2e-4, atol=3e-5)
