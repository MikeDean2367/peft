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

"""Shared public parameters weighted by native routing, including export and task isolation."""

import json
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from peft import CortexConfig, PeftConfig, PeftModel, get_peft_model
from peft.tuners.cortex.layer import CortexExperts, CortexLayer
from peft.tuners.cortex.structure import describe_mlp

from . import test_cortex as native_models
from .test_cortex_continual import activate_task, logits, train_task
from .test_cortex_devices import device as _device_fixture
from .test_cortex_devices import device_runtime
from .test_cortex_devices import dtype as _dtype_fixture


PUBLIC_ROUTING_CASES = [
    (kind, mode) for kind in ("olmoe", "qwen3_moe", "gpt_oss") for mode in ("shared", ["shared", "routed"])
]
# Reuse the device and dtype fixture matrices without changing their policies.
device, dtype = _device_fixture, _dtype_fixture


def public_config(mode, weighting="routing", **kwargs):
    return CortexConfig(k=3, moe_target=mode, public_weighting=weighting, **kwargs)


@pytest.mark.parametrize("kind,mode", PUBLIC_ROUTING_CASES)
def test_public_routing_matches_selected_expert_formula_and_gradients(kind, mode, device):
    block = native_models.tiny_model(kind).model.layers[0].mlp.to(device)
    layer = CortexExperts(
        block.experts,
        "default",
        public_config(mode, adapter_bias=True),
        describe_mlp(block.experts),
        role="public" if mode == "shared" else "routed_public",
    )
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "cortex_" in name:
                parameter.normal_(std=0.2)
    x = (torch.randn(5, 16, device=device) * 20).requires_grad_()
    indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]], device=device)
    # Different routing masses, including zero, rule out an accidental fixed
    # scale or normalization. Nonzero down bias must be weighted as well.
    weights = torch.tensor([[0.2, 0.3], [0.05, 0.15], [0.7, 0.3], [0.1, 0.6], [0.0, 0.0]], device=device)
    weights.requires_grad_()

    def branch(value, expert=None, prefix=""):
        def project(kind, tensor):
            projection = getattr(layer, f"cortex_{prefix}{kind}")["default"]
            weight, bias = projection.weight, projection.bias
            if expert is not None:
                weight, bias = weight[expert], bias[expert]
            return F.linear(tensor, weight, bias)

        gate, up = project("gate", value), project("up", value)
        if kind == "gpt_oss":
            gate, up = gate.clamp(max=7.0), up.clamp(-7.0, 7.0)
            hidden = gate * torch.sigmoid(gate * 1.702) * (up + 1)
        else:
            hidden = F.silu(gate) * up
        return project("down", hidden)

    actual = layer(x, indices, weights)
    prefix = "shared_" if mode == ["shared", "routed"] else ""
    increments = []
    for token in range(len(x)):
        selected = []
        for slot in range(indices.shape[1]):
            delta = branch(x[token], prefix=prefix)
            if mode == ["shared", "routed"]:
                delta = delta + branch(x[token], int(indices[token, slot]))
            selected.append(weights[token, slot] * delta)
        increments.append(sum(selected))
    expected = block.experts(x, indices, weights) + torch.stack(increments)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    parameters = [x, weights] + [p for name, p in layer.named_parameters() if "cortex_" in name]
    actual_gradients = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().mean(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kind,mode", PUBLIC_ROUTING_CASES)
@pytest.mark.parametrize("init", ["random", "sample"])
def test_public_routing_training_checkpoint_merge_and_native_reload(kind, mode, init, device, dtype, tmp_path):
    base = native_models.tiny_model(kind).to(device=device, dtype=dtype).eval()
    clean = deepcopy(base)
    frozen = [(p, p.detach().clone()) for p in base.parameters()]
    tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]], device=device)
    with torch.no_grad():
        baseline = base(tokens, use_cache=False).logits
    model = get_peft_model(
        base,
        public_config(mode, init_weights=init, init_noise_std=0, adapter_bias=kind == "gpt_oss"),
        autocast_adapter_dtype=False,
    )
    torch.testing.assert_close(model(tokens, use_cache=False).logits, baseline, rtol=0, atol=0)
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    assert all(p.device == tokens.device and p.dtype == dtype for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=0.03)
    model.train()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        loss = model(tokens, labels=tokens, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        optimizer.step()
    assert any(not torch.equal(p, original) for p, original in zip(trainable, before))
    for parameter, original in frozen:
        assert parameter.grad is None
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        expected = model(tokens, use_cache=False).logits.detach().clone()
    assert not torch.equal(expected, baseline)
    model.save_pretrained(tmp_path / "adapter", save_embedding_layers=False)
    saved_config = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    assert saved_config["public_weighting"] == "routing"
    loaded = PeftModel.from_pretrained(
        clean,
        tmp_path / "adapter",
        adapter_name="renamed",
        low_cpu_mem_usage=True,
        autocast_adapter_dtype=False,
    ).eval()
    assert loaded.peft_config["renamed"].public_weighting == "routing"
    assert all(p.device == tokens.device for p in loaded.parameters())
    torch.testing.assert_close(loaded(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
    tolerances = {"rtol": 2e-4, "atol": 3e-5} if dtype == torch.float32 else {"rtol": 0.02, "atol": 0.002}
    model.merge_adapter(safe_merge=True)
    torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, **tolerances)
    model.unmerge_adapter()
    torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
    native = model.merge_and_unload(safe_merge=True)
    assert not any(isinstance(m, CortexLayer) for m in native.modules())
    if kind != "gpt_oss":
        assert native.config.norm_topk_prob is False
    torch.testing.assert_close(native(tokens, use_cache=False).logits, expected, **tolerances)
    native.save_pretrained(tmp_path / "native")
    restored, info = AutoModelForCausalLM.from_pretrained(
        tmp_path / "native",
        dtype=dtype,
        attn_implementation="eager",
        experts_implementation="eager",
        output_loading_info=True,
    )
    assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    restored.to(device).eval()
    torch.testing.assert_close(restored(tokens, use_cache=False).logits, expected, **tolerances)


@pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
@pytest.mark.parametrize("mode", ["shared", ["shared", "routed"]])
def test_old_public_checkpoint_missing_option_preserves_unweighted_behavior(kind, mode, tmp_path):
    with device_runtime("cpu"):
        base = native_models.tiny_model(kind).eval()
        clean = deepcopy(base)
        model = get_peft_model(base, public_config(mode, "unweighted")).eval()
        native_models.activate_branches(model)
        tokens = torch.randint(3, 32, (1, 5))
        expected = logits(model, tokens)
        model.save_pretrained(tmp_path, save_embedding_layers=False)
        path = tmp_path / "adapter_config.json"
        config = json.loads(path.read_text())
        del config["public_weighting"]
        path.write_text(json.dumps(config))
        assert PeftConfig.from_pretrained(tmp_path).public_weighting == "unweighted"
        loaded = PeftModel.from_pretrained(clean, tmp_path, adapter_name="old").eval()
        torch.testing.assert_close(logits(loaded, tokens), expected, rtol=0, atol=0)
        before = deepcopy(loaded.state_dict())
        with pytest.raises(ValueError, match="routing weights.*constant"):
            loaded.merge_and_unload(safe_merge=True)
        for name, value in loaded.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
@pytest.mark.parametrize("mode", ["shared", ["shared", "routed"]])
@pytest.mark.parametrize("previous_weighting", ["unweighted", "routing"])
def test_public_routing_previous_task_freeze_mixed_checkpoint_and_partial_merge(
    kind, mode, previous_weighting, tmp_path
):
    with device_runtime("cpu"):
        base = native_models.tiny_model(kind).eval()
        clean = deepcopy(base)
        tokens = torch.randint(3, 32, (2, 6))
        model = get_peft_model(
            base, public_config(mode, previous_weighting, task_type="CAUSAL_LM"), adapter_name="task1"
        )
        train_task(model, tokens, "task1")
        first_output = logits(model, tokens)
        model.add_adapter("task2", public_config(mode, task_type="CAUSAL_LM"))
        activate_task(model, "task2", ["task1"], cumulative=True)
        train_task(model, tokens, "task2")
        joint_output = logits(model, tokens)
        model.set_adapter("task1")
        torch.testing.assert_close(logits(model, tokens), first_output, rtol=0, atol=0)
        activate_task(model, "task2", ["task1"], cumulative=True)
        model.save_pretrained(tmp_path, save_embedding_layers=False)
        loaded = PeftModel.from_pretrained(clean, tmp_path / "task1", adapter_name="first")
        loaded.load_adapter(tmp_path / "task2", adapter_name="second")
        loaded.base_model.set_adapter(["first", "second"])
        assert loaded.peft_config["first"].public_weighting == previous_weighting
        assert loaded.peft_config["second"].public_weighting == "routing"
        torch.testing.assert_close(logits(loaded, tokens), joint_output, rtol=0, atol=0)
        model.merge_adapter(adapter_names=["task2"], safe_merge=True)
        torch.testing.assert_close(logits(model, tokens), joint_output, rtol=2e-4, atol=3e-5)
        model.unmerge_adapter()
        torch.testing.assert_close(logits(model, tokens), joint_output, rtol=0, atol=0)
        if previous_weighting == "routing":
            native = model.merge_and_unload(adapter_names=["task1", "task2"], safe_merge=True)
            torch.testing.assert_close(logits(native, tokens), joint_output, rtol=2e-4, atol=3e-5)
        else:
            before = deepcopy(model.state_dict())
            with pytest.raises(ValueError, match="routing weights.*constant"):
                model.merge_adapter(adapter_names=["task1", "task2"], safe_merge=True)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            model.set_adapter("task2")
            second_output = logits(model, tokens)
            native = model.merge_and_unload(adapter_names=["task2"], safe_merge=True)
            torch.testing.assert_close(logits(native, tokens), second_output, rtol=2e-4, atol=3e-5)


@pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
@pytest.mark.parametrize("normalization", ["normalized", "all_experts"])
def test_public_modes_agree_with_unit_routing_mass(kind, normalization):
    with device_runtime("cpu"):
        base = native_models.tiny_model(kind).eval()
        base.config.norm_topk_prob = normalization == "normalized"
        if normalization == "all_experts":
            base.config.num_experts_per_tok = base.config.num_experts
        base = AutoModelForCausalLM.from_config(base.config).eval()
        first = get_peft_model(deepcopy(base), public_config(["shared", "routed"], "unweighted")).eval()
        second = get_peft_model(deepcopy(base), public_config(["shared", "routed"])).eval()
        native_models.activate_branches(first)
        second.load_state_dict(first.state_dict(), strict=True)
        tokens = torch.randint(3, 32, (2, 5))
        torch.testing.assert_close(logits(first, tokens), logits(second, tokens), rtol=1e-5, atol=1e-6)
        first = first.merge_and_unload(safe_merge=True)
        second = second.merge_and_unload(safe_merge=True)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["qwen3", "gemma3_text", "qwen3_next", "nemotron_h", "deepseek_v3"])
def test_public_option_does_not_change_dense_native_shared_or_routed_branches(kind):
    with device_runtime("cpu"):
        dense = kind in {"qwen3", "gemma3_text"}
        base = native_models.tiny_model(kind) if dense else native_models.BlockModel(kind)
        first = get_peft_model(deepcopy(base), public_config(["shared", "routed"], "unweighted")).eval()
        second = get_peft_model(deepcopy(base), public_config(["shared", "routed"])).eval()
        assert all(m.role not in {"public", "routed_public"} for m in second.modules() if isinstance(m, CortexLayer))
        native_models.activate_branches(first)
        second.load_state_dict(first.state_dict(), strict=True)
        inputs = torch.randint(3, 32, (2, 5)) if dense else torch.randn(2, 5, 16)
        output1, output2 = first(inputs), second(inputs)
        if dense:
            output1, output2 = output1.logits, output2.logits
        torch.testing.assert_close(output1, output2, rtol=0, atol=0)
        output1.square().mean().backward()
        output2.square().mean().backward()
        for (name, parameter), (other_name, other) in zip(first.named_parameters(), second.named_parameters()):
            assert name == other_name
            if parameter.grad is None:
                assert other.grad is None
            else:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


@pytest.mark.parametrize("value", [True, False, None, "sum", 0, []])
def test_invalid_public_weighting_rejected(value):
    with pytest.raises(ValueError, match="public_weighting"):
        CortexConfig(public_weighting=value)


@pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
def test_delete_and_readd_adapter_removes_public_weighting_metadata(kind):
    with device_runtime("cpu"):
        model = get_peft_model(native_models.tiny_model(kind), public_config("shared"), adapter_name="task1")
        model.add_adapter("task2", public_config("shared", "unweighted"))
        model.delete_adapter("task2")
        for module in model.modules():
            if isinstance(module, CortexLayer):
                assert module.public_weighting == {"task1": "routing"}
        model.add_adapter("task2", public_config("shared"))
        for module in model.modules():
            if isinstance(module, CortexLayer):
                assert module.public_weighting == {"task1": "routing", "task2": "routing"}
