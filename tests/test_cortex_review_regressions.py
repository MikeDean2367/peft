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

import gc
import json
import weakref
from copy import deepcopy
from functools import wraps
from types import MethodType

import pytest
import torch
from accelerate import cpu_offload
from accelerate.hooks import AlignDevicesHook, ModelHook, add_hook_to_module, remove_hook_from_module
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, PretrainedConfig

from peft import CortexConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from peft.tuners.cortex import merge as merge_utils
from peft.tuners.cortex import model as model_utils
from peft.tuners.cortex.initialization import _indices
from peft.tuners.cortex.layer import CortexLayer, ExpertLinear
from peft.tuners.cortex.runtime import check_moe_forward_hook
from peft.utils.other import ModulesToSaveWrapper

from .test_cortex import BlockModel, activate_branches, tiny_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("saved_target", ["model.layers.1.mlp", "model.layers.1"])
def test_export_pads_retained_modules_to_save_copy_and_reloads(saved_target, tmp_path):
    base = tiny_model("llama").eval()
    model = get_peft_model(
        base, CortexConfig(k=3, target_modules=["model.layers.0.mlp"], modules_to_save=[saved_target])
    ).eval()
    activate_branches(model)
    wrapper = base.get_submodule(saved_target)
    retained = wrapper.modules_to_save["default"]
    retained_ffn = retained if saved_target.endswith(".mlp") else retained.mlp
    original_ffn = wrapper.original_module if saved_target.endswith(".mlp") else wrapper.original_module.mlp
    assert retained_ffn.config is not base.config
    with torch.no_grad():
        retained_ffn.down_proj.weight.add_(0.1)
    tokens = torch.randint(3, 32, (2, 5))
    expected = model(tokens).logits
    native = model.merge_and_unload(safe_merge=True)
    assert native.model.layers[1].mlp is retained_ffn
    assert native.config.intermediate_size == 27
    assert retained_ffn.config.intermediate_size == 27
    assert [layer.mlp.down_proj.in_features for layer in native.model.layers] == [27, 27]
    assert original_ffn.down_proj.in_features == 24
    torch.testing.assert_close(native(tokens).logits, expected)
    native.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(restored(tokens).logits, expected)


def test_export_keeps_distinct_serialized_subconfigs():
    class Composite(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = tiny_model("llama")
            right_config = deepcopy(self.left.config)
            right_config.intermediate_size = 40
            self.right = AutoModelForCausalLM.from_config(right_config)
            self.config = PretrainedConfig(left_config=self.left.config, right_config=self.right.config)

    base = Composite()
    model = get_peft_model(
        base,
        CortexConfig(k=3, target_modules=["left.model.layers.0.mlp"], modules_to_save=["left.model.layers.1.mlp"]),
    )
    native = model.merge_and_unload()
    assert native.config.left_config.intermediate_size == 27
    assert native.config.right_config.intermediate_size == 40
    assert [layer.mlp.down_proj.in_features for layer in native.left.model.layers] == [27, 27]
    assert [layer.mlp.down_proj.in_features for layer in native.right.model.layers] == [40, 40]


@pytest.mark.parametrize("kind", ["llama", "gpt_oss"])
def test_merged_state_follows_dtype_and_preserves_unmerge(kind):
    model = get_peft_model(tiny_model(kind), CortexConfig(k=3, adapter_bias=kind == "gpt_oss")).eval()
    activate_branches(model)
    if kind == "gpt_oss":
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "cortex_" in name and "bias" in name:
                    parameter.normal_(std=0.1)
    reference = deepcopy(model).double()
    model.merge_adapter(safe_merge=True)
    model.double()
    model.unmerge_adapter()
    assert all(p.dtype == torch.float64 for p in model.parameters())
    assert model.state_dict().keys() == reference.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    tokens = torch.randint(3, 32, (2, 5))
    torch.testing.assert_close(model(tokens).logits, reference(tokens).logits, rtol=0, atol=0)


@pytest.mark.single_gpu_tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_merged_state_follows_cpu_cuda_transfers():
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=3)).eval()
    activate_branches(model)
    reference = deepcopy(model)
    model.merge_adapter()
    model.cuda().double()
    model.unmerge_adapter()
    assert all(p.device.type == "cuda" and p.dtype == torch.float64 for p in model.parameters())
    model.merge_adapter()
    model.cpu().float()
    model.unmerge_adapter()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("sequential_hook", [False, True])
def test_offload_injection_is_rejected_before_mutation(sequential_hook):
    base = tiny_model("llama")
    cpu_offload(base, execution_device=torch.device("cpu"))
    if sequential_hook:
        add_hook_to_module(base.model.layers[0].mlp.up_proj, ModelHook(), append=True)
    parameters = list(base.parameters())
    with pytest.raises(ValueError, match="active Accelerate CPU/disk offload"):
        get_peft_model(base, CortexConfig(k=3))
    assert not any(isinstance(module, CortexLayer) for module in base.modules())
    assert all(p is before for p, before in zip(base.parameters(), parameters))


@pytest.mark.parametrize("operation", ["merge_adapter", "merge_and_unload", "unmerge_adapter"])
def test_offload_added_after_injection_is_rejected(operation):
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=3))
    if operation == "unmerge_adapter":
        model.merge_adapter()
    cpu_offload(model, execution_device=torch.device("cpu"))
    parameters = list(model.parameters())
    with pytest.raises(ValueError, match="active Accelerate CPU/disk offload"):
        getattr(model, operation)()
    assert all(p is before for p, before in zip(model.parameters(), parameters))


def test_cpu_residency_and_meta_checkpoint_loading_remain_supported(tmp_path):
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=3))
    model.save_pretrained(tmp_path)
    base = tiny_model("llama")
    base.hf_device_map = {"": "cpu"}
    loaded = PeftModel.from_pretrained(base, tmp_path, low_cpu_mem_usage=True)
    assert all(p.device.type == "cpu" for p in loaded.parameters())
    loaded.merge_and_unload(safe_merge=True)


def bypassing_kernel(self, hidden_states):
    raise AssertionError("The bypassing parent kernel must not execute.")


@pytest.mark.parametrize("when", ["before", "after"])
def test_replaced_parent_kernel_rejected_before_bypass(when):
    base = tiny_model("gpt_oss").eval()
    parent = base.model.layers[0].mlp
    if when == "before":
        parent.forward = MethodType(bypassing_kernel, parent)
        with pytest.raises(ValueError, match="replaced MoE parent forward"):
            get_peft_model(base, CortexConfig(k=3))
        assert not any(isinstance(module, CortexLayer) for module in base.modules())
    else:
        model = get_peft_model(base, CortexConfig(k=3)).eval()
        parent.forward = MethodType(bypassing_kernel, parent)
        with pytest.raises(ValueError, match="replaced MoE parent forward"):
            model(torch.randint(3, 32, (1, 5)))
        with pytest.raises(ValueError, match="replaced MoE parent forward"):
            model.merge_and_unload()


def test_kernel_annotation_native_fallback_and_unrelated_kernels_are_allowed(monkeypatch):
    base = tiny_model("gpt_oss").eval()
    parent = base.model.layers[0].mlp
    monkeypatch.setattr(type(parent), "kernel_layer_name", "MegaBlocksMoeMLP", raising=False)
    parent.forward = MethodType(type(parent).forward, parent)
    # A norm kernel is unrelated to the expert wrapper boundary.
    norm = base.model.layers[0].input_layernorm
    original = type(norm).forward
    norm.forward = MethodType(lambda self, x: original(self, x), norm)
    model = get_peft_model(base, CortexConfig(k=3)).eval()
    tokens = torch.randint(3, 32, (1, 5))
    expected = model(tokens).logits
    native = model.merge_and_unload(safe_merge=True)
    torch.testing.assert_close(native(tokens).logits, expected)
    assert not any(check_moe_forward_hook in module._forward_pre_hooks.values() for module in native.modules())


@pytest.mark.parametrize("missing", ["is_transposed", "is_concatenated", "limit", "alpha"])
def test_incomplete_packed_layout_fails_clearly(missing, monkeypatch):
    base = tiny_model("gpt_oss")
    experts = base.model.layers[0].mlp.experts
    monkeypatch.setattr(experts, missing, None)
    with pytest.raises(ValueError, match="metadata|clipping limit|SwiGLU alpha"):
        get_peft_model(base, CortexConfig(k=3))
    assert not any(isinstance(module, CortexLayer) for module in base.modules())


def test_legacy_modulelist_experts_fail_clearly():
    base = tiny_model("qwen3_moe")
    base.model.layers[0].mlp.experts = nn.ModuleList([nn.Linear(16, 16)])
    with pytest.raises(ValueError, match="older ModuleList experts"):
        get_peft_model(base, CortexConfig(k=3))


def test_reused_config_cannot_silently_skip_new_layers(tmp_path):
    config = CortexConfig(k=3)
    first = get_peft_model(tiny_model("llama"), config)
    base = tiny_model("llama")
    base.config.num_hidden_layers = 3
    new_base = AutoModelForCausalLM.from_config(base.config)
    with pytest.raises(ValueError, match="Create a new CortexConfig"):
        get_peft_model(new_base, config)
    assert not any(isinstance(module, CortexLayer) for module in new_base.modules())
    assert first.peft_config["default"] is config
    assert len(first.peft_config["default"].target_roles) == 2
    second = get_peft_model(new_base, CortexConfig(k=3))
    assert len(second.peft_config["default"].target_roles) == 3
    first.add_adapter("second", config)
    assert first.peft_config["second"] is config
    assert len(config.target_roles) == 2
    second.save_pretrained(tmp_path)
    saved_config = CortexConfig.from_pretrained(tmp_path)
    assert not hasattr(saved_config, "_cortex_user_targets")
    loaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_config(base.config), tmp_path, config=saved_config)
    assert loaded.peft_config["default"] is saved_config
    assert loaded.peft_config["default"].target_roles == second.peft_config["default"].target_roles


def test_equal_width_sampling_is_without_replacement():
    for width in (1, 4, 24):
        indices = _indices(width, width, torch.device("cpu"))
        assert indices.sort().values.tolist() == list(range(width))
    assert _indices(4, 7, torch.device("cpu")).shape == (7,)


def test_random_test_initialization_keeps_bias_zero():
    model = get_peft_model(
        tiny_model("gpt_oss"), CortexConfig(k=3, moe_target="routed", adapter_bias=True, init_weights=False)
    )
    biases = [p for name, p in model.named_parameters() if "cortex_" in name and name.endswith("bias")]
    assert biases and all(torch.count_nonzero(p) == 0 for p in biases)
    assert any(
        p.count_nonzero() > 0 for name, p in model.named_parameters() if "cortex_down" in name and "weight" in name
    )


def test_routed_unbind_once_preserves_input_router_and_parameter_gradients(monkeypatch):
    model = get_peft_model(
        tiny_model("gpt_oss"), CortexConfig(k=3, moe_target="routed", adapter_bias=True, init_weights=False)
    )
    layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
    reference = deepcopy(layer)
    inputs = torch.randn(5, 16, requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [2, 1]])
    weights = torch.rand(5, 2, requires_grad=True)
    params = [p for p in layer.parameters() if p.requires_grad]
    calls = []
    original = ExpertLinear.expert_views

    def counted(self):
        calls.append(id(self))
        return original(self)

    monkeypatch.setattr(ExpertLinear, "expert_views", counted)
    actual = layer(inputs, indices, weights)
    actual_grads = torch.autograd.grad(actual.square().sum(), [inputs, weights, *params])
    assert len(calls) == len(set(calls)) == 3

    def selecting(self, x, expert, views=None):
        return F.linear(x, self.weight[expert], None if self.bias is None else self.bias[expert])

    monkeypatch.setattr(ExpertLinear, "forward", selecting)
    expected = reference(inputs, indices, weights)
    reference_params = [p for p in reference.parameters() if p.requires_grad]
    expected_grads = torch.autograd.grad(expected.square().sum(), [inputs, weights, *reference_params])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_preflight_does_not_construct_wide_ffns(monkeypatch):
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=3))
    original = merge_utils.merged_components
    calls = []

    def counted(layer, names, safe_merge=False):
        calls.append(id(layer))
        return original(layer, names, safe_merge)

    monkeypatch.setattr(merge_utils, "merged_components", counted)
    model.merge_adapter(safe_merge=True)
    assert len(calls) == len(set(calls)) == 2


@pytest.mark.parametrize("cause", ["base_nan", "cast_overflow", "bias_sum_overflow"])
def test_safe_preflight_detects_nonfinite_results_before_any_layer_changes(cause):
    kind = "gpt_oss" if cause == "bias_sum_overflow" else "llama"
    model = get_peft_model(tiny_model(kind).half(), CortexConfig(k=3, adapter_bias=kind == "gpt_oss"))
    layers = [module for module in model.modules() if isinstance(module, CortexLayer)]
    with torch.no_grad():
        if cause == "base_nan":
            next(layers[-1].base_layer.parameters()).flatten()[0] = float("nan")
        elif cause == "cast_overflow":
            layers[-1].cortex_up["default"].weight.fill_(1e10)
        else:
            layers[-1].base_layer.down_proj_bias.fill_(60000)
            layers[-1].cortex_down["default"].bias.fill_(60000)
    parameters = list(model.parameters())
    with pytest.raises(ValueError, match="Non-finite"):
        model.merge_adapter(safe_merge=True)
    assert all(not layer.merged and layer._merge_state is None for layer in layers)
    assert all(p is before for p, before in zip(model.parameters(), parameters))


def test_merge_allocation_failure_leaves_layer_unchanged(monkeypatch):
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=3))
    layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
    original = nn.Parameter
    parameters = list(layer.base_layer.parameters())
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(merge_utils.nn, "Parameter", fail_second)
    with pytest.raises(RuntimeError, match="simulated allocation failure"):
        layer.merge(safe_merge=True)
    assert all(p is before for p, before in zip(layer.base_layer.parameters(), parameters))
    assert layer.base_layer.up_proj.out_features == 24
    assert not layer.merged and layer._merge_state is None


def test_merge_assignment_failure_restores_parameters_and_dimensions(monkeypatch):
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=3))
    layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
    parameters = list(layer.base_layer.parameters())
    original = nn.Linear.__setattr__
    failed = False

    def fail_once(self, name, value):
        nonlocal failed
        if self is layer.base_layer.down_proj and name == "weight" and not failed:
            failed = True
            raise RuntimeError("simulated assignment failure")
        return original(self, name, value)

    monkeypatch.setattr(nn.Linear, "__setattr__", fail_once)
    with pytest.raises(RuntimeError, match="simulated assignment failure"):
        layer.merge(safe_merge=True)
    assert all(p is before for p, before in zip(layer.base_layer.parameters(), parameters))
    assert layer.base_layer.up_proj.out_features == layer.base_layer.down_proj.in_features == 24
    assert not layer.merged and layer._merge_state is None


@pytest.mark.parametrize("mode", ["shared", "routed"])
@pytest.mark.parametrize("order", ["native", "before", "after", "sequential"])
def test_native_and_accelerate_paths_preserve_nonzero_adapter_and_gradients(mode, order):
    base = tiny_model("gpt_oss").eval()
    parent = base.model.layers[0].mlp
    if order == "before":
        add_hook_to_module(parent, AlignDevicesHook(execution_device="cpu", offload=False))
    model = get_peft_model(base, CortexConfig(k=3, moe_target=mode)).eval()
    activate_branches(model)
    if order in ("after", "sequential"):
        add_hook_to_module(parent, AlignDevicesHook(execution_device="cpu", offload=False))
    if order == "sequential":
        add_hook_to_module(parent, AlignDevicesHook(execution_device="cpu", offload=False), append=True)
    reference = deepcopy(model)
    remove_hook_from_module(reference.get_base_model().model.layers[0].mlp)
    tokens = torch.randint(3, 32, (2, 5))
    actual, expected = model(tokens).logits, reference(tokens).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with model.disable_adapter():
        baseline = model(tokens).logits
    assert not torch.allclose(actual, baseline)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, parameter), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
        if "cortex_" in name:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.count_nonzero() > 0
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
        else:
            assert parameter.grad is None
    native = model.merge_and_unload(safe_merge=True)
    torch.testing.assert_close(native(tokens).logits, expected, rtol=2e-4, atol=3e-5)


@pytest.mark.parametrize("where", ["outer", "inner"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_accelerate_metadata_does_not_hide_bypassing_forward(where, when):
    base = tiny_model("gpt_oss")
    parent = base.model.layers[0].mlp
    add_hook_to_module(parent, AlignDevicesHook(execution_device="cpu", offload=False))
    if when == "after":
        model = get_peft_model(base, CortexConfig(k=3))

    @wraps(type(parent).forward)
    def replacement(self, *args, **kwargs):
        raise AssertionError("The bypassing replacement must not run.")

    setattr(parent, "forward" if where == "outer" else "_old_forward", MethodType(replacement, parent))
    with pytest.raises(ValueError, match="replaced MoE parent forward"):
        if when == "before":
            get_peft_model(base, CortexConfig(k=3))
        else:
            model(torch.randint(3, 32, (1, 5)))


def test_class_forward_replacement_after_injection_is_rejected(monkeypatch):
    base = tiny_model("gpt_oss")
    model = get_peft_model(base, CortexConfig(k=3))
    monkeypatch.setattr(type(base.model.layers[0].mlp), "forward", bypassing_kernel)
    with pytest.raises(ValueError, match="replaced MoE parent forward"):
        model(torch.randint(3, 32, (1, 5)))


def test_post_expert_norm_is_rejected_before_injection():
    base = tiny_model("gpt_oss")
    experts = base.model.layers[0].mlp.experts
    experts.has_post_expert_norm = True
    experts.post_expert_norm = nn.LayerNorm(base.config.hidden_size)
    with pytest.raises(ValueError, match="normalization inside individual expert outputs"):
        get_peft_model(base, CortexConfig(k=3))
    assert not any(isinstance(module, CortexLayer) for module in base.modules())


@pytest.mark.parametrize("mode", ["shared", "routed", ["shared", "routed"]])
def test_unmerge_releases_expanded_storage_and_does_not_retain_original_weights(mode):
    model = get_peft_model(tiny_model("gpt_oss"), CortexConfig(k=3, moe_target=mode, adapter_bias=True))
    activate_branches(model)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    layers = [module for module in model.modules() if isinstance(module, CortexLayer)]
    old_parameters = [weakref.ref(p) for layer in layers for p in layer.base_layer.parameters()]
    model.merge_adapter(safe_merge=True)
    gc.collect()
    assert all(reference() is None for reference in old_parameters)
    assert all(layer._original_down_bias.shape == (3, 16) for layer in layers)
    assert not any("_original_down_bias" in name for name in model.state_dict())
    wide_parameters = [weakref.ref(p) for layer in layers for p in layer.base_layer.parameters()]
    model.unmerge_adapter()
    gc.collect()
    assert all(reference() is None for reference in wide_parameters)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    for layer in layers:
        assert layer._merge_state is None and layer._original_down_bias is None
        for parameter in layer.base_layer.parameters():
            assert parameter.untyped_storage().nbytes() == parameter.numel() * parameter.element_size()


def test_unmerge_restores_bfloat16_down_bias_without_subtraction_roundoff():
    model = get_peft_model(tiny_model("gpt_oss").to(torch.bfloat16), CortexConfig(k=3, adapter_bias=True))
    layers = [module for module in model.modules() if isinstance(module, CortexLayer)]
    with torch.no_grad():
        for layer in layers:
            layer.base_layer.down_proj_bias.fill_(1.0)
            layer.cortex_down["default"].bias.fill_(1 / 256)
    model.merge_adapter()
    model.unmerge_adapter()
    for layer in layers:
        torch.testing.assert_close(
            layer.base_layer.down_proj_bias, torch.ones_like(layer.base_layer.down_proj_bias), rtol=0, atol=0
        )


@pytest.mark.parametrize("failure", ["setter", "registration_hook"])
def test_persistent_assignment_failure_restores_layer(monkeypatch, failure):
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=3)).eval()
    layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
    parameters = list(model.parameters())
    tokens = torch.randint(3, 32, (1, 5))
    expected = model(tokens).logits.detach()
    original_setattr = nn.Linear.__setattr__
    handle = None

    def reject(self, name, value):
        if self is layer.base_layer.down_proj and name == "weight":
            raise RuntimeError("persistent assignment failure")
        if failure == "setter":
            return original_setattr(self, name, value)

    if failure == "setter":
        monkeypatch.setattr(nn.Linear, "__setattr__", reject)
    else:
        handle = nn.modules.module.register_module_parameter_registration_hook(reject)
    try:
        with pytest.raises(RuntimeError, match="persistent assignment failure"):
            layer.merge(safe_merge=True)
    finally:
        if handle is not None:
            handle.remove()
    assert all(actual is before for actual, before in zip(model.parameters(), parameters))
    assert layer.base_layer.up_proj.out_features == layer.base_layer.down_proj.in_features == 24
    assert not layer.merged and layer._merge_state is None
    torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["merge_adapter", "merge_and_unload", "unmerge_adapter"])
def test_later_layer_failure_keeps_committed_ffns_and_checkpoint_can_restore(monkeypatch, operation, tmp_path):
    clean = tiny_model("llama").eval()
    model = get_peft_model(deepcopy(clean), CortexConfig(k=3)).eval()
    activate_branches(model)
    tokens = torch.randint(3, 32, (1, 5))
    with torch.no_grad():
        expected = model(tokens).logits
    model.save_pretrained(tmp_path)
    if operation == "unmerge_adapter":
        model.merge_adapter()
    first, second = [module for module in model.modules() if isinstance(module, CortexLayer)]
    first_parameters = list(first.base_layer.parameters())
    second_parameters = list(second.parameters())
    second_state = deepcopy(second.state_dict())
    original_setattr = nn.Linear.__setattr__

    def reject(self, name, value):
        if self is second.base_layer.down_proj and name == "weight":
            raise RuntimeError("later layer failure")
        return original_setattr(self, name, value)

    with monkeypatch.context() as patch:
        patch.setattr(nn.Linear, "__setattr__", reject)
        with pytest.raises(RuntimeError, match="later layer failure"):
            getattr(model, operation)()
    first_is_merged = operation != "unmerge_adapter"
    assert first.merged == first_is_merged
    assert second.merged != first_is_merged
    assert first.base_layer.up_proj.out_features == (27 if first_is_merged else 24)
    assert second.base_layer.up_proj.out_features == (24 if first_is_merged else 27)
    assert all(actual is not before for actual, before in zip(first.base_layer.parameters(), first_parameters))
    assert all(actual is before for actual, before in zip(second.parameters(), second_parameters))
    assert second.state_dict().keys() == second_state.keys()
    for name, value in second.state_dict().items():
        torch.testing.assert_close(value, second_state[name], rtol=0, atol=0)
    assert model.get_base_model().model.layers[0].mlp is (
        first.base_layer if operation == "merge_and_unload" else first
    )
    assert model.get_base_model().model.layers[1].mlp is second
    # Do not continue training or exporting the partially committed instance.
    restored = PeftModel.from_pretrained(clean, tmp_path).eval()
    torch.testing.assert_close(restored(tokens).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["merge_adapter", "merge_and_unload", "unmerge_adapter"])
def test_merge_operations_release_committed_ffn_parameters_before_next_layer(monkeypatch, operation):
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=3)).eval()
    activate_branches(model)
    tokens = torch.randint(3, 32, (1, 5))
    with torch.no_grad():
        expected = model(tokens).logits
    if operation == "unmerge_adapter":
        model.merge_adapter()
    first, second = [module for module in model.modules() if isinstance(module, CortexLayer)]
    old_parameters = [weakref.ref(parameter) for parameter in first.base_layer.parameters()]
    original = merge_utils._apply_components
    checked = False

    def check_released(module, *args, **kwargs):
        nonlocal checked
        if module is second.base_layer:
            gc.collect()
            assert all(reference() is None for reference in old_parameters)
            checked = True
        return original(module, *args, **kwargs)

    monkeypatch.setattr(merge_utils, "_apply_components", check_released)
    result = getattr(model, operation)()
    assert checked
    result = result if operation == "merge_and_unload" else model
    torch.testing.assert_close(result(tokens).logits, expected)


def test_export_failure_keeps_completed_updates_and_checkpoint_can_restore(monkeypatch, tmp_path):
    clean = tiny_model("llama").eval()
    base = deepcopy(clean)
    model = get_peft_model(
        base, CortexConfig(k=3, target_modules=["model.layers.0.mlp"], modules_to_save=["model.layers.1.mlp"])
    ).eval()
    activate_branches(model)
    retained = base.model.layers[1].mlp.modules_to_save["default"]
    with torch.no_grad():
        retained.down_proj.weight.add_(0.1)
    tokens = torch.randint(3, 32, (1, 5))
    with torch.no_grad():
        expected = model(tokens).logits
    model.save_pretrained(tmp_path)
    original_plan = model_utils.apply_export_plan

    def fail_after_export(plan):
        original_plan(plan)
        raise RuntimeError("failure after export updates")

    monkeypatch.setattr(model_utils, "apply_export_plan", fail_after_export)
    with pytest.raises(RuntimeError, match="failure after export updates"):
        model.merge_and_unload(safe_merge=True)
    assert base.config.intermediate_size == retained.config.intermediate_size == 27
    assert base.model.layers[1].mlp is retained
    assert [layer.mlp.down_proj.in_features for layer in base.model.layers] == [27, 27]
    assert not any(isinstance(module, CortexLayer) for module in base.modules())
    restored = PeftModel.from_pretrained(clean, tmp_path).eval()
    torch.testing.assert_close(restored(tokens).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["setter", "registration_hook"])
def test_padding_assignment_failure_restores_bare_ffn(monkeypatch, failure):
    layer = tiny_model("llama").model.layers[0].mlp.eval()
    parameters = list(layer.parameters())
    state = deepcopy(layer.state_dict())
    inputs = torch.randn(2, 5, 16)
    with torch.no_grad():
        expected = layer(inputs)
    original_setattr = nn.Linear.__setattr__
    handle = None

    def reject(self, name, value):
        if self is layer.down_proj and name == "weight":
            raise RuntimeError("persistent padding assignment failure")
        if failure == "setter":
            return original_setattr(self, name, value)

    if failure == "setter":
        monkeypatch.setattr(nn.Linear, "__setattr__", reject)
    else:
        handle = nn.modules.module.register_module_parameter_registration_hook(reject)
    try:
        with pytest.raises(RuntimeError, match="persistent padding assignment failure"):
            merge_utils.pad_width(layer, 27)
    finally:
        if handle is not None:
            handle.remove()
    assert all(actual is before for actual, before in zip(layer.parameters(), parameters))
    assert layer.up_proj.out_features == layer.down_proj.in_features == 24
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["merge", "unmerge"])
def test_merge_state_assignment_failure_restores_ffn_and_bias_buffer(monkeypatch, operation):
    model = get_peft_model(tiny_model("gpt_oss"), CortexConfig(k=3, adapter_bias=True)).eval()
    activate_branches(model)
    layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
    if operation == "unmerge":
        layer.merge(safe_merge=True)
    parameters = list(layer.parameters())
    state = deepcopy(layer.state_dict())
    merge_state = layer._merge_state
    down_bias = layer._original_down_bias
    merged_adapters = layer.merged_adapters
    assert (down_bias is not None) == (operation == "unmerge")
    tokens = torch.randint(3, 32, (1, 5))
    with torch.no_grad():
        expected = model(tokens).logits
    original_setattr = CortexLayer.__setattr__
    saw_updated_weights_and_state = False

    def reject(self, name, value):
        nonlocal saw_updated_weights_and_state
        if self is layer and name == "merged_adapters":
            saw_updated_weights_and_state = (
                merge_utils.describe_mlp(self.base_layer).intermediate_dim == (27 if operation == "merge" else 24)
                and (self._original_down_bias is None) == (operation == "unmerge")
                and (self._merge_state is None) == (operation == "unmerge")
                and bool(value) == (operation == "merge")
            )
            raise RuntimeError("merge state assignment failure")
        return original_setattr(self, name, value)

    monkeypatch.setattr(CortexLayer, "__setattr__", reject)
    with pytest.raises(RuntimeError, match="merge state assignment failure"):
        getattr(layer, operation)()
    assert saw_updated_weights_and_state
    assert layer._merge_state is merge_state
    assert layer._original_down_bias is down_bias
    assert layer.merged_adapters is merged_adapters
    assert all(actual is before for actual, before in zip(layer.parameters(), parameters))
    assert layer.state_dict().keys() == state.keys()
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("selection", [["routed", "shared"], ["shared", "routed"]])
def test_moe_selection_normalizes_and_roundtrips(selection, tmp_path):
    config = CortexConfig(k=3, moe_target=selection)
    assert config.moe_target == ["shared", "routed"]
    model = get_peft_model(tiny_model("gpt_oss"), config)
    model.add_adapter("second", CortexConfig(k=2, moe_target=["shared", "routed"]))
    assert all(layer.role == "routed_public" for layer in model.modules() if isinstance(layer, CortexLayer))
    model.save_pretrained(tmp_path)
    assert CortexConfig.from_pretrained(tmp_path).moe_target == ["shared", "routed"]


@pytest.mark.parametrize("selection", ["shared", "routed"])
def test_singleton_moe_selection_roundtrips(selection, tmp_path):
    config = CortexConfig(k=3, moe_target=[selection])
    assert config.moe_target == selection
    config.save_pretrained(tmp_path)
    assert CortexConfig.from_pretrained(tmp_path).moe_target == selection


@pytest.mark.parametrize(
    "selection", ["all", [], ["shared", "shared"], ["all"], ["shared", "all"], ["shared", None], True, {}]
)
def test_invalid_moe_selections_fail_clearly(selection, tmp_path):
    with pytest.raises(ValueError, match="moe_target"):
        CortexConfig(moe_target=selection)
    CortexConfig().save_pretrained(tmp_path)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["moe_target"] = selection
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="moe_target"):
        CortexConfig.from_pretrained(tmp_path)


def test_default_moe_selection_is_shared():
    assert CortexConfig().moe_target == "shared"
    model = get_peft_model(tiny_model("gpt_oss"), CortexConfig(k=3))
    layers = [module for module in model.modules() if isinstance(module, CortexLayer)]
    assert layers and all(layer.role == "public" for layer in layers)
    assert all(not layer.per_expert for layer in layers)


@pytest.mark.parametrize("mode", ["shared", "routed", ["shared", "routed"]])
def test_moe_sample_copies_paired_real_channels_and_biases(mode):
    from .test_cortex_distributed import assert_initialization

    base = tiny_model("gpt_oss")
    with torch.no_grad():
        for name, parameter in base.named_parameters():
            if "experts" in name and "bias" in name:
                parameter.normal_()
    model = get_peft_model(
        base, CortexConfig(k=4, moe_target=mode, adapter_bias=True, init_weights="sample", init_noise_std=0)
    )
    assert_initialization(model, stage=0, init="sample")


@pytest.mark.parametrize("operation", ["merge", "unmerge"])
@pytest.mark.parametrize("with_kwargs", [False, True])
def test_failed_layer_merge_preserves_user_hook_handles(monkeypatch, operation, with_kwargs):
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=3)).eval()
    activate_branches(model)
    layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
    if operation == "unmerge":
        layer.merge()
    base = layer.base_layer
    calls = []

    def plain_hook(module, args):
        calls.append("plain")

    def kwargs_hook(module, args, kwargs):
        calls.append("kwargs")

    handle = base.register_forward_pre_hook(kwargs_hook if with_kwargs else plain_hook, with_kwargs=with_kwargs)
    hooks = base._forward_pre_hooks
    kwargs_flags = base._forward_pre_hooks_with_kwargs
    hook_contents, flag_contents = list(hooks.items()), list(kwargs_flags.items())
    parameters = list(layer.parameters())
    inputs = torch.randn(2, 5, 16)
    with torch.no_grad():
        expected = layer(inputs)
    original_setattr = nn.Linear.__setattr__

    def reject(self, name, value):
        if self is base.down_proj and name == "weight":
            raise RuntimeError("merge failure with user hook")
        return original_setattr(self, name, value)

    monkeypatch.setattr(nn.Linear, "__setattr__", reject)
    with pytest.raises(RuntimeError, match="merge failure with user hook"):
        getattr(layer, operation)()
    assert all(actual is before for actual, before in zip(layer.parameters(), parameters))
    assert base._forward_pre_hooks is hooks
    assert base._forward_pre_hooks_with_kwargs is kwargs_flags
    assert list(hooks.items()) == hook_contents and list(kwargs_flags.items()) == flag_contents
    calls.clear()
    torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
    assert calls == (["kwargs"] if with_kwargs else ["plain"])
    calls.clear()
    handle.remove()
    assert handle.id not in hooks and handle.id not in kwargs_flags
    torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
    assert not calls


@pytest.mark.parametrize("failure", ["setter", "registration_hook"])
def test_direct_unmerge_assignment_failure_preserves_merged_layer(monkeypatch, failure):
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=3)).eval()
    activate_branches(model)
    layer = next(module for module in model.modules() if isinstance(module, CortexLayer))
    layer.merge(safe_merge=True)
    parameters = list(layer.parameters())
    state = deepcopy(layer.state_dict())
    merge_state = layer._merge_state
    inputs = torch.randn(2, 5, 16)
    expected = layer(inputs).detach()
    original_setattr = nn.Linear.__setattr__
    handle = None

    def reject(self, name, value):
        if self is layer.base_layer.down_proj and name == "weight":
            raise RuntimeError("persistent unmerge assignment failure")
        if failure == "setter":
            return original_setattr(self, name, value)

    if failure == "setter":
        monkeypatch.setattr(nn.Linear, "__setattr__", reject)
    else:
        handle = nn.modules.module.register_module_parameter_registration_hook(reject)
    try:
        with pytest.raises(RuntimeError, match="persistent unmerge assignment failure"):
            layer.unmerge()
    finally:
        if handle is not None:
            handle.remove()

    assert layer.merged_adapters == ["default"]
    assert layer._merge_state is merge_state
    assert layer.base_layer.up_proj.out_features == layer.base_layer.down_proj.in_features == 27
    assert all(actual is original for actual, original in zip(layer.parameters(), parameters))
    assert layer.state_dict().keys() == state.keys()
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "excluded,retained,role",
    [("experts", "shared_expert", "shared"), ("shared_expert", "experts", "routed")],
)
def test_resolved_moe_exclusion_survives_config_reuse_and_checkpoint(excluded, retained, role, tmp_path):
    base = BlockModel("qwen3_next").eval()
    clean = deepcopy(base)
    config = CortexConfig(
        k=3, target_modules=["mlp"], exclude_modules=[f"mlp.{excluded}"], moe_target=["shared", "routed"]
    )
    model = get_peft_model(base, config).eval()
    activate_branches(model)
    expected_roles = {f"mlp.{retained}": role}
    assert config.target_roles == expected_roles
    assert not isinstance(base.get_submodule(f"mlp.{excluded}"), CortexLayer)

    reused = get_peft_model(deepcopy(clean), config).eval()
    assert reused.peft_config["default"] is config
    assert config.target_roles == expected_roles
    assert not isinstance(reused.get_base_model().get_submodule(f"mlp.{excluded}"), CortexLayer)
    inputs = torch.randn(2, 5, 16)
    expected = model(inputs).detach()
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    loaded_config = CortexConfig.from_pretrained(tmp_path)
    assert loaded_config.target_roles == expected_roles
    assert not hasattr(loaded_config, "_cortex_user_targets")
    loaded = PeftModel.from_pretrained(clean, tmp_path, config=loaded_config).eval()
    assert loaded.peft_config["default"].target_roles == expected_roles
    assert not isinstance(clean.get_submodule(f"mlp.{excluded}"), CortexLayer)
    torch.testing.assert_close(loaded(inputs), expected)


def test_loaded_targets_stay_fixed_on_larger_model_and_conflicting_exclusion_fails(tmp_path):
    model = get_peft_model(
        tiny_model("llama"),
        CortexConfig(k=3, exclude_modules=["model.layers.1.mlp"]),
    )
    model.save_pretrained(tmp_path)
    enlarged_config = deepcopy(model.get_base_model().config)
    enlarged_config.num_hidden_layers = 3
    saved_config = CortexConfig.from_pretrained(tmp_path)
    expected_roles = {"model.layers.0.mlp": "dense"}
    assert saved_config.target_roles == expected_roles
    loaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_config(enlarged_config), tmp_path, config=saved_config
    )
    assert loaded.peft_config["default"].target_roles == expected_roles
    assert sum(isinstance(module, CortexLayer) for module in loaded.modules()) == 1
    assert not isinstance(loaded.get_base_model().model.layers[2].mlp, CortexLayer)

    conflicting_config = CortexConfig.from_pretrained(tmp_path)
    conflicting_config.exclude_modules = {"model.layers.0.mlp"}
    base = AutoModelForCausalLM.from_config(enlarged_config)
    parameters = list(base.parameters())
    with pytest.raises(ValueError, match="Resolved adapter targets do not match"):
        PeftModel.from_pretrained(base, tmp_path, config=conflicting_config)
    assert not any(isinstance(module, CortexLayer) for module in base.modules())
    assert all(actual is original for actual, original in zip(base.parameters(), parameters))
    assert conflicting_config.target_roles == expected_roles


@pytest.mark.parametrize("kind", ["gpt_oss", "qwen3_moe"])
@pytest.mark.parametrize("selection,expected_role", [("parent", "public"), ("leaf", "routed")])
def test_explicit_expert_selection_and_saved_public_role(kind, selection, expected_role, tmp_path):
    base = tiny_model(kind).eval()
    clean = deepcopy(base)
    parent_path = "model.layers.0.mlp"
    expert_path = parent_path + ".experts"
    target = parent_path if selection == "parent" else expert_path
    model = get_peft_model(base, CortexConfig(k=3, target_modules=[target], moe_target="shared")).eval()
    expected_roles = {expert_path: expected_role}
    assert model.peft_config["default"].target_roles == expected_roles
    layer = base.get_submodule(expert_path)
    assert layer.role == expected_role
    expected_shape = (3, 3, 16) if expected_role == "routed" else (3, 16)
    assert tuple(layer.cortex_up["default"].weight.shape) == expected_shape
    per_branch_parameters = 166 if kind == "gpt_oss" else 144
    expected_parameters = per_branch_parameters * (3 if expected_role == "routed" else 1)
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == expected_parameters
    activate_branches(model)
    tokens = torch.randint(3, 32, (2, 5))
    expected = model(tokens).logits.detach()
    assert not torch.allclose(expected, clean(tokens).logits)
    model.save_pretrained(tmp_path, save_embedding_layers=False)

    saved_config = CortexConfig.from_pretrained(tmp_path)
    assert saved_config.target_modules == {expert_path}
    assert saved_config.target_roles == expected_roles
    assert not hasattr(saved_config, "_cortex_user_targets")
    restored = PeftModel.from_pretrained(clean, tmp_path, config=saved_config).eval()
    assert restored.get_base_model().get_submodule(expert_path).role == expected_role
    assert restored.peft_config["default"].target_roles == expected_roles
    torch.testing.assert_close(restored(tokens).logits, expected)


@pytest.mark.parametrize("kind", ["gpt_oss", "qwen3_moe"])
@pytest.mark.parametrize("first_selection", ["parent", "leaf"])
def test_conflicting_parent_and_expert_selections_fail_before_mutation(kind, first_selection):
    parent_path = "model.layers.1.mlp"
    expert_path = parent_path + ".experts"
    first_target, second_target = (
        (parent_path, expert_path) if first_selection == "parent" else (expert_path, parent_path)
    )
    config = CortexConfig(k=3, target_modules=[first_target], moe_target="shared")
    model = get_peft_model(tiny_model(kind), config).eval()
    activate_branches(model)
    # Layer zero is otherwise valid; resolving a later conflicting role must
    # fail before adding any part of the new adapter.
    second_config = CortexConfig(k=5, target_modules=["model.layers.0.mlp", second_target], moe_target="shared")
    configs_before = (deepcopy(config.to_dict()), deepcopy(second_config.to_dict()))
    roles_before = {name: layer.role for name, layer in model.named_modules() if isinstance(layer, CortexLayer)}
    modules_before = list(model.modules())
    parameters_before = list(model.parameters())
    trainable_before = [parameter.requires_grad for parameter in model.parameters()]
    weights_before = deepcopy(model.state_dict())
    tokens = torch.randint(3, 32, (2, 5))
    expected = model(tokens).logits.detach()

    with pytest.raises(ValueError, match="role"):
        model.add_adapter("second", second_config)

    assert set(model.peft_config) == {"default"}
    assert config.to_dict() == configs_before[0]
    assert second_config.to_dict() == configs_before[1]
    assert {
        name: layer.role for name, layer in model.named_modules() if isinstance(layer, CortexLayer)
    } == roles_before
    assert len(list(model.modules())) == len(modules_before)
    assert all(actual is original for actual, original in zip(model.modules(), modules_before))
    assert len(list(model.parameters())) == len(parameters_before)
    assert all(actual is original for actual, original in zip(model.parameters(), parameters_before))
    assert [parameter.requires_grad for parameter in model.parameters()] == trainable_before
    assert model.state_dict().keys() == weights_before.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights_before[name], rtol=0, atol=0)
    torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind,first_target,second_target,first_roles,second_roles",
    [
        pytest.param(
            "llama",
            "model.layers.0.mlp",
            "model.layers.0.mlp",
            {"model.layers.0.mlp": "dense"},
            {"model.layers.0.mlp": "dense"},
            id="dense-same-layer",
        ),
        pytest.param(
            "llama",
            "model.layers.0.mlp",
            "model.layers.1.mlp",
            {"model.layers.0.mlp": "dense"},
            {"model.layers.1.mlp": "dense"},
            id="dense-different-layers",
        ),
        pytest.param(
            "gpt_oss",
            "model.layers.0.mlp",
            "model.layers.1.mlp",
            {"model.layers.0.mlp.experts": "public"},
            {"model.layers.1.mlp.experts": "routed"},
            id="gpt-oss-different-layers",
        ),
        pytest.param(
            "qwen3_moe",
            "model.layers.0.mlp",
            "model.layers.1.mlp",
            {"model.layers.0.mlp.experts": "public"},
            {"model.layers.1.mlp.experts": "routed"},
            id="qwen3-moe-different-layers",
        ),
        pytest.param(
            "qwen3_next",
            "mlp",
            "mlp",
            {"mlp.shared_expert": "shared"},
            {"mlp.experts": "routed"},
            id="native-shared-and-routed",
        ),
    ],
)
def test_different_moe_selections_with_compatible_physical_roles(
    kind, first_target, second_target, first_roles, second_roles, tmp_path
):
    clean = (BlockModel(kind) if kind == "qwen3_next" else tiny_model(kind)).eval()
    model = get_peft_model(
        deepcopy(clean),
        CortexConfig(
            k=3,
            target_modules=[first_target],
            moe_target="shared",
            public_weighting="routing" if kind == "qwen3_moe" else "unweighted",
        ),
    ).eval()
    model.add_adapter("second", CortexConfig(k=5, target_modules=[second_target], moe_target="routed"))
    assert model.peft_config["default"].target_roles == first_roles
    assert model.peft_config["second"].target_roles == second_roles
    activate_branches(model)
    inputs = torch.randn(2, 5, 16) if kind == "qwen3_next" else torch.randint(3, 32, (2, 5))

    def output(current):
        result = current(inputs)
        return result if kind == "qwen3_next" else result.logits

    reference = output(clean).detach()
    expected = {}
    for adapter in ("default", "second"):
        model.set_adapter(adapter)
        model.zero_grad(set_to_none=True)
        result = output(model)
        expected[adapter] = result.detach()
        assert not torch.allclose(result, reference)
        result.square().mean().backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert trainable
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
        assert any(parameter.grad.abs().sum() > 0 for parameter in trainable)
        assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)

    model.base_model.set_adapter(["default", "second"])
    expected_combined = output(model).detach()
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    restored = PeftModel.from_pretrained(deepcopy(clean), tmp_path, is_trainable=True).eval()
    restored.load_adapter(tmp_path / "second", "second", is_trainable=True)
    assert restored.peft_config["default"].target_roles == first_roles
    assert restored.peft_config["second"].target_roles == second_roles
    assert restored.peft_config["default"].moe_target == "shared"
    assert restored.peft_config["second"].moe_target == "routed"
    for adapter in ("default", "second"):
        restored.set_adapter(adapter)
        torch.testing.assert_close(output(restored), expected[adapter])
    restored.base_model.set_adapter(["default", "second"])
    torch.testing.assert_close(output(restored), expected_combined)
    restored.merge_adapter(adapter_names=["default"], safe_merge=True)
    restored.zero_grad(set_to_none=True)
    partial = output(restored)
    torch.testing.assert_close(partial, expected_combined, rtol=2e-4, atol=3e-5)
    partial.square().mean().backward()
    remaining = [parameter for name, parameter in restored.named_parameters() if ".second." in name]
    assert remaining
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in remaining)
    assert any(parameter.grad.abs().sum() > 0 for parameter in remaining)
    restored.unmerge_adapter()
    torch.testing.assert_close(output(restored), expected_combined, rtol=0, atol=0)
    restored.merge_adapter(safe_merge=True)
    torch.testing.assert_close(output(restored), expected_combined, rtol=2e-4, atol=3e-5)
    restored.unmerge_adapter()
    torch.testing.assert_close(output(restored), expected_combined, rtol=0, atol=0)
    native = restored.merge_and_unload(safe_merge=True)
    assert not any(isinstance(module, CortexLayer) for module in native.modules())
    torch.testing.assert_close(output(native), expected_combined, rtol=2e-4, atol=3e-5)


@pytest.mark.parametrize("saved_target", ["model.layers.0", "model.layers.0.mlp"])
def test_auto_discovery_preserves_auxiliary_ownership_and_independent_checkpoints(saved_target, tmp_path):
    clean = tiny_model("llama").eval()
    model = get_peft_model(deepcopy(clean), CortexConfig(k=2, modules_to_save=[saved_target])).eval()
    wrapper = model.get_base_model().get_submodule(saved_target)
    saved_copy = wrapper.modules_to_save["default"]
    copied_ffn = saved_copy if saved_target.endswith(".mlp") else saved_copy.mlp
    with torch.no_grad():
        copied_ffn.down_proj.weight.add_(0.2)
    copy_parameters = list(saved_copy.parameters())
    copy_weights = deepcopy(saved_copy.state_dict())

    model.add_adapter("b", CortexConfig(k=3))
    assert model.peft_config["b"].target_roles == {"model.layers.1.mlp": "dense"}
    assert model.peft_config["b"].target_modules == {"model.layers.1.mlp"}
    assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
    assert all(actual is before for actual, before in zip(saved_copy.parameters(), copy_parameters))
    for name, value in saved_copy.state_dict().items():
        torch.testing.assert_close(value, copy_weights[name], rtol=0, atol=0)
    assert sum(parameter.numel() for name, parameter in model.named_parameters() if ".b." in name) == 144

    activate_branches(model)
    tokens = torch.randint(3, 32, (2, 5))
    expected = {}
    for adapter in ("default", "b"):
        model.set_adapter(adapter)
        expected[adapter] = model(tokens).logits.detach()
    assert not torch.allclose(expected["b"], clean(tokens).logits)
    assert not torch.allclose(expected["default"], expected["b"])
    model.zero_grad(set_to_none=True)
    model(tokens).logits.square().mean().backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert sum(parameter.numel() for parameter in trainable) == 144
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert any(parameter.grad.abs().sum() > 0 for parameter in trainable)
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)

    model.save_pretrained(tmp_path / "b_only", selected_adapters=["b"], save_embedding_layers=False)
    model.save_pretrained(tmp_path / "first", selected_adapters=["default"], save_embedding_layers=False)
    locations = {"b": tmp_path / "b_only" / "b", "default": tmp_path / "first"}
    disk_weights = load_file(str(locations["b"] / "adapter_model.safetensors"))
    assert len(disk_weights) == 3
    assert sum(value.numel() for value in disk_weights.values()) == 144
    expected_weights = {
        name: get_peft_model_state_dict(model, adapter_name=name, save_embedding_layers=False)
        for name in ("default", "b")
    }
    assert disk_weights.keys() == expected_weights["b"].keys()
    for name, value in disk_weights.items():
        torch.testing.assert_close(value, expected_weights["b"][name], rtol=0, atol=0)

    # Each checkpoint first loads independently; then load the other adapter.
    for first, second in (("b", "default"), ("default", "b")):
        fresh_base = deepcopy(clean)
        base_parameters = [(parameter, parameter.detach().clone()) for parameter in fresh_base.parameters()]
        restored = PeftModel.from_pretrained(fresh_base, locations[first], adapter_name=first).eval()
        torch.testing.assert_close(restored(tokens).logits, expected[first])
        for parameter, original in base_parameters:
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        restored.load_adapter(locations[second], adapter_name=second)
        for adapter in ("default", "b"):
            restored.set_adapter(adapter)
            assert restored.peft_config[adapter].target_roles == {"model.layers.1.mlp": "dense"}
            restored_weights = get_peft_model_state_dict(restored, adapter_name=adapter, save_embedding_layers=False)
            assert restored_weights.keys() == expected_weights[adapter].keys()
            for name, value in restored_weights.items():
                torch.testing.assert_close(value, expected_weights[adapter][name], rtol=0, atol=0)
            torch.testing.assert_close(restored(tokens).logits, expected[adapter])


@pytest.mark.parametrize("selection", ["wrapper", "original", "copy", "regex", "saved_roles"])
def test_explicit_auxiliary_targets_fail_before_adapter_mutation(selection, tmp_path):
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=2, modules_to_save=["model.layers.1.mlp"])).eval()
    activate_branches(model)
    wrapper_path = "model.layers.1.mlp"
    invalid_path = (
        wrapper_path
        + {
            "wrapper": "",
            "original": ".original_module",
            "copy": ".modules_to_save.default",
            "regex": ".original_module",
            "saved_roles": ".modules_to_save.default",
        }[selection]
    )
    targets = (
        r"model\.layers\.(?:0\.mlp|1\.mlp\.original_module)"
        if selection == "regex"
        else ["model.layers.0.mlp", invalid_path]
    )
    config = CortexConfig(k=3, target_modules=targets)
    if selection == "saved_roles":
        config.target_roles = {"model.layers.0.mlp": "dense", invalid_path: "dense"}
        config.save_pretrained(tmp_path)
        config = CortexConfig.from_pretrained(tmp_path)
    config_before = deepcopy(config.to_dict())
    existing_config = deepcopy(model.peft_config["default"].to_dict())
    modules = list(model.modules())
    parameters = list(model.parameters())
    trainable = [parameter.requires_grad for parameter in parameters]
    state = deepcopy(model.state_dict())
    tokens = torch.randint(3, 32, (2, 5))
    expected = model(tokens).logits.detach()

    with pytest.raises(ValueError, match="auxiliary training wrapper"):
        model.add_adapter("b", config)

    assert set(model.peft_config) == {"default"}
    assert config.to_dict() == config_before
    assert model.peft_config["default"].to_dict() == existing_config
    assert len(list(model.modules())) == len(modules)
    assert all(actual is before for actual, before in zip(model.modules(), modules))
    assert len(list(model.parameters())) == len(parameters)
    assert all(actual is before for actual, before in zip(model.parameters(), parameters))
    assert [parameter.requires_grad for parameter in model.parameters()] == trainable
    assert model.state_dict().keys() == state.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("exclusion", ["exclude_modules", "modules_to_save"])
def test_current_adapter_exclusion_precedes_auxiliary_target_rejection(exclusion):
    wrapper_path = "model.layers.1.mlp"
    model = get_peft_model(tiny_model("llama"), CortexConfig(k=2, modules_to_save=[wrapper_path]))
    config = CortexConfig(
        k=3,
        target_modules=["model.layers.0.mlp", wrapper_path],
        **{exclusion: [wrapper_path]},
    )
    model.add_adapter("b", config)
    assert config.target_roles == {"model.layers.0.mlp": "dense"}
    wrapper = model.get_base_model().get_submodule(wrapper_path)
    assert isinstance(wrapper, ModulesToSaveWrapper)
    assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
    assert set(wrapper.modules_to_save) == ({"default", "b"} if exclusion == "modules_to_save" else {"default"})
    model.set_adapter("b")
    tokens = torch.randint(3, 32, (2, 5))
    assert torch.isfinite(model(tokens).logits).all()


def test_auto_discovery_rejects_when_all_ffns_belong_to_auxiliary_wrapper():
    base = tiny_model("llama")
    base.model = ModulesToSaveWrapper(base.model, adapter_name="old")
    modules = list(base.modules())
    parameters = list(base.parameters())
    with pytest.raises(ValueError, match="FFN"):
        get_peft_model(base, CortexConfig(k=3))
    assert not any(isinstance(module, CortexLayer) for module in base.modules())
    assert len(list(base.modules())) == len(modules)
    assert all(actual is before for actual, before in zip(base.modules(), modules))
    assert all(actual is before for actual, before in zip(base.parameters(), parameters))


def _protected_moe(protected):
    torch.manual_seed(123)
    base = BlockModel("qwen3_next").eval()
    clean = deepcopy(base)
    retained = "shared_expert" if protected == "experts" else "experts"
    role = "shared" if retained == "shared_expert" else "routed"
    model = get_peft_model(
        base,
        CortexConfig(
            k=2,
            target_modules=[f"mlp.{retained}"],
            modules_to_save=[f"mlp.{protected}"],
            moe_target=role,
        ),
    ).eval()
    wrapper = base.get_submodule(f"mlp.{protected}")
    assert isinstance(wrapper, ModulesToSaveWrapper)
    return clean, model, wrapper, retained, role


@pytest.mark.parametrize("protected", ["experts", "shared_expert"])
@pytest.mark.parametrize("selection", ["auto-both", "auto-retained", "explicit-retained"])
def test_aux_moe_retained_branch_checkpoint_and_nonzero_output(protected, selection, tmp_path):
    clean, model, wrapper, retained, role = _protected_moe(protected)
    targets = None if selection.startswith("auto") else ["mlp"]
    mode = ["shared", "routed"] if selection == "auto-both" else role
    model.add_adapter("b", CortexConfig(k=3, target_modules=targets, moe_target=mode))
    assert model.peft_config["b"].target_roles == {f"mlp.{retained}": role}
    assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
    activate_branches(model)
    # Distinguish default's owned copy from the original base used by b alone.
    with torch.no_grad():
        next(wrapper.modules_to_save["default"].parameters()).add_(0.2)
    inputs = torch.randn(2, 5, 16)
    model.set_adapter("b")
    expected = model(inputs)
    with model.disable_adapter():
        baseline = model(inputs)
    assert not torch.allclose(expected, baseline)
    expected.square().mean().backward()
    parameters = [p for name, p in model.named_parameters() if "cortex_" in name and ".b." in name]
    assert parameters
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert all(p.grad.count_nonzero() > 0 for p in parameters)
    state = get_peft_model_state_dict(model, adapter_name="b")
    assert len(state) == 3
    assert all(f"mlp.{retained}.cortex_" in name for name in state)
    assert not any("original_module" in name or "modules_to_save" in name for name in state)
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    loaded = PeftModel.from_pretrained(deepcopy(clean), tmp_path / "b", adapter_name="b").eval()
    torch.testing.assert_close(loaded(inputs), expected, rtol=0, atol=0)
    model.base_model.set_adapter(["default", "b"])
    cumulative = model(inputs).detach()
    loaded = PeftModel.from_pretrained(deepcopy(clean), tmp_path).eval()
    loaded.load_adapter(tmp_path / "b", adapter_name="b")
    loaded.base_model.set_adapter(["default", "b"])
    torch.testing.assert_close(loaded(inputs), cumulative, rtol=0, atol=0)
    native = loaded.merge_and_unload(safe_merge=True)
    torch.testing.assert_close(native(inputs), cumulative, rtol=2e-4, atol=3e-5)
    reloaded_native = BlockModel("qwen3_next", config=deepcopy(native.config)).eval()
    reloaded_native.load_state_dict(native.state_dict(), strict=True)
    torch.testing.assert_close(reloaded_native(inputs), cumulative, rtol=2e-4, atol=3e-5)


@pytest.mark.parametrize("protected", ["experts", "shared_expert"])
@pytest.mark.parametrize("mode", ["protected", "both"])
def test_aux_moe_explicit_parent_rejects_requested_protected_branch(protected, mode):
    _, model, wrapper, _, _ = _protected_moe(protected)
    protected_role = "routed" if protected == "experts" else "shared"
    selection = ["shared", "routed"] if mode == "both" else protected_role
    original_parameters = list(model.parameters())
    config = CortexConfig(k=3, target_modules=["mlp"], moe_target=selection)
    with pytest.raises(ValueError, match="auxiliary training wrapper") as error:
        model.add_adapter("b", config)
    assert "older ModuleList" not in str(error.value)
    assert "b" not in model.peft_config
    assert all(actual is original for actual, original in zip(model.parameters(), original_parameters))
    assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
    assert not any(".b." in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("protected", ["experts", "shared_expert"])
def test_aux_moe_auto_only_protected_branch_fails_without_mutating_model(protected):
    _, model, wrapper, _, _ = _protected_moe(protected)
    selection = "routed" if protected == "experts" else "shared"
    with pytest.raises(ValueError) as error:
        model.add_adapter("b", CortexConfig(k=3, moe_target=selection))
    assert "older ModuleList" not in str(error.value)
    assert "b" not in model.peft_config
    assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
    assert not any(".b." in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("exclusion", ["exclude_modules", "modules_to_save"])
def test_aux_moe_current_config_exclusion_precedes_existing_expert_boundary(exclusion, tmp_path):
    clean, model, wrapper, retained, role = _protected_moe("experts")
    config = CortexConfig(
        k=3,
        target_modules=["mlp"],
        moe_target=["shared", "routed"],
        **{exclusion: ["mlp.experts"]},
    )
    model.add_adapter("b", config)
    assert config.target_roles == {f"mlp.{retained}": role}
    assert not any(isinstance(module, CortexLayer) for module in wrapper.modules())
    assert set(wrapper.modules_to_save) == ({"default", "b"} if exclusion == "modules_to_save" else {"default"})
    activate_branches(model)
    model.set_adapter("b")
    inputs = torch.randn(2, 5, 16)
    expected = model(inputs).detach()
    with model.disable_adapter():
        assert not torch.allclose(model(inputs), expected)
    model.save_pretrained(tmp_path, selected_adapters=["b"], save_embedding_layers=False)
    loaded = PeftModel.from_pretrained(clean, tmp_path / "b", adapter_name="b").eval()
    torch.testing.assert_close(loaded(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("saved_target", ["model.layers.0", "model.layers.0.mlp"])
def test_modules_to_save_cannot_enclose_existing_cortex_before_other_injection(saved_target, tmp_path):
    clean = tiny_model("llama").eval()
    old_config = CortexConfig(k=3)
    model = get_peft_model(deepcopy(clean), old_config).eval()
    activate_branches(model)
    tokens = torch.randint(3, 32, (2, 5))
    result = model(tokens).logits
    expected = result.detach()
    result.square().mean().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())

    config = CortexConfig(k=2, target_modules=["model.layers.1.mlp"], modules_to_save=[saved_target])
    configs = (deepcopy(old_config.to_dict()), deepcopy(config.to_dict()))
    modules = list(model.modules())
    parameters = list(model.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    gradients = [None if parameter.grad is None else parameter.grad.clone() for parameter in parameters]
    state = deepcopy(model.state_dict())
    active = model.active_adapter
    active_layers = {
        name: list(layer.active_adapters) for name, layer in model.named_modules() if isinstance(layer, CortexLayer)
    }

    with pytest.raises(ValueError, match="modules_to_save.*existing Cortex"):
        model.add_adapter("b", config)

    assert set(model.peft_config) == {"default"}
    assert model.peft_config["default"] is old_config
    assert old_config.to_dict() == configs[0]
    assert config.to_dict() == configs[1]
    assert model.active_adapter == active
    assert {
        name: list(layer.active_adapters) for name, layer in model.named_modules() if isinstance(layer, CortexLayer)
    } == active_layers
    assert len(list(model.modules())) == len(modules)
    assert all(actual is before for actual, before in zip(model.modules(), modules))
    assert len(list(model.parameters())) == len(parameters)
    assert all(actual is before for actual, before in zip(model.parameters(), parameters))
    assert [parameter.requires_grad for parameter in model.parameters()] == requires_grad
    for parameter, gradient in zip(model.parameters(), gradients):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert model.state_dict().keys() == state.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)

    adapter_state = get_peft_model_state_dict(model, save_embedding_layers=False)
    model.save_pretrained(tmp_path, selected_adapters=["default"], save_embedding_layers=False)
    restored = PeftModel.from_pretrained(deepcopy(clean), tmp_path).eval()
    restored_state = get_peft_model_state_dict(restored, save_embedding_layers=False)
    assert restored_state.keys() == adapter_state.keys()
    for name, value in restored_state.items():
        torch.testing.assert_close(value, adapter_state[name], rtol=0, atol=0)
    torch.testing.assert_close(restored(tokens).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind,init",
    [
        pytest.param("gpt_oss", "sample", id="sample-with-source-bias"),
        pytest.param("qwen3_moe", "sample", id="sample-without-source-bias"),
        pytest.param("gpt_oss", "random", id="random"),
        pytest.param("gpt_oss", False, id="random-down"),
    ],
)
@pytest.mark.parametrize("bias_init", ["uniform", "zeros"])
def test_branch_initialization_overwrites_all_previous_parameter_values(kind, init, bias_init):
    from peft.tuners.cortex.initialization import initialize_branch
    from peft.tuners.cortex.structure import read_components

    base = tiny_model(kind)
    source = read_components(base.model.layers[0].mlp.experts, include_down=False)
    with torch.no_grad():
        for projection_kind, value in (("gate", 0.375), ("up", 0.75)):
            if projection_kind + "_bias" in source:
                source[projection_kind + "_bias"].fill_(value)
    config = CortexConfig(
        k=3,
        moe_target=["shared", "routed"],
        adapter_bias=True,
        adapter_bias_init=bias_init,
        init_weights=init,
        init_noise_std=0,
    )
    model = get_peft_model(base, config).eval()
    layer = next(module for module in model.modules() if isinstance(module, CortexLayer))

    # Both packed expert projections and the shared nn.Linear branch must be
    # completely initialized independently of their constructor's old contents.
    for prefix, per_expert in (("", True), ("shared_", False)):
        projections = {
            projection_kind: getattr(layer, f"cortex_{prefix}{projection_kind}")["default"]
            for projection_kind in ("gate", "up", "down")
        }
        with torch.no_grad():
            for projection in projections.values():
                for parameter in projection.parameters():
                    parameter.fill_(float("nan"))
        initialize_branch(layer, "default", config, prefix=prefix, per_expert=per_expert)
        for projection_kind, projection in projections.items():
            assert all(torch.isfinite(parameter).all() for parameter in projection.parameters())
            if projection_kind == "down" and init is not False:
                assert torch.count_nonzero(projection.weight) == 0
            elif init != "sample":
                bound = projection.weight.shape[-1] ** -0.5
                assert projection.weight.abs().max() <= bound
                assert torch.count_nonzero(projection.weight) > 0
            if init == "random" and bias_init == "uniform" and projection_kind != "down":
                assert 0 < projection.bias.abs().max() <= projection.weight.shape[-1] ** -0.5
                continue
            expected_bias = 0.0
            if init == "sample" and projection_kind + "_bias" in source:
                expected_bias = 0.375 if projection_kind == "gate" else 0.75
            torch.testing.assert_close(
                projection.bias,
                torch.full_like(projection.bias, expected_bias),
                rtol=0,
                atol=0,
            )

    tokens = torch.randint(3, 32, (1, 5))
    actual = model(tokens).logits
    assert torch.isfinite(actual).all()
    if init is not False:
        with model.disable_adapter():
            expected = model(tokens).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unmerge_restores_only_dimension_metadata_changed_by_merge():
    class CustomFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 12, bias=False)
            self.up_proj = nn.Linear(4, 12, bias=False)
            self.down_proj = nn.Linear(12, 4, bias=False)
            self.act_fn = nn.SiLU()
            # Custom metadata need not be the tensor dimensions. Save its actual
            # original value when merge changes it instead of inferring a value.
            self.intermediate_dim, self.intermediate_size = 101, 102
            for index, projection in enumerate((self.gate_proj, self.up_proj, self.down_proj)):
                projection.in_features = 200 + 2 * index
                projection.out_features = 201 + 2 * index
            # Merge does not write these unrelated attributes.
            self.in_features = 301
            self.act_fn.intermediate_size = 302
            self.up_proj.intermediate_dim = 303

        def forward(self, inputs):
            return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))

    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = CustomFFN()

        def forward(self, inputs):
            return self.mlp(inputs)

    base = CustomModel().eval()
    ffn = base.mlp
    projections = (ffn.gate_proj, ffn.up_proj, ffn.down_proj)
    original_dimensions = [(projection.in_features, projection.out_features) for projection in projections]
    original_weights = {name: parameter.detach().clone() for name, parameter in ffn.named_parameters()}
    model = get_peft_model(base, CortexConfig(k=3, target_modules=["mlp"], init_weights=False)).eval()
    inputs = torch.randn(2, 4)
    expected = model(inputs)
    model.merge_adapter(safe_merge=True)
    assert ffn.intermediate_dim == ffn.intermediate_size == 15
    assert ffn.down_proj.weight.shape == (4, 15)
    torch.testing.assert_close(model(inputs), expected)

    ffn.in_features = 401
    ffn.act_fn.intermediate_size = 402
    ffn.up_proj.intermediate_dim = 403
    model.unmerge_adapter()

    assert (ffn.intermediate_dim, ffn.intermediate_size) == (101, 102)
    assert [(projection.in_features, projection.out_features) for projection in projections] == original_dimensions
    for name, parameter in ffn.named_parameters():
        torch.testing.assert_close(parameter, original_weights[name], rtol=0, atol=0)
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    assert ffn.in_features == 401
    assert ffn.act_fn.intermediate_size == 402
    assert ffn.up_proj.intermediate_dim == 403


def _module_name_boundary_model(nested=False):
    class CustomFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 12, bias=False)
            self.up_proj = nn.Linear(4, 12, bias=False)
            self.down_proj = nn.Linear(12, 4, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, inputs):
            return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))

    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            if nested:
                self.nested = nn.ModuleDict({"foo_mlp": nn.Sequential(CustomFFN())})
            else:
                self.foo_mlp = CustomFFN()
            self.other_ffn = CustomFFN()
            self.mlp = nn.Linear(4, 4)

        def forward(self, inputs):
            branch = self.nested["foo_mlp"] if nested else self.foo_mlp
            return branch(inputs) + self.other_ffn(inputs) + self.mlp(inputs)

    return CustomModel().eval(), "nested.foo_mlp.0" if nested else "foo_mlp"


def test_modules_to_save_matches_module_names_without_matching_character_suffixes(tmp_path):
    clean, target = _module_name_boundary_model()
    model = get_peft_model(deepcopy(clean), CortexConfig(k=2, target_modules=[target])).eval()
    base = model.get_base_model()
    old_layer, original_mlp = base.foo_mlp, base.mlp
    model.add_adapter("b", CortexConfig(k=3, target_modules=[target], modules_to_save=["mlp"]))
    assert base.foo_mlp is old_layer
    assert {name for name, module in base.named_modules() if isinstance(module, ModulesToSaveWrapper)} == {"mlp"}
    assert base.mlp.original_module is original_mlp
    assert set(base.mlp.modules_to_save) == {"b"}
    assert not any(isinstance(module, CortexLayer) for module in base.mlp.modules())
    assert model.peft_config["b"].target_roles == {target: "dense"}
    activate_branches(model)
    with torch.no_grad():
        base.mlp.modules_to_save["b"].weight.add_(0.25)
        base.mlp.modules_to_save["b"].bias.add_(0.1)
    inputs = torch.randn(3, 4)
    baseline = clean(inputs)
    expected = {}
    for adapter in ("default", "b"):
        model.set_adapter(adapter)
        expected[adapter] = model(inputs).detach()
        assert not torch.allclose(expected[adapter], baseline)
    assert not torch.allclose(expected["default"], expected["b"])

    for adapter in ("default", "b"):
        expected_state = get_peft_model_state_dict(model, adapter_name=adapter, save_embedding_layers=False)
        folder = tmp_path / adapter
        model.save_pretrained(folder, selected_adapters=[adapter], save_embedding_layers=False)
        checkpoint = folder if adapter == "default" else folder / adapter
        loaded = PeftModel.from_pretrained(deepcopy(clean), checkpoint, adapter_name=adapter).eval()
        restored_state = get_peft_model_state_dict(loaded, adapter_name=adapter, save_embedding_layers=False)
        assert restored_state.keys() == expected_state.keys()
        for name, value in restored_state.items():
            torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
        torch.testing.assert_close(loaded(inputs), expected[adapter], rtol=0, atol=0)


@pytest.mark.parametrize("nested", [False, True], ids=["exact-ffn", "dotted-ancestor-suffix"])
def test_modules_to_save_rejects_actual_cortex_owner_before_mutation(nested):
    clean, target = _module_name_boundary_model(nested)
    old_config = CortexConfig(k=2, target_modules=[target])
    model = get_peft_model(clean, old_config).eval()
    activate_branches(model)
    inputs = torch.randn(3, 4)
    result = model(inputs)
    expected = result.detach()
    result.square().sum().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())

    # The new adapter has a legitimate target which must not be modified before
    # discovering the forbidden saved module, matched exactly or after a dot.
    config = CortexConfig(k=3, target_modules=["other_ffn"], modules_to_save=["foo_mlp"])
    configs = (deepcopy(old_config.to_dict()), deepcopy(config.to_dict()))
    modules, parameters = list(model.modules()), list(model.parameters())
    trainable = [parameter.requires_grad for parameter in parameters]
    gradients = [None if parameter.grad is None else parameter.grad.clone() for parameter in parameters]
    state = deepcopy(model.state_dict())
    active = model.active_adapter
    active_layers = {
        name: list(layer.active_adapters) for name, layer in model.named_modules() if isinstance(layer, CortexLayer)
    }

    with pytest.raises(ValueError, match="modules_to_save.*existing Cortex"):
        model.add_adapter("b", config)

    assert set(model.peft_config) == {"default"}
    assert model.peft_config["default"] is old_config
    assert old_config.to_dict() == configs[0]
    assert config.to_dict() == configs[1]
    assert model.active_adapter == active
    assert {
        name: list(layer.active_adapters) for name, layer in model.named_modules() if isinstance(layer, CortexLayer)
    } == active_layers
    assert len(list(model.modules())) == len(modules)
    assert all(actual is before for actual, before in zip(model.modules(), modules))
    assert len(list(model.parameters())) == len(parameters)
    assert all(actual is before for actual, before in zip(model.parameters(), parameters))
    assert [parameter.requires_grad for parameter in model.parameters()] == trainable
    for parameter, gradient in zip(model.parameters(), gradients):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert model.state_dict().keys() == state.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
