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

from functools import partial

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

from peft import CortexConfig, get_peft_model
from peft.tuners.cortex.layer import CortexLayer

from .test_cortex import BlockModel, activate_branches, tiny_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    torch.manual_seed(123)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize(
    "kind,mode",
    [
        ("qwen3", "routed"),
        ("gemma3_text", "routed"),
        ("gpt_oss", "shared"),
        ("gpt_oss", "routed"),
        ("gpt_oss", ["shared", "routed"]),
        ("qwen3_next", ["shared", "routed"]),
    ],
)
def test_partial_merge_keeps_unmerged_active_adapter_output_and_gradients(kind, mode):
    pytest.importorskip("transformers", minversion="5.17.0")
    base = BlockModel(kind).eval() if kind == "qwen3_next" else tiny_model(kind).eval()
    model = get_peft_model(base, CortexConfig(k=3, moe_target=mode)).eval()
    model.add_adapter("second", CortexConfig(k=5, moe_target=mode))
    activate_branches(model)
    model.base_model.set_adapter(["default", "second"])
    inputs = torch.randn(2, 5, 16) if kind == "qwen3_next" else torch.randint(3, 32, (2, 5))

    def output():
        result = model(inputs) if kind == "qwen3_next" else model(inputs, use_cache=False)
        return result if kind == "qwen3_next" else result.logits

    second_parameters = [
        parameter for name, parameter in model.named_parameters() if "cortex_" in name and ".second." in name
    ]
    expected = output()
    expected_grads = torch.autograd.grad(expected.square().mean(), second_parameters)
    model.merge_adapter(adapter_names=["default"], safe_merge=True)
    actual = output()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
    actual_grads = torch.autograd.grad(actual.square().mean(), second_parameters)
    assert any(gradient.abs().sum() > 0 for gradient in actual_grads)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-4, atol=5e-5)
    for layer in model.modules():
        if isinstance(layer, CortexLayer):
            assert layer.active_adapters == ["default", "second"]
            assert layer.merged_adapters == ["default"]
    model.merge_adapter(adapter_names=["second"], safe_merge=True)
    torch.testing.assert_close(output(), expected, rtol=2e-4, atol=3e-5)
    model.unmerge_adapter()
    torch.testing.assert_close(output(), expected, rtol=0, atol=0)


class CustomFFN(nn.Module):
    def __init__(self, activation, gated=True):
        super().__init__()
        self.up_proj = nn.Linear(4, 12, bias=False)
        self.down_proj = nn.Linear(12, 4, bias=False)
        if gated:
            self.gate_proj = nn.Linear(4, 12, bias=False)
        self.act_fn = activation

    def forward(self, x):
        up = self.up_proj(x)
        hidden = self.act_fn(self.gate_proj(x)) * up if hasattr(self, "gate_proj") else self.act_fn(up)
        return self.down_proj(hidden)


class CustomModel(nn.Module):
    def __init__(self, activation, gated=True):
        super().__init__()
        self.layers = nn.Sequential(CustomFFN(nn.SiLU(), gated), CustomFFN(activation, gated))

    def forward(self, x):
        return self.layers(x)


class CoupledSiLU(nn.SiLU):
    """A familiar activation base class does not establish elementwise behavior."""

    def forward(self, x):
        return F.softmax(x, dim=-1)


@pytest.mark.parametrize(
    "activation",
    [
        nn.Softmax(dim=-1),
        nn.LayerNorm(12, elementwise_affine=False),
        nn.Dropout(0.1),
        partial(F.softmax, dim=-1),
        CoupledSiLU(),
    ],
    ids=["softmax", "layernorm", "dropout", "softmax-function", "overridden-silu"],
)
def test_unaudited_activations_fail_before_any_target_is_replaced(activation):
    base = CustomModel(activation).eval()
    original_parameters = list(base.parameters())
    inputs = torch.randn(2, 4)
    expected = base(inputs).detach()
    with pytest.raises(ValueError, match="audited deterministic elementwise activation"):
        get_peft_model(base, CortexConfig(k=3, target_modules=["layers.0", "layers.1"]))
    assert not any(isinstance(layer, CortexLayer) for layer in base.modules())
    assert all(parameter is original for parameter, original in zip(base.parameters(), original_parameters))
    torch.testing.assert_close(base(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("gated", [True, False], ids=["gated", "non-gated"])
@pytest.mark.parametrize(
    "activation",
    [
        nn.SiLU(),
        F.silu,
        nn.GELU(approximate="tanh"),
        F.gelu,
        nn.ReLU(),
        nn.Identity(),
        ACT2FN["gelu_new"],
        ACT2FN["silu"],
    ],
    ids=[
        "silu",
        "silu-function",
        "gelu-tanh",
        "gelu-function",
        "relu",
        "identity",
        "transformers-gelu",
        "transformers-silu",
    ],
)
def test_elementwise_custom_ffn_merge_preserves_output_and_input_gradient(activation, gated):
    model = get_peft_model(
        CustomModel(activation, gated).eval(),
        CortexConfig(k=3, target_modules=["layers.1"], init_weights=False),
    ).eval()
    inputs = torch.randn(2, 4, requires_grad=True)
    expected = model(inputs)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)[0]
    native = model.merge_and_unload(safe_merge=True)
    actual = native(inputs)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
