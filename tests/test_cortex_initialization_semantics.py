"""Check which initialization choices affect identity and first-step learning.

These tests use the current implementation unchanged. They show that a zero down projection preserves the starting
model for different finite upstream features, while those features still change the first down-weight gradient. They do
not assert identical learning curves or old/new numerical execution.
"""

import pytest
import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from peft import CortexConfig, get_peft_model


CASES = [
    ("qwen3", "dense"),
    ("qwen3_moe", "shared"),
    ("qwen3_moe", "routed"),
    ("gpt_oss", "shared"),
    ("gpt_oss", "routed"),
]


class NativeFFNModel(nn.Module):
    """Keep the native FFN/router API without requiring the research checkout."""

    def __init__(self, mlp, config):
        super().__init__()
        self.mlp, self.config = mlp, config

    def forward(self, inputs):
        return self.mlp(inputs)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(917)
        yield
    torch.set_num_threads(previous_threads)


def configured_branch(kind, role, dtype, init="random"):
    config = AutoConfig.for_model(
        kind,
        hidden_size=8,
        intermediate_size=12,
        moe_intermediate_size=12,
        num_experts=3,
        num_local_experts=3,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
    )
    config._experts_implementation = "eager"
    cls = {"qwen3": Qwen3MLP, "gpt_oss": GptOssMLP, "qwen3_moe": Qwen3MoeSparseMoeBlock}[kind]
    mlp = cls(config).to(dtype)
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.normal_(std=0.2)
    current = get_peft_model(
        NativeFFNModel(mlp, config),
        CortexConfig(
            k=3,
            moe_target=role if role != "dense" else "routed",
            adapter_bias=True,
            adapter_bias_init="zeros",
            init_weights=init,
        ),
        autocast_adapter_dtype=False,
    ).eval()
    layer = current.get_base_model().mlp
    if kind != "qwen3":
        layer = layer.experts
    return current, layer


def hidden_output(model, inputs):
    # Native OSS additionally returns router logits; this contract concerns
    # the FFN output and leaves those auxiliary outputs unchanged.
    result = model(inputs)
    return result[0] if isinstance(result, tuple) else result


@pytest.mark.parametrize("kind,role", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("init", ["random", "sample"])
def test_zero_down_preserves_base_and_learns_with_nonzero_upstream_bias(kind, role, dtype, init):
    current, layer = configured_branch(kind, role, dtype, init)
    # Both zero-biased and random-biased gate/up can satisfy the starting-model
    # contract. Use nonzero finite biases to exercise the stronger case.
    with torch.no_grad():
        layer.cortex_gate["default"].bias.fill_(0.35)
        layer.cortex_up["default"].bias.fill_(0.7)
    down = layer.cortex_down["default"]
    assert torch.count_nonzero(down.weight) == torch.count_nonzero(down.bias) == 0
    inputs = torch.randn(2, 5, 8, dtype=dtype)
    with current.disable_adapter():
        expected = hidden_output(current, inputs).detach()
    actual = hidden_output(current, inputs)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    # At W_down=0, data gradients reach down, but gate/up have zero data
    # gradients. Optimizer weight decay is separate from this assertion.
    for projection_kind in ("gate", "up"):
        projection = getattr(layer, f"cortex_{projection_kind}")["default"]
        assert torch.count_nonzero(projection.weight.grad) == 0
        assert torch.count_nonzero(projection.bias.grad) == 0
    assert torch.isfinite(down.weight.grad).all() and torch.isfinite(down.bias.grad).all()
    assert down.weight.grad.abs().sum() > 0
    assert down.bias.grad.abs().sum() > 0


@pytest.mark.parametrize("kind,role", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_upstream_bias_changes_first_down_gradient_without_changing_starting_output(kind, role, dtype):
    current, layer = configured_branch(kind, role, dtype)
    gate = layer.cortex_gate["default"]
    up = layer.cortex_up["default"]
    down = layer.cortex_down["default"]
    assert torch.count_nonzero(gate.bias) == torch.count_nonzero(up.bias) == 0
    inputs = torch.randn(2, 5, 8, dtype=dtype)
    original = hidden_output(current, inputs)
    gradient_zero_bias = torch.autograd.grad(original.float().sum(), down.weight)[0]
    with torch.no_grad():
        gate.bias.fill_(0.35)
        up.bias.fill_(0.7)
    shifted = hidden_output(current, inputs)
    gradient_shifted_bias = torch.autograd.grad(shifted.float().sum(), down.weight)[0]
    torch.testing.assert_close(original, shifted, rtol=0, atol=0)
    assert torch.isfinite(gradient_zero_bias).all() and torch.isfinite(gradient_shifted_bias).all()
    assert (gradient_shifted_bias != gradient_zero_bias).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_zero_down_weight_alone_does_not_preserve_base_when_down_bias_is_nonzero(dtype):
    current, layer = configured_branch("qwen3", "dense", dtype)
    down = layer.cortex_down["default"]
    assert torch.count_nonzero(down.weight) == 0
    with torch.no_grad():
        down.bias.fill_(0.5)
    inputs = torch.randn(2, 5, 8, dtype=dtype)
    with current.disable_adapter():
        expected = hidden_output(current, inputs).detach()
    actual = hidden_output(current, inputs)
    torch.testing.assert_close(actual, expected + 0.5, rtol=0, atol=0)
    assert (actual != expected).any()
