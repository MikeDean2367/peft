"""Automatic bias layouts and initialization through the public Cortex API."""

import json
import math
from copy import deepcopy

import pytest
import torch
from torch import nn

from peft import CortexConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from peft.tuners.cortex.layer import CortexLayer, CortexMLP
from peft.tuners.cortex.structure import MlpStructure

from .test_cortex import tiny_model


class BiasTestFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 12, bias=False)
        self.up_proj = nn.Linear(4, 12, bias=False)
        self.down_proj = nn.Linear(12, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, inputs):
        return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))


class BiasTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = BiasTestFFN()

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


@pytest.mark.parametrize("native_biases", [(False, False, False), (True, True, True), (True, False, True)])
@pytest.mark.parametrize("override", [None, False, True])
@pytest.mark.parametrize("bias_init", ["uniform", "zeros"])
def test_bias_layout_overrides_initial_output_and_gradients(native_biases, override, bias_init):
    base = BiasTestModel()
    for kind, present in zip(("gate", "up", "down"), native_biases):
        projection = getattr(base.mlp, kind + "_proj")
        if present:
            projection.bias = nn.Parameter(torch.full((projection.out_features,), 0.25))
    reference = deepcopy(base)
    config = CortexConfig(k=7, target_modules=["mlp"], adapter_bias=override, adapter_bias_init=bias_init)
    model = get_peft_model(base, config)
    layer = model.get_base_model().mlp
    for kind, native_bias in zip(("gate", "up", "down"), native_biases):
        projection = getattr(layer, "cortex_" + kind)["default"]
        assert (projection.bias is not None) == (native_bias if override is None else override)
        if projection.bias is not None:
            if kind == "down" or bias_init == "zeros":
                assert torch.count_nonzero(projection.bias) == 0
            else:
                assert 0 < projection.bias.abs().max() <= 1 / math.sqrt(4)
    inputs = torch.randn(2, 4)
    actual = model(inputs)
    torch.testing.assert_close(actual, reference(inputs), rtol=0, atol=0)
    actual.sum().backward()
    down = layer.cortex_down["default"]
    assert torch.isfinite(down.weight.grad).all() and down.weight.grad.abs().sum() > 0
    for kind in ("gate", "up"):
        projection = getattr(layer, "cortex_" + kind)["default"]
        assert torch.count_nonzero(projection.weight.grad) == 0
        if projection.bias is not None:
            assert torch.count_nonzero(projection.bias.grad) == 0


@pytest.mark.parametrize("kind", ["gpt_oss", "qwen3_moe"])
@pytest.mark.parametrize("mode", ["shared", "routed", ["shared", "routed"]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.single_gpu_tests,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
            ],
        ),
    ],
)
def test_default_biases_follow_native_experts_in_every_branch(kind, mode, dtype, device):
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("Native BF16 required")
    base = tiny_model(kind).to(device=device, dtype=dtype).eval()
    inputs = torch.tensor([[1, 2, 3]], device=device)
    with torch.no_grad():
        expected = base(inputs).logits
    model = get_peft_model(base, CortexConfig(k=7, moe_target=mode), autocast_adapter_dtype=False).eval()
    for layer in model.modules():
        if not isinstance(layer, CortexLayer):
            continue
        for prefix in ("", "shared_") if mode == ["shared", "routed"] else ("",):
            for projection_kind in ("gate", "up", "down"):
                projection = getattr(layer, f"cortex_{prefix}{projection_kind}")["default"]
                assert (projection.bias is not None) == (kind == "gpt_oss")
                if projection_kind == "down":
                    assert torch.count_nonzero(projection.weight) == 0
                    if projection.bias is not None:
                        assert torch.count_nonzero(projection.bias) == 0
                elif projection.bias is not None:
                    assert 0 < projection.bias.abs().max() <= 1 / math.sqrt(16)
    torch.testing.assert_close(model(inputs).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_auto_bias_handles_fused_and_nongated_projections_without_reading_values(fused, device):
    base = nn.Module()
    setattr(base, "gate_up_proj" if fused else "up_proj", nn.Linear(8, 24 if fused else 12, device=device))
    base.down_proj = nn.Linear(12, 8, bias=False, device=device)
    spec = MlpStructure(8, 8, 12, fused=fused, has_gate=fused, act_fn=nn.SiLU())
    layer = CortexMLP(base, "default", CortexConfig(k=3), spec)
    assert layer.cortex_up["default"].bias is not None
    assert layer.cortex_down["default"].bias is None
    if fused:
        assert layer.cortex_gate["default"].bias is not None
    else:
        assert not layer.cortex_gate


@pytest.mark.parametrize("kind", ["qwen3", "gpt_oss"])
@pytest.mark.parametrize("low_cpu_mem_usage", [False, True])
def test_auto_bias_checkpoint_reload_and_merge_preserve_trained_values(kind, low_cpu_mem_usage, tmp_path):
    base = tiny_model(kind).eval()
    clean = deepcopy(base)
    model = get_peft_model(base, CortexConfig(k=3, moe_target=["shared", "routed"])).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "cortex_" in name:
                parameter.uniform_(-0.2, 0.2)
    tokens = torch.tensor([[1, 2, 3, 4]])
    expected = model(tokens).logits.detach()
    assert not torch.equal(expected, clean(tokens).logits)
    state = get_peft_model_state_dict(model, save_embedding_layers=False)
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    saved = json.loads((tmp_path / "adapter_config.json").read_text())
    assert saved["adapter_bias"] is None and saved["adapter_bias_init"] == "uniform"
    restored = PeftModel.from_pretrained(clean, tmp_path, low_cpu_mem_usage=low_cpu_mem_usage).eval()
    restored_state = get_peft_model_state_dict(restored, save_embedding_layers=False)
    assert restored_state.keys() == state.keys()
    for name, value in state.items():
        torch.testing.assert_close(restored_state[name], value, rtol=0, atol=0)
    torch.testing.assert_close(restored(tokens).logits, expected, rtol=0, atol=0)
    merged = restored.merge_and_unload(safe_merge=True).eval()
    torch.testing.assert_close(merged(tokens).logits, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("value", [0, 1, "auto", "true"])
def test_invalid_bias_override_is_rejected(value):
    with pytest.raises(TypeError, match="adapter_bias"):
        CortexConfig(adapter_bias=value)


@pytest.mark.parametrize("value", [None, True, "normal"])
def test_invalid_bias_initializer_is_rejected(value):
    with pytest.raises(ValueError, match="adapter_bias_init"):
        CortexConfig(adapter_bias_init=value)


def test_uniform_bias_initialization_repeats_with_the_same_seed():
    base = tiny_model("gpt_oss")
    states = []
    for _ in range(2):
        torch.manual_seed(1234)
        model = get_peft_model(deepcopy(base), CortexConfig(k=7, init_weights=True))
        states.append(get_peft_model_state_dict(model, save_embedding_layers=False))
    for name, value in states[0].items():
        torch.testing.assert_close(states[1][name], value, rtol=0, atol=0)
