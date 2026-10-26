"""Native Gemma3 discovery, initialization and PEFT checkpoint/export checks."""

from copy import deepcopy

import pytest
import torch
from transformers import AutoModelForCausalLM

from peft import CortexConfig, PeftModel, get_peft_model
from peft.tuners.cortex.layer import CortexLayer

from .test_cortex import activate_branches, tiny_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        yield
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("init", ["random", "sample"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_gemma3_auto_discovery_bias_and_initialization_preserve_native_output(init, bias, dtype):
    base = tiny_model("gemma3_text").to(dtype).eval()
    tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]])
    expected = base(tokens, use_cache=False).logits.detach()
    model = get_peft_model(
        base,
        CortexConfig(k=3, init_weights=init, init_noise_std=0, adapter_bias=bias, task_type="CAUSAL_LM"),
        autocast_adapter_dtype=False,
    ).eval()
    assert model.peft_config["default"].target_roles == {
        "model.layers.0.mlp": "dense",
        "model.layers.1.mlp": "dense",
    }
    layers = [layer for layer in model.modules() if isinstance(layer, CortexLayer)]
    assert len(layers) == 2
    for layer in layers:
        assert type(layer.base_layer).__name__ == "Gemma3MLP"
        for projection_kind in ("gate", "up", "down"):
            projection = getattr(layer, f"cortex_{projection_kind}")["default"]
            assert projection.weight.dtype == dtype
            assert (projection.bias is not None) == bias
            if projection.bias is not None:
                if init == "random" and projection_kind != "down":
                    assert 0 < projection.bias.abs().max() <= projection.weight.shape[-1] ** -0.5
                else:
                    assert torch.count_nonzero(projection.bias) == 0
        assert torch.count_nonzero(layer.cortex_down["default"].weight) == 0
        if init == "sample":
            gate = layer.cortex_gate["default"].weight
            matches = (gate[:, None, :] == layer.base_layer.gate_proj.weight[None, :, :]).all(-1)
            assert (matches.sum(-1) == 1).all()
            indices = matches.int().argmax(-1)
            torch.testing.assert_close(
                layer.cortex_up["default"].weight,
                layer.base_layer.up_proj.weight[indices],
                rtol=0,
                atol=0,
            )
    actual = model(tokens, labels=tokens, use_cache=False)
    torch.testing.assert_close(actual.logits, expected, rtol=0, atol=0)
    assert torch.isfinite(actual.loss)
    actual.loss.backward()
    for layer in layers:
        gradient = layer.cortex_down["default"].weight.grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None for layer in layers for parameter in layer.base_layer.parameters())


@pytest.mark.parametrize("init", ["random", "sample"])
def test_gemma3_multiple_adapters_nonzero_gradients_checkpoint_and_native_export(init, tmp_path):
    base = tiny_model("gemma3_text").eval()
    clean = deepcopy(base)
    model = get_peft_model(base, CortexConfig(k=3, init_weights=init, task_type="CAUSAL_LM")).eval()
    model.add_adapter("second", CortexConfig(k=5, init_weights=init, task_type="CAUSAL_LM"))
    model.base_model.set_adapter(["default", "second"])
    activate_branches(model)
    tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]])
    actual = model(tokens, labels=tokens, use_cache=False)
    expected = actual.logits.detach()
    second = [p for name, p in model.named_parameters() if "cortex_" in name and ".second." in name]
    gradients = torch.autograd.grad(actual.loss, second)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    model.save_pretrained(tmp_path / "adapter", save_embedding_layers=False)
    loaded = PeftModel.from_pretrained(clean, tmp_path / "adapter", is_trainable=True)
    loaded.load_adapter(tmp_path / "adapter" / "second", "second", is_trainable=True)
    loaded.base_model.set_adapter(["default", "second"])
    loaded.eval()
    torch.testing.assert_close(loaded(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
    merged = model.merge_and_unload(safe_merge=True)
    assert merged.config.intermediate_size == 32
    torch.testing.assert_close(merged(tokens, use_cache=False).logits, expected, rtol=2e-4, atol=3e-5)
    merged.save_pretrained(tmp_path / "native")
    restored, info = AutoModelForCausalLM.from_pretrained(
        tmp_path / "native", output_loading_info=True, attn_implementation="eager"
    )
    assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    restored.eval()
    torch.testing.assert_close(restored(tokens, use_cache=False).logits, expected, rtol=2e-4, atol=3e-5)


def test_gemma3_partial_target_export_pads_untargeted_native_ffns(tmp_path):
    base = tiny_model("gemma3_text").eval()
    untargeted = deepcopy(base.model.layers[1].mlp.state_dict())
    model = get_peft_model(
        base, CortexConfig(k=5, init_weights="sample", target_modules=["model.layers.0.mlp"])
    ).eval()
    activate_branches(model)
    tokens = torch.tensor([[1, 4, 7, 3, 2]])
    expected = model(tokens, use_cache=False).logits.detach()
    merged = model.merge_and_unload(safe_merge=True)
    assert merged.config.intermediate_size == 29
    padded = merged.model.layers[1].mlp
    for kind in ("gate", "up"):
        projection = getattr(padded, f"{kind}_proj")
        torch.testing.assert_close(projection.weight[:24], untargeted[f"{kind}_proj.weight"], rtol=0, atol=0)
        assert torch.count_nonzero(projection.weight[24:]) == 0
    torch.testing.assert_close(padded.down_proj.weight[:, :24], untargeted["down_proj.weight"], rtol=0, atol=0)
    assert torch.count_nonzero(padded.down_proj.weight[:, 24:]) == 0
    merged.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation="eager").eval()
    torch.testing.assert_close(restored(tokens, use_cache=False).logits, expected, rtol=2e-4, atol=3e-5)
