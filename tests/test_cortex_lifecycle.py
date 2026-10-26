from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from peft import CortexConfig, PeftConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from peft.tuners.cortex.layer import CortexExperts, CortexLayer, CortexMLP
from peft.tuners.cortex.structure import describe_mlp

from .test_cortex import activate_branches, tiny_model


CASES = [
    ("llama", "shared"),
    ("qwen2", "shared"),
    ("qwen3", "shared"),
    ("qwen3_moe", "shared"),
    ("qwen3_moe", "routed"),
    ("gpt_oss", "shared"),
    ("gpt_oss", "routed"),
]


@pytest.fixture(autouse=True)
def deterministic_cpu():
    pytest.importorskip("transformers", minversion="5.17.0")
    previous_threads = torch.get_num_threads()
    torch.manual_seed(123)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def cortex_config(mode="shared", **kwargs):
    return CortexConfig(k=3, moe_target=mode, task_type="CAUSAL_LM", **kwargs)


@pytest.mark.parametrize("kind,mode", CASES)
def test_train_disable_save_reload_and_generate(kind, mode, tmp_path):
    base = tiny_model(kind).eval()
    clean = deepcopy(base)
    base_parameters = [(p, p.detach().clone()) for p in base.parameters()]
    tokens = torch.randint(3, 32, (2, 6))
    with torch.no_grad():
        reference = base(tokens, use_cache=False).logits
    model = get_peft_model(base, cortex_config(mode, adapter_bias=True)).eval()
    torch.testing.assert_close(model(tokens, use_cache=False).logits, reference, rtol=0, atol=0)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(not p.requires_grad for p, _ in base_parameters)
    optimizer = torch.optim.AdamW(trainable, lr=0.03)
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        # Checks the gradient path through later frozen base layers.
        first = next(m for m in model.modules() if isinstance(m, CortexLayer))
        assert first.cortex_down["default"].weight.grad.abs().sum() > 0
        optimizer.step()
    for parameter, before in base_parameters:
        assert parameter.grad is None
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    model.eval()
    expected = model(tokens, use_cache=False).logits.detach()
    assert not torch.allclose(expected, reference)
    with model.disable_adapter():
        torch.testing.assert_close(model(tokens, use_cache=False).logits, reference, rtol=0, atol=0)
    state = get_peft_model_state_dict(model, save_embedding_layers=False)
    assert state and all("cortex_" in key and ".default." not in key for key in state)
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    assert isinstance(PeftConfig.from_pretrained(tmp_path), CortexConfig)
    restored = PeftModel.from_pretrained(clean, tmp_path, adapter_name="renamed").eval()
    assert not any(p.requires_grad for p in restored.parameters())
    torch.testing.assert_close(restored(tokens, use_cache=False).logits, expected)
    assert restored.generate(tokens[:1, :2], max_new_tokens=2, do_sample=False).shape[1] <= 4


@pytest.mark.parametrize("kind,mode", CASES)
def test_multiple_adapters_save_only_selected_and_unload(kind, mode, tmp_path):
    original = tiny_model(kind).eval()
    clean = deepcopy(original)
    model = get_peft_model(original, cortex_config(mode)).eval()
    activate_branches(model)
    tokens = torch.randint(3, 32, (1, 5))
    first = model(tokens, use_cache=False).logits.detach()
    model.add_adapter("second", CortexConfig(k=5, moe_target=mode, task_type="CAUSAL_LM"))
    model.set_adapter("second")
    torch.testing.assert_close(model(tokens, use_cache=False).logits, clean(tokens, use_cache=False).logits)
    model.set_adapter("default")
    torch.testing.assert_close(model(tokens, use_cache=False).logits, first)
    state = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    assert all("second" not in key and 5 not in tensor.shape for key, tensor in state.items())
    model.save_pretrained(tmp_path, selected_adapters=["default"], save_embedding_layers=False)
    restored = PeftModel.from_pretrained(deepcopy(clean), tmp_path).eval()
    torch.testing.assert_close(restored(tokens, use_cache=False).logits, first)
    unloaded = model.unload().eval()
    assert not any(isinstance(m, CortexLayer) for m in unloaded.modules())
    torch.testing.assert_close(unloaded(tokens, use_cache=False).logits, clean(tokens, use_cache=False).logits)


@pytest.mark.parametrize("kind", ["qwen3_moe", "gpt_oss"])
@pytest.mark.parametrize("mode", ["shared", "routed"])
def test_moe_matches_explicit_mathematical_reference_and_gradients(kind, mode):
    block = tiny_model(kind).model.layers[0].mlp
    spec = describe_mlp(block)
    config = cortex_config(mode, adapter_bias=True)
    if mode == "shared":
        layer = CortexMLP(block, "default", config, spec)
    else:
        layer = CortexExperts(block.experts, "default", config, spec, role="routed")
    # Large inputs exercise GPT-OSS's clipping, not just the near-zero regime.
    x = (torch.randn(5, 16) * 20).requires_grad_()
    indices = torch.tensor([[0, 2], [1, 2], [2, 0], [1, 0], [2, 1]])
    weights = torch.tensor([[0.2, 0.3], [0.7, 0.1], [0.4, 0.5], [0.6, 0.2], [0.1, 0.8]], requires_grad=True)
    activate_branches(layer)
    gate, up, down = [getattr(layer, "cortex_" + name)["default"] for name in ("gate", "up", "down")]

    def explicit_branch(value, expert=None):
        def linear(proj, tensor):
            w = proj.weight if expert is None else proj.weight[expert]
            b = proj.bias if expert is None else proj.bias[expert]
            return F.linear(tensor, w, b)

        g, u = linear(gate, value), linear(up, value)
        if kind == "gpt_oss":
            g, u = g.clamp(max=7.0), u.clamp(-7.0, 7.0)
            hidden = g * torch.sigmoid(1.702 * g) * (u + 1)
        else:
            hidden = F.silu(g) * u
        return linear(down, hidden)

    if mode == "shared":
        actual_output = layer(x.unsqueeze(0))
        base_output = block(x.unsqueeze(0))
        actual = actual_output[0] if isinstance(actual_output, tuple) else actual_output
        base_value = base_output[0] if isinstance(base_output, tuple) else base_output
        if isinstance(base_output, tuple):
            torch.testing.assert_close(actual_output[1], base_output[1], rtol=0, atol=0)
        expected = base_value + explicit_branch(x).unsqueeze(0)
        parameters = (x, down.weight)
    else:
        actual = layer(x, indices, weights)
        correction = torch.stack(
            [sum(weights[t, j] * explicit_branch(x[t], int(indices[t, j])) for j in range(2)) for t in range(5)]
        )
        expected = block.experts(x, indices, weights) + correction
        parameters = (x, down.weight, weights)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual_grads = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().mean(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_width(value):
    with pytest.raises(ValueError, match="positive integer"):
        CortexConfig(k=value)


def test_invalid_targets_fail_before_replacement():
    model = tiny_model("qwen3")
    with pytest.raises(ValueError, match="Unsupported Cortex target"):
        get_peft_model(model, cortex_config(target_modules=["mlp", "q_proj"]))
    assert not any(isinstance(m, CortexLayer) for m in model.modules())


def test_bfloat16_placement_and_adapter_autocast():
    model = get_peft_model(tiny_model("qwen3").to(torch.bfloat16), cortex_config())
    for module in model.modules():
        if isinstance(module, CortexLayer):
            assert module.cortex_gate["default"].weight.dtype == torch.float32
            assert next(module.base_layer.parameters()).dtype == torch.bfloat16
    tokens = torch.randint(3, 32, (1, 5))
    model(tokens, labels=tokens, use_cache=False).loss.backward()


def test_exclude_modules_and_checkpointed_training():
    model = get_peft_model(tiny_model("qwen3"), cortex_config(exclude_modules=["model.layers.1.mlp"]))
    assert sum(isinstance(m, CortexLayer) for m in model.modules()) == 1
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokens = torch.randint(3, 32, (1, 5))
    model(tokens, labels=tokens, use_cache=False).loss.backward()
    first = next(m for m in model.modules() if isinstance(m, CortexLayer))
    assert first.cortex_down["default"].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["shared", "routed"])
def test_low_memory_reload_and_loading_inactive_adapter(mode, tmp_path):
    base = tiny_model("qwen3_moe")
    clean = deepcopy(base)
    model = get_peft_model(base, cortex_config(mode)).eval()
    activate_branches(model)
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    tokens = torch.randint(3, 32, (1, 5))
    expected = model(tokens, use_cache=False).logits
    loaded = PeftModel.from_pretrained(clean, tmp_path, low_cpu_mem_usage=True).eval()
    assert not any(p.is_meta for p in loaded.parameters())
    torch.testing.assert_close(loaded(tokens, use_cache=False).logits, expected)
    model.load_adapter(tmp_path, "loaded", is_trainable=False)
    model.set_adapter("loaded")
    torch.testing.assert_close(model(tokens, use_cache=False).logits, expected)
    model.delete_adapter("default")
    torch.testing.assert_close(model(tokens, use_cache=False).logits, expected)


@pytest.mark.parametrize("kind", ["qwen3_moe", "gpt_oss"])
def test_state_dict_driven_direct_injection(kind):
    from peft import inject_adapter_in_model, set_peft_model_state_dict

    base = tiny_model(kind).eval()
    clean = deepcopy(base)
    model = get_peft_model(base, cortex_config("routed")).eval()
    activate_branches(model)
    state = get_peft_model_state_dict(model.get_base_model(), save_embedding_layers=False)
    config = deepcopy(model.peft_config["default"])
    restored = inject_adapter_in_model(config, clean, adapter_name="restored", state_dict=state)
    result = set_peft_model_state_dict(restored, state, adapter_name="restored")
    assert not result.unexpected_keys
    tokens = torch.randint(3, 32, (1, 5))
    torch.testing.assert_close(restored(tokens, use_cache=False).logits, model(tokens, use_cache=False).logits)
