"""Run the same native-model checks on CPU and CUDA; skipped CUDA is not a pass."""

from contextlib import contextmanager
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from peft import CortexConfig, PeftModel, get_peft_model
from peft.tuners.cortex.layer import CortexExperts, CortexLayer, CortexMLP
from peft.tuners.cortex.structure import describe_mlp

from .test_cortex import BlockModel, activate_branches, tiny_model


CASES = [
    ("gemma3_text", "routed"),
    ("qwen3", "routed"),
    ("qwen3_moe", "shared"),
    ("qwen3_moe", "routed"),
    ("qwen3_moe", ["shared", "routed"]),
    ("gpt_oss", "shared"),
    ("gpt_oss", "routed"),
    ("gpt_oss", ["shared", "routed"]),
]


@contextmanager
def device_runtime(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime unavailable; this test has not run on a GPU.")
    previous_threads = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(123)
            if devices:
                torch.cuda.manual_seed(123)
            yield device
            if devices:
                torch.cuda.synchronize(device)
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_num_threads(previous_threads)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.single_gpu_tests)])
def device(request):
    with device_runtime(request.param) as current:
        yield current


@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def dtype(request, device):
    if device.type == "cuda" and request.param == torch.bfloat16:
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            pytest.skip("This CUDA GPU does not support native BF16.")
    return request.param


@pytest.mark.parametrize("kind,mode", CASES)
@pytest.mark.parametrize("init", ["random", "sample"])
def test_device_initialization_placement_and_two_training_steps(kind, mode, init, device, dtype):
    base = tiny_model(kind).to(device=device, dtype=dtype).eval()
    tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]], device=device)
    with torch.no_grad():
        expected = base(tokens, use_cache=False).logits
    frozen = [(parameter, parameter.detach().clone()) for parameter in base.parameters()]
    model = get_peft_model(
        base,
        CortexConfig(k=3, moe_target=mode, init_weights=init, adapter_bias=True, task_type="CAUSAL_LM"),
        autocast_adapter_dtype=False,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable and all(parameter.device == tokens.device and parameter.dtype == dtype for parameter in trainable)
    for name, parameter in model.named_parameters():
        if "cortex_" in name and "down" in name:
            assert torch.count_nonzero(parameter) == 0
    torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(trainable, lr=0.03)
    model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = model(tokens, labels=tokens, use_cache=False)
        assert torch.isfinite(result.loss) and torch.isfinite(result.logits).all()
        result.loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                if step == 0 and "down" not in name:
                    assert torch.count_nonzero(parameter.grad) == 0
        downs = [
            parameter for name, parameter in model.named_parameters() if "down" in name and parameter.requires_grad
        ]
        assert any(parameter.grad.abs().sum() > 0 for parameter in downs)
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in trainable)
    for parameter, before in frozen:
        assert not parameter.requires_grad and parameter.grad is None
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    model.eval()
    with model.disable_adapter():
        torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind,mode", CASES)
def test_device_fp32_checkpoint_and_native_export_contract(kind, mode, device, tmp_path):
    base = tiny_model(kind).to(device).eval()
    clean = deepcopy(base)
    model = get_peft_model(
        base,
        CortexConfig(k=3, moe_target=mode, init_weights="sample", adapter_bias=kind == "gpt_oss"),
    ).eval()
    activate_branches(model)
    tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]], device=device)
    with torch.no_grad():
        expected = model(tokens, use_cache=False).logits
    model.save_pretrained(tmp_path / "adapter", save_embedding_layers=False)
    loaded = PeftModel.from_pretrained(clean, tmp_path / "adapter", adapter_name="renamed").eval()
    assert all(parameter.device == tokens.device for parameter in loaded.parameters())
    torch.testing.assert_close(loaded(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
    if kind == "qwen3_moe" and mode in ("shared", ["shared", "routed"]) and not base.config.norm_topk_prob:
        # Preserve the native routing policy. The default unnormalized public
        # branch can be checkpointed but cannot be folded into routed experts.
        with pytest.raises(ValueError, match="routing weights.*constant"):
            model.merge_adapter(safe_merge=True)
        assert not any(module.merged for module in model.modules() if isinstance(module, CortexLayer))
        torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
        return
    model.merge_adapter(safe_merge=True)
    torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=2e-4, atol=3e-5)
    model.unmerge_adapter()
    torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
    native = model.merge_and_unload(safe_merge=True)
    assert not any(isinstance(module, CortexLayer) for module in native.modules())
    native.save_pretrained(tmp_path / "native")
    restored, info = AutoModelForCausalLM.from_pretrained(
        tmp_path / "native", output_loading_info=True, attn_implementation="eager", experts_implementation="eager"
    )
    assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    restored.to(device).eval()
    torch.testing.assert_close(restored(tokens, use_cache=False).logits, expected, rtol=2e-4, atol=3e-5)


@pytest.mark.parametrize("kind,mode", CASES)
def test_device_bf16_base_fp32_adapter_migration_and_training(kind, mode, device):
    if device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("This CUDA GPU does not support native BF16.")
    # Exercise migration after injection as well as injection into a CUDA base.
    model = get_peft_model(
        tiny_model(kind).to(torch.bfloat16), CortexConfig(k=3, moe_target=mode, task_type="CAUSAL_LM")
    ).to(device)
    tokens = torch.tensor([[1, 4, 7, 3, 2], [1, 5, 8, 4, 2]], device=device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable and all(
        parameter.dtype == torch.float32 and parameter.device == tokens.device for parameter in trainable
    )
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters() if not parameter.requires_grad)
    result = model(tokens, labels=tokens, use_cache=False)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert any(parameter.grad.abs().sum() > 0 for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=0.01)
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in trainable)


@pytest.mark.parametrize("kind,mode", CASES)
def test_device_fp32_nonzero_formula_and_input_adapter_routing_gradients(kind, mode, device):
    block = tiny_model(kind).model.layers[0].mlp.to(device)
    dense = kind in {"gemma3_text", "qwen3"}
    base = block if dense else block.experts
    config = CortexConfig(k=3, adapter_bias=True)
    if dense:
        layer = CortexMLP(base, "default", config, describe_mlp(base))
    else:
        role = "routed_public" if isinstance(mode, list) else {"shared": "public", "routed": "routed"}[mode]
        layer = CortexExperts(base, "default", config, describe_mlp(base), role=role, public_scale=1.0)
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "cortex_" in name:
                parameter.normal_(std=0.2)
    inputs = (torch.randn(5, 16, device=device) * 20).requires_grad_()
    indices = torch.tensor([[0, 2], [1, 2], [2, 0], [1, 0], [2, 1]], device=device)
    weights = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.1, 0.9]], device=device)
    weights.requires_grad_()

    def branch(value, expert=None, prefix=""):
        def project(projection_kind, tensor):
            projection = getattr(layer, f"cortex_{prefix}{projection_kind}")["default"]
            weight = projection.weight if expert is None else projection.weight[expert]
            bias = projection.bias if expert is None else projection.bias[expert]
            return F.linear(tensor, weight, bias)

        gate, up = project("gate", value), project("up", value)
        if kind == "gpt_oss":
            gate, up = gate.clamp(max=7.0), up.clamp(-7.0, 7.0)
            hidden = gate * torch.sigmoid(gate * 1.702) * (up + 1)
        elif kind == "gemma3_text":
            hidden = F.gelu(gate, approximate="tanh") * up
        else:
            hidden = F.silu(gate) * up
        return project("down", hidden)

    if dense:
        actual = layer(inputs)
        expected = base(inputs) + branch(inputs)
        parameters = [inputs]
    else:
        actual = layer(inputs, indices, weights)
        expected = base(inputs, indices, weights)
        if mode in ("routed", ["shared", "routed"]):
            expected = expected + torch.stack(
                [sum(weights[t, j] * branch(inputs[t], int(indices[t, j])) for j in range(2)) for t in range(5)]
            )
        if mode in ("shared", ["shared", "routed"]):
            expected = expected + branch(inputs, prefix="shared_" if mode == ["shared", "routed"] else "")
        parameters = [inputs, weights]
    parameters += [parameter for name, parameter in layer.named_parameters() if "cortex_" in name]
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    actual_gradients = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().mean(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("kind", ["deepseek_v3", "qwen3_next", "nemotron_h"])
def test_device_native_shared_and_latent_moe_training(kind, device):
    base = BlockModel(kind).to(device).eval()
    inputs = torch.randn(2, 5, 16, device=device)
    expected = base(inputs).detach()
    model = get_peft_model(base, CortexConfig(k=3, moe_target=["shared", "routed"]))
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    model(inputs).square().mean().backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable and all(parameter.device == inputs.device for parameter in trainable)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert any(parameter.grad.abs().sum() > 0 for parameter in trainable)


@pytest.mark.parametrize("kind", ["qwen3_moe", "gpt_oss"])
def test_device_packed_adamw_inactive_expert_decay(kind, device):
    base = tiny_model(kind).to(device)
    model = get_peft_model(base, CortexConfig(k=3, moe_target="routed", adapter_bias=True))
    layer = model.get_base_model().model.layers[0].mlp.experts
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "cortex_" in name:
                parameter.normal_(std=0.2)
    parameters = [parameter for name, parameter in layer.named_parameters() if "cortex_" in name]
    optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.2)
    before = [parameter[2].detach().clone() for parameter in parameters]
    inputs = torch.ones(4, 16, device=device)
    indices = torch.tensor([[0, 1]] * 4, device=device)
    weights = torch.tensor([[0.7, 0.3]] * 4, device=device)
    layer(inputs, indices, weights).square().mean().backward()
    assert any(parameter.grad[:2].abs().sum() > 0 for parameter in parameters)
    assert all(torch.count_nonzero(parameter.grad[2]) == 0 for parameter in parameters)
    optimizer.step()
    for parameter, original in zip(parameters, before):
        torch.testing.assert_close(parameter[2], original * 0.98, rtol=1e-6, atol=1e-7)
        assert optimizer.state[parameter]["step"] == 1
