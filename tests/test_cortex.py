from copy import deepcopy
from importlib import import_module

import pytest
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from peft import CortexConfig, PeftModel, get_peft_model
from peft.tuners.cortex.layer import CortexLayer
from peft.tuners.cortex.structure import shared_names


def tiny_model(kind):
    config = AutoConfig.for_model(
        kind,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=3,
        num_local_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        enable_moe_block=True,
        top_k_experts=2,
        num_kv_shared_layers=0,
        global_head_dim=8,
        hidden_size_per_layer_input=0,
        max_position_embeddings=131072 if kind == "gpt_oss" else 64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return AutoModelForCausalLM.from_config(config)


def activate_branches(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, CortexLayer):
                for name in ("cortex_down", "cortex_shared_down"):
                    for projection in getattr(module, name).values():
                        projection.weight.normal_(std=0.1)


def test_olmoe_public_merge_rejected_without_modifying_model():
    base = tiny_model("olmoe").eval()
    assert base.config.norm_topk_prob is False
    model = get_peft_model(base, CortexConfig(k=3, moe_target="shared")).eval()
    activate_branches(model)
    tokens = torch.randint(3, 32, (1, 5))
    before = model(tokens).logits.detach()
    state = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="routing weights.*constant"):
        model.merge_and_unload()
    torch.testing.assert_close(model(tokens).logits, before, rtol=0, atol=0)
    assert state.keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


ARCHITECTURES = {
    "minimax": ("MiniMaxConfig", "MiniMaxSparseMoeBlock"),
    "minimax_m2": ("MiniMaxM2Config", "MiniMaxM2SparseMoeBlock"),
    "minimax_m3_vl": ("MiniMaxM3VLTextConfig", "MiniMaxM3VLSparseMoeBlock"),
    "deepseek_v2": ("DeepseekV2Config", "DeepseekV2Moe"),
    "deepseek_v3": ("DeepseekV3Config", "DeepseekV3MoE"),
    "deepseek_v32": ("DeepseekV32Config", "DeepseekV32MoE"),
    "deepseek_v4": ("DeepseekV4Config", "DeepseekV4SparseMoeBlock"),
    "kimi_linear": ("KimiLinearConfig", "KimiLinearMoE"),
    "glm4_moe": ("Glm4MoeConfig", "Glm4MoeMoE"),
    "glm4_moe_lite": ("Glm4MoeLiteConfig", "Glm4MoeLiteMoE"),
    "glm_moe_dsa": ("GlmMoeDsaConfig", "GlmMoeDsaMoE"),
    "glm5_next": ("Glm5NextTextConfig", "Glm5NextTextMoE"),
    "qwen3_next": ("Qwen3NextConfig", "Qwen3NextSparseMoeBlock"),
    "qwen3_5_moe": ("Qwen3_5MoeTextConfig", "Qwen3_5MoeSparseMoeBlock"),
    "nemotron_h": ("NemotronHConfig", "NemotronHMoE"),
}


class BlockModel(nn.Module):
    """Exercise real native FFNs without instantiating unrelated attention stacks."""

    def __init__(self, kind, config=None):
        super().__init__()
        config_name, block_name = ARCHITECTURES[kind]
        package = f"transformers.models.{kind}"
        config_cls = getattr(import_module(package + f".configuration_{kind}"), config_name)
        self.config = config or config_cls(
            hidden_size=16,
            intermediate_size=24,
            moe_intermediate_size=12,
            dense_intermediate_size=20,
            shared_intermediate_size=18,
            shared_expert_intermediate_size=18,
            moe_shared_expert_intermediate_size=18,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=32,
            num_experts=4,
            num_local_experts=4,
            n_routed_experts=4,
            n_shared_experts=2,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            mlp_layer_types=["moe" if kind == "deepseek_v4" else "sparse"] * 2,
            moe_latent_size=8,
            max_position_embeddings=64,
            layer_types=["sliding_attention" if kind == "deepseek_v4" else "full_attention"] * 2,
        )
        self.config._experts_implementation = "eager"
        cls = getattr(import_module(package + f".modeling_{kind}"), block_name)
        self.mlp = cls(self.config, layer_idx=0) if kind == "deepseek_v4" else cls(self.config)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.normal_(std=0.15)

    def forward(self, x):
        result = self.mlp(x)
        return result[0] if isinstance(result, tuple) else result


@pytest.fixture(autouse=True)
def deterministic_cpu():
    pytest.importorskip("transformers", minversion="5.17.0")
    previous_threads = torch.get_num_threads()
    torch.manual_seed(123)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("kind", ARCHITECTURES)
@pytest.mark.parametrize("mode", ["shared", "routed", ["shared", "routed"]])
def test_native_blocks_adapter_merge_and_native_reload(kind, mode, tmp_path):
    base = BlockModel(kind).eval()
    clean = deepcopy(base)
    native_shared = shared_names(base.mlp)
    x = torch.randn(2, 5, 16) * 20
    reference = base(x).detach()
    config = CortexConfig(k=3, k_shared=5, k_routed=2, moe_target=mode)
    model = get_peft_model(base, config).eval()
    torch.testing.assert_close(model(x), reference, rtol=0, atol=0)
    activate_branches(model)
    expected = model(x).detach()
    assert not torch.allclose(expected, reference)
    model(x).square().mean().backward()
    for name, parameter in model.named_parameters():
        if "cortex_" in name:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
    with model.disable_adapter():
        torch.testing.assert_close(model(x), reference, rtol=0, atol=0)
    if native_shared:
        roles = {module.role for module in model.modules() if isinstance(module, CortexLayer)}
        assert roles == ({"shared", "routed"} if mode == ["shared", "routed"] else {mode})
    if kind == "nemotron_h" and mode != "shared":
        assert model.get_base_model().mlp.experts.structure.input_dim == 8
        assert model.get_base_model().mlp.experts.structure.has_gate is False
    model.save_pretrained(tmp_path / "adapter", save_embedding_layers=False)
    loaded = PeftModel.from_pretrained(clean, tmp_path / "adapter", adapter_name="renamed").eval()
    torch.testing.assert_close(loaded(x), expected)
    value = x.clone().requires_grad_()
    before_grad = torch.autograd.grad(model(value).sum(), value)[0]
    merged = model.merge_and_unload(safe_merge=True)
    assert not any(isinstance(m, CortexLayer) for m in merged.modules())
    torch.testing.assert_close(merged(x), expected, rtol=2e-4, atol=3e-5)
    after_grad = torch.autograd.grad(merged(value).sum(), value)[0]
    torch.testing.assert_close(after_grad, before_grad, rtol=5e-4, atol=5e-5)
    native = BlockModel(kind, type(merged.config).from_dict(merged.config.to_dict())).eval()
    native.load_state_dict(merged.state_dict(), strict=True)
    torch.testing.assert_close(native(x), merged(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind,mode",
    [
        ("llama", "routed"),
        ("qwen2", "routed"),
        ("qwen3", "shared"),
        ("gemma3_text", "routed"),
        ("qwen3_moe", "routed"),
        ("mixtral", ["shared", "routed"]),
        ("gpt_oss", "shared"),
        ("gpt_oss", "routed"),
        ("gpt_oss", ["shared", "routed"]),
        ("olmoe", "routed"),
        ("gemma4_text", "shared"),
        ("gemma4_text", "routed"),
        ("gemma4_text", ["shared", "routed"]),
    ],
)
def test_full_native_merge_unmerge_and_save_reload(kind, mode, tmp_path):
    base = tiny_model(kind).eval()
    if kind == "gemma4_text":
        # Rebuild with a shared-KV, double-width MLP to test width coupling.
        base.config.num_kv_shared_layers = 1
        base.config.use_double_wide_mlp = True
        base.config.layer_types = ["full_attention", "full_attention"]
        base = AutoModelForCausalLM.from_config(base.config).eval()
    original_config = deepcopy(base.config.to_dict())
    original_state = deepcopy(base.state_dict())
    model = get_peft_model(base, CortexConfig(k=3, moe_target=mode, adapter_bias=kind == "gpt_oss")).eval()
    activate_branches(model)
    if kind == "gpt_oss":
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "cortex_" in name and name.endswith("bias"):
                    parameter.normal_(std=0.1)
    tokens = torch.randint(3, 32, (2, 5))
    expected = model(tokens).logits.detach()
    model.merge_adapter(safe_merge=True)
    torch.testing.assert_close(model(tokens).logits, expected)
    model.unmerge_adapter()
    torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)
    assert model.get_base_model().config.to_dict() == original_config
    for name, module in model.get_base_model().named_modules():
        if isinstance(module, CortexLayer):
            for key, parameter in module.base_layer.state_dict().items():
                torch.testing.assert_close(parameter, original_state[name + "." + key], rtol=0, atol=0)
    model.merge_adapter()
    merged = model.merge_and_unload(safe_merge=True)
    torch.testing.assert_close(merged(tokens).logits, expected)
    merged.save_pretrained(tmp_path)
    restored, info = AutoModelForCausalLM.from_pretrained(
        tmp_path, output_loading_info=True, experts_implementation="eager", attn_implementation="eager"
    )
    assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    restored.eval()
    torch.testing.assert_close(restored(tokens).logits, expected)


@pytest.mark.parametrize("mode", ["shared", ["shared", "routed"]])
def test_olmoe_public_branch_is_not_multiplied_by_routing_mass(mode):
    base = tiny_model("olmoe").eval()
    model = get_peft_model(base, CortexConfig(k=3, moe_target=mode))
    layer = model.get_base_model().model.layers[0].mlp.experts
    activate_branches(model)
    x = torch.randn(5, 16, requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 0], [0, 2]])
    weights = torch.tensor([[0.2, 0.3], [0.1, 0.2], [0.4, 0.5], [0.3, 0.1], [0.2, 0.4]], requires_grad=True)
    delta = layer(x, indices, weights) - layer.base_layer(x, indices, weights)
    prefix = "shared_" if mode == ["shared", "routed"] else ""
    expected = layer._branch(x, "default", prefix=prefix)
    if mode == ["shared", "routed"]:
        expected = expected + torch.stack(
            [
                sum(weights[t, j] * layer._branch(x[t], "default", int(indices[t, j])) for j in range(2))
                for t in range(5)
            ]
        )
    torch.testing.assert_close(delta, expected)
    if mode == "shared":
        grad = torch.autograd.grad(delta.sum(), weights)[0]
        torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=0)
    with pytest.raises(ValueError, match="routing weights.*constant"):
        model.merge_adapter()


def test_partial_layer_export_pads_untargeted_layers(tmp_path):
    base = tiny_model("qwen3").eval()
    before = deepcopy(base.model.layers[1].mlp.state_dict())
    model = get_peft_model(base, CortexConfig(k=3, target_modules=["model.layers.0.mlp"]))
    activate_branches(model)
    tokens = torch.randint(3, 32, (1, 5))
    expected = model(tokens).logits.detach()
    merged = model.merge_and_unload()
    assert merged.config.intermediate_size == 27
    torch.testing.assert_close(merged.model.layers[1].mlp.down_proj.weight[:, :24], before["down_proj.weight"])
    assert merged.model.layers[1].mlp.down_proj.weight[:, 24:].count_nonzero() == 0
    merged.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(restored(tokens).logits, expected)


def test_safe_merge_failure_is_atomic_and_bias_free_export_rejected():
    model = get_peft_model(tiny_model("qwen3"), CortexConfig(k=3))
    layers = [m for m in model.modules() if isinstance(m, CortexLayer)]
    with torch.no_grad():
        layers[-1].cortex_down["default"].weight[0, 0] = float("nan")
    first_weight = layers[0].base_layer.up_proj.weight
    with pytest.raises(ValueError, match="Non-finite"):
        model.merge_and_unload(safe_merge=True)
    assert layers[0].base_layer.up_proj.weight is first_weight
    assert not any(layer.merged for layer in layers)
    biased = get_peft_model(tiny_model("qwen3"), CortexConfig(k=3, adapter_bias=True))
    with pytest.raises(ValueError, match="bias-free"):
        biased.merge_and_unload()


@pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
def test_normalized_public_branch_merges_but_changed_router_is_rejected(kind):
    base = tiny_model(kind).eval()
    base.config.norm_topk_prob = True
    for layer in base.model.layers:
        layer.mlp.gate.norm_topk_prob = True
    model = get_peft_model(base, CortexConfig(k=3, moe_target=["shared", "routed"]))
    activate_branches(model)
    tokens = torch.randint(3, 32, (1, 5))
    expected = model(tokens).logits.detach()
    block = model.get_base_model().model.layers[-1].mlp
    block.gate.norm_topk_prob = False
    with pytest.raises(ValueError, match="normalization changed"):
        model.merge_and_unload()
    assert not any(m.merged for m in model.modules() if isinstance(m, CortexLayer))
    block.gate.norm_topk_prob = True
    merged = model.merge_and_unload()
    torch.testing.assert_close(merged(tokens).logits, expected)


@pytest.mark.parametrize("kind,mode", [("gpt_oss", ["shared", "routed"]), ("olmoe", "routed")])
def test_multiple_active_adapters_merge_and_unmerge(kind, mode):
    model = get_peft_model(tiny_model(kind).eval(), CortexConfig(k=3, moe_target=mode))
    model.add_adapter("second", CortexConfig(k=5, moe_target=mode))
    activate_branches(model)
    model.base_model.set_adapter(["default", "second"])
    tokens = torch.randint(3, 32, (1, 5))
    expected = model(tokens).logits.detach()
    model.merge_adapter()
    torch.testing.assert_close(model(tokens).logits, expected)
    model.unmerge_adapter()
    torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)
    model.set_adapter("second")
    second = model(tokens).logits.detach()
    merged = model.merge_and_unload(adapter_names=["second"])
    torch.testing.assert_close(merged(tokens).logits, second)


def test_gemma4_disabled_moe_discovers_dense_ffns_and_exports(tmp_path):
    torch.manual_seed(123)
    config = tiny_model("gemma4_text").config
    config.enable_moe_block = False
    config.num_kv_shared_layers = 1
    config.use_double_wide_mlp = True
    config.layer_types = ["full_attention", "full_attention"]
    base = AutoModelForCausalLM.from_config(config).eval()
    assert all(not layer.enable_moe_block for layer in base.model.layers)
    tokens = torch.randint(3, 32, (2, 5))
    original = base(tokens).logits.detach()
    model = get_peft_model(base, CortexConfig(k=3)).eval()
    assert model.peft_config["default"].target_roles == {
        "model.layers.0.mlp": "dense",
        "model.layers.1.mlp": "dense",
    }
    assert sum(isinstance(module, CortexLayer) for module in model.modules()) == 2
    activate_branches(model)
    expected = model(tokens).logits.detach()
    assert not torch.allclose(expected, original)
    native = model.merge_and_unload(safe_merge=True)
    assert not any(isinstance(module, CortexLayer) for module in native.modules())
    assert native.config.enable_moe_block is False
    assert native.config.intermediate_size == 27
    assert [layer.mlp.down_proj.in_features for layer in native.model.layers] == [27, 54]
    torch.testing.assert_close(native(tokens).logits, expected, rtol=2e-4, atol=3e-5)
    native.save_pretrained(tmp_path)
    restored, info = AutoModelForCausalLM.from_pretrained(
        tmp_path, output_loading_info=True, attn_implementation="eager", experts_implementation="eager"
    )
    assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    restored.eval()
    torch.testing.assert_close(restored(tokens).logits, expected, rtol=2e-4, atol=3e-5)
