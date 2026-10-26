<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Cortex

Cortex in PEFT implements the structural FFN expansion parameterization (Cortex-Structural) introduced in [Cortex: Consolidating Context into Parametric Memory](https://arxiv.org/abs/2610.20399).

Cortex freezes the base FFN and trains a narrow additive FFN with intermediate width `k`. For a standard gated FFN, its contribution is `down(act(gate(x)) * up(x))`. The adapter uses the base FFN's activation; non-gated FFNs use `down(act(up(x)))`.

This tuner provides the structural FFN update substrate. Cortex's prospective queries, context-conditioned teacher, distillation losses, and continual replay belong to the training pipeline and are not implemented by this tuner. The added parameters can also be trained with a standard task loss.

Compared with LoRA, this method allocates capacity to additional nonlinear FFN channels. For a bias-free gated FFN with input/output dimension `d`, the added parameter count is `3*d*k`; a non-gated FFN uses `2*d*k`. Structural merging widens the native FFN and increases deployment parameters and computation.

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM

from peft import CortexConfig, get_peft_model

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(device)
config = CortexConfig(
    k=16,
    init_weights="sample",
    init_noise_std=0.01,
    adapter_bias=False,
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

The [fine-tuning example](https://github.com/huggingface/peft/tree/main/examples/cortex/finetune) trains from a local JSONL dataset and saves the adapter and tokenizer. Its README shows how to reload them in a separate process. For BF16 training, load the base with `dtype=torch.bfloat16`; PEFT promotes adapter weights to FP32 by default.

For one documentation-to-adapter update, the [consolidation example](https://github.com/huggingface/peft/tree/main/examples/cortex/consolidate) supplies question generation, teacher targets, distillation, and evaluation. Continual replay and the policy for selecting past tasks remain application responsibilities. Ordinary supervised benchmark configurations and reported results are in the [MetaMathQA example](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA/experiments/cortex).

## Target selection

`target_modules` selects complete FFNs or MoE blocks. It does not select their individual linear projections. Lists use suffix matching and strings use full regular expression matching. With no explicit selection, the tuner discovers registered native Transformers FFN structures. `exclude_modules` can exclude a block or an individual resolved FFN.

Automatic discovery skips existing auxiliary training wrappers, including `ModulesToSaveWrapper`, and all their descendants. These wrappers own their parameters and checkpoint entries. Explicit Cortex targets cannot enter them, including through a regular expression or a resolved MoE branch. A new adapter's own `modules_to_save` and `exclude_modules` still exclude targets normally. If no eligible FFNs remain, adapter creation raises an error.

For example, if an earlier adapter fully trains `model.layers.0.mlp` through `modules_to_save`, a later adapter using automatic discovery adapts only the other eligible FFNs. It does not inject branches into `model.layers.0.mlp.original_module` or `model.layers.0.mlp.modules_to_save.<adapter>`. To fully train a separate copy for the later adapter, include the stable module path in that adapter's `modules_to_save` as well. Conversely, a new `modules_to_save` wrapper cannot enclose an existing Cortex layer, either directly or through an ancestor block: doing so would change the earlier adapter's checkpoint ownership. This is rejected before injection; adding a copy to an existing auxiliary wrapper remains supported.

For a new adapter, explicitly selecting a packed experts module, such as `model.layers.0.mlp.experts`, selects routed branches regardless of `moe_target`. Select the parent MoE block, such as `model.layers.0.mlp`, to choose shared, routed, or both with `moe_target`. These rules also apply when another Cortex adapter already wraps the target; the existing adapter does not determine the new adapter's role. Loading a checkpoint instead restores its saved `target_roles`.

Registered dense structures include the native Gemma3 MLP, as well as Llama, Qwen2, Qwen3 and Gemma4. Gemma3 text causal LMs use native auto-discovery and do not need a custom FFN wrapper.

Explicit custom FFNs must implement the standard gated expression above, with ordinary `nn.Linear` projections named `gate_proj`, `up_proj`, and `down_proj`, and an audited, deterministic elementwise `act_fn`, such as `nn.SiLU`, `nn.GELU`, or their PyTorch functional forms. Parameter-free softmax, normalization, dropout, and unaudited custom callables are rejected before targets are replaced: structural merging requires `act(cat(a, b)) == cat(act(a), act(b))`. Subclasses overriding an approved activation's forward also require an audit. Omitting `gate_proj` selects the non-gated expression. Extra internal normalization, residuals, or gating require a structure implementation and numerical tests; matching attribute names alone does not establish equivalent arithmetic. Custom FFNs are not auto-discovered.

The [custom FFN classification example](https://github.com/huggingface/peft/tree/main/examples/cortex/finetune#custom-ffn-classification) uses this contract with a PyTorch training loop. It leaves `task_type` unset and trains the classification head through `modules_to_save`.

Packed native experts require three-dimensional floating-point parameters. Older ModuleList expert layouts, quantized projections, TP/EP shards, and unknown remote model implementations are unsupported. The current packed-layout validation baseline is Transformers 5.17.0; this is not a global PEFT minimum. Packed experts must expose the audited layout metadata and `[tokens, top_k]` routing interface. Special clipped/offset activations must expose their clipping limit and, where applicable, alpha; missing metadata is rejected instead of guessed. Expert-internal output normalization (`has_post_expert_norm=True`) is unsupported because moving the added channels inside that normalization would change their function.

## MoE branches

| `moe_target` | Native shared FFN present | No native shared FFN |
| --- | --- | --- |
| `shared` (default) | Extend the native shared FFN | Add an unrouted public branch |
| `routed` | Extend each routed expert | Extend each routed expert |
| `["shared", "routed"]` | Extend both | Add public and routed branches |

`k_routed` overrides the width **per expert**. `k_shared` overrides the width per physical shared FFN or public branch. Both fall back to `k`; dense FFNs always use `k`. A fused native shared FFN gets `k_shared` new neurons, without another multiplication by the number of logical shared experts.

By default, the public branch implements `sum_e g_e(x) * E_e(x) + E_s(x)`: all tokens traverse the same added parameters independently of expert selection. On models without native shared experts, select `moe_target="shared"` for this Cortex-Structural MoE construction; `routed` selects independent expert expansions. The API default is `shared`. Combined selections are passed and saved as lists. On models with native shared experts, `shared` expands their internal FFNs, retaining their gates and normalization; this is a different construction from adding an unconditional public branch when those outer operations affect the increment. A native shared path has the same additive semantics only with unit coefficient, matching input/output domains, and no extra normalization. `["shared", "routed"]` combines the extensions and is not the shared-only construction.

For an added public branch, set `public_weighting="routing"` to multiply its output by `c(x)`, the sum of the selected native router weights: `sum_e g_e(x) * E_e(x) + c(x) * E_s(x)`. One set of shared parameters is evaluated once, and the complete branch output, including down bias, is weighted. This equals adding the same branch inside every selected expert before routing, so it can merge into experts even with unnormalized top-k probabilities. It does not renormalize weights, change expert selection, or modify the router.

```python
config = CortexConfig(
    k=16,
    moe_target="shared",  # ["shared", "routed"] also supports this option
    public_weighting="routing",
    init_weights="sample",
    adapter_bias=False,
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, config)
```

The default `"unweighted"` preserves existing behavior, including checkpoints without this field. The option is saved separately for each adapter and only affects added public branches: dense FFNs, native shared FFNs, and independent routed branches retain their existing arithmetic. When `c(x) == 1`, the two public definitions are mathematically equal. Otherwise they give different outputs and gradients; do not switch an already-trained adapter to `"routing"` only to enable merging. Both modes retain zero-down initialization.

Routed branches reuse the native expert indices and weights without renormalization. Native shared gates, external normalization, and latent projections remain at their original boundaries. Latent expert dimensions are inferred from the expert weights. Native clipped/offset SwiGLU and GELU semantics are retained by the structure descriptors.

The native base output and routed increments are computed separately and then added. This is mathematically equivalent to adding the increment within each expert before routing, but floating-point rounding need not be identical.

Each routed projection is one packed `[E, out, in]` parameter. Standard AdamW shares its step counter across expert slices. An unselected slice has a zero data gradient when other slices participate, but can still change through weight decay or existing momentum. This differs from separate expert parameters with `grad=None`, which the optimizer skips. Without weight decay, shared bias correction and momentum can still produce different updates; mathematical forward equivalence does not establish identical training trajectories.

Ignoring biases, a gated branch with input/output dimension `d` has `3*d*k` trainable matrix parameters. For `E` routed experts this is `3*d*E*k_routed` in total, although only the selected experts contribute per token. A single public branch has `3*d*k_shared` parameters and contributes to every token. Matching total parameters and matching active width are different comparisons. For latent or non-gated FFNs, use the actual input/output dimensions and projection count rather than this simplified formula.

## Initialization

- `"random"` (default): Kaiming-uniform gate/up weights with `a=sqrt(5)`, using
  the input dimension as fan-in for each expert. Existing gate/up biases use
  uniform `[-1/sqrt(fan_in), 1/sqrt(fan_in)]` initialization by default; set
  `adapter_bias_init="zeros"` to initialize them to zero.
- `"sample"`: copy paired gate/up neurons from the corresponding base FFN.
  Native shared FFNs sample themselves; routed experts sample independently;
  public branches sample real channels from the pool of all experts, keeping each
  gate/up pair and its biases from the same expert and channel. Gate/up biases are also
  copied when both the base and adapter have them. `init_noise_std` adds
  Gaussian noise to the sampled weights (default `0.01`; use `0` for exact
  copying). Sample without replacement when the width fits the source pool
  (one FFN for dense/shared/routed branches, all experts for a public branch);
  larger widths sample with replacement.

Both strategies zero the down weights and down bias, preserving the initial model output. `True` aliases `"random"`; `False` keeps random down weights for PEFT testing and is not an additional training initialization strategy. All biases still start at zero in this testing mode. Sampling and the absolute noise scale are initialization choices, not guarantees of matching the paper's training recipe or improving convergence. Loading an existing adapter restores its saved weights independently of the initialization strategy used for new adapters.

This identity contract assumes finite gate/up features and requires both the down weights and its bias, when present, to be zero. Changing gate/up features still changes the first down-weight gradient, even though the initial output is identical.

`adapter_bias=None` (default) follows bias presence in each corresponding native gate/up/down projection; added public branches follow the native expert projections. Fused gate/up projections share the same bias presence. Set `adapter_bias=True` to create all adapter biases, or `False` to create none. `adapter_bias_init="uniform"` (default) or `"zeros"` controls existing gate/up biases only during random initialization; sample initialization still copies available native biases and zeros missing ones. A zero initial value does not disable a trainable bias. PEFT promotes BF16 adapter weights to FP32 by default; pass `autocast_adapter_dtype=False` to `get_peft_model` when matching BF16 adapter execution is required.

Base-dependent initialization uses PEFT's `gather_params_ctx` and recognizes ZeRO-3 logical parameter shapes. Checkpoint loading with `low_cpu_mem_usage=True` skips initialization on meta tensors. Use a fully materialized model to initialize a new adapter; `low_cpu_mem_usage=True` is intended for loading trained adapters.

## Task adapters

Use `get_peft_model(..., adapter_name="task1")` and `add_adapter("task2", config)` to allocate separate parameters for successive tasks. Adapters may use different `moe_target` selections when their resolved physical FFNs are distinct. Dense FFNs ignore this option, including when multiple adapters target the same dense FFN. Adapters sharing a physical FFN must resolve to the same role; their widths, initialization, and `public_weighting` may differ. `model.set_adapter("task2")` selects an isolated task; task1 is inactive. For cumulative training, keep previous adapters active but frozen:

```python
import torch

model.base_model.set_adapter(["task1", "task2"])
model.set_requires_grad("task1", requires_grad=False)
model.set_requires_grad("task2", requires_grad=True)
model.zero_grad(set_to_none=True)
optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=1e-3,
)
```

Compatibility is checked after target expansion. Native shared FFNs and routed experts in the same MoE block are distinct physical targets and can have separate adapters. Without native shared experts, public, routed, and combined extensions share one experts wrapper and cannot use different roles across adapters. A conflicting request raises before injecting the new adapter, regardless of which adapter was added first.

The outer `PeftModel.set_adapter` takes a single name; the tuner's `model.base_model.set_adapter` accepts multiple names. Activation enables gradients on active adapters, so apply the freeze rules after activation and reapply them after adding or switching adapters. Clear old gradients and build a fresh optimizer containing only the new task's trainable parameters.

Save named adapters with `model.save_pretrained("outputs/tasks", selected_adapters=["task1", "task2"])`; their checkpoints are in `outputs/tasks/task1` and `outputs/tasks/task2`. Reload the same base model, load each adapter, and reapply the activation and freeze rules. Adapter checkpoints do not include optimizer state or the cumulative activation policy. Frozen previous parameters can still contribute to the forward pass, and enabling a new adapter can change old task outputs; this parameter isolation does not establish freedom from forgetting.

When an earlier adapter owns a `modules_to_save` copy, selecting only a later adapter that does not own that module uses the original module. Activating both adapters also uses the earlier adapter's saved copy. A checkpoint of the later adapter alone therefore reproduces its isolated behavior on the original base, not the combined model. Save and reload all required adapters when training or inference depends on their combined state.

When merging is supported, explicitly select all task adapters with `merge_and_unload(adapter_names=["task1", "task2"], safe_merge=True)`. To merge after each task, reinject a new adapter into the returned native model, which becomes the frozen base for the next task. Unmerge before adding an adapter to an existing merged wrapper. Keep adapter checkpoints if individual task switching is required.

## Saving and structural merging

Use the standard PEFT `save_pretrained` / `PeftModel.from_pretrained` APIs for adapter checkpoints. The tuner records resolved physical paths and their roles in `target_roles`; do not edit that internal field. Router properties are checked at runtime, rather than serialized as adapter configuration. PEFT retains explicitly supplied configuration objects, and Cortex populates their resolved targets. Create a fresh `CortexConfig` for a different model. Reusing an in-memory configuration checks the original target selection again and raises if the resolved layers or roles would change, rather than silently omitting new layers. Saved configurations intentionally retain their resolved paths and roles; they are not rediscovered on loading. Loading onto a different architecture requires explicit validation; a state-dict `key_mapping` does not remap `target_roles`.

```python
from peft import PeftModel

# After training, save only the adapter and reload the same base checkpoint.
model.save_pretrained("structural-adapter")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
restored = PeftModel.from_pretrained(base_model, "structural-adapter").eval()

# Optional native export. Keep the adapter checkpoint for task switching.
merged = restored.merge_and_unload(safe_merge=True)
merged.save_pretrained("structural-native")
native = AutoModelForCausalLM.from_pretrained("structural-native")
```

`merge_adapter` concatenates neurons; `unmerge_adapter` restores the original parameters and dimensions. `merge_and_unload` merges the selected adapters into the current base weights and returns the widened native model. It removes all adapter wrappers, including those for unselected adapters, but does not undo previously merged channels. Previously merged adapters remain in the exported model even when omitted from `adapter_names`. Save any adapters you want to keep before calling it.

For example, calling `merge_adapter(adapter_names=["a"])` followed by `merge_and_unload(adapter_names=["b"])` retains both adapters' effects. To export only `b` relative to the current wrapper's original base, undo its reversible merges first:

```python
model.unmerge_adapter()
exported = model.merge_and_unload(adapter_names=["b"], safe_merge=True)
```

Merging a subset of active adapters keeps the other active, unmerged branches in the forward and backward computation. Unmerging restores the original parameters before computing all active branches separately again. For registered native FFNs, export updates configuration widths and zero-pads unadapted FFNs that share those widths. The export plan follows the module copies retained by `modules_to_save` and the configuration fields used by the native constructor, including separate serialized sub-configurations. Custom model constructors must account for the enlarged dimensions when reconstructing a merged model. Reversible merges retain the original widths and only the dimension attributes modified by merging, plus the original down bias when present; they do not retain a full copy of the original FFN weights. Unmerge copies the original channel slices into compact storage and restores the saved down bias, avoiding subtraction roundoff. This small bias buffer follows device/dtype conversions and is excluded from checkpoints. Successful unmerge creates new base Parameter objects; external references to old base Parameters are not preserved.

Merge, unmerge, and native export commit one FFN at a time. Preparing a replacement temporarily retains that FFN's old parameters; those references are released after the FFN succeeds, rather than accumulating across previously processed FFNs. A failed parameter or dimension assignment restores the fields managed by Cortex for the current FFN; it does not undo already completed FFNs or arbitrary external hook side effects.

Preflight checks reject unsupported configurations before changes begin; with `safe_merge=True`, they also check for non-finite values. These checks cannot prevent allocation failures or arbitrary failures during assignment, padding, or configuration updates. If an operation raises after changes begin, discard that model instance and reload the original base model and an adapter checkpoint saved before the operation. Do not retry the partially modified instance or save it as a complete export.

A routing-weighted public branch merges by copying its neurons into every routed expert, without a constant-mass requirement. An unweighted public branch requires a proven constant nonzero sum of selected router weights. OLMoE's default `norm_topk_prob=False`, and unnormalized Qwen3-MoE routing, have token-dependent mass, so their unweighted public branches cannot merge; their routed and routing-weighted public branches can. Mixed adapters are checked individually: an active, unmerged unweighted adapter does not prevent merging a selected routing-weighted adapter. Adapter biases cannot merge into bias-free native FFNs. Reload a fully materialized, unpartitioned model before structural merging; replacing shapes inside a ZeRO engine is unsupported.

Copying a public branch into every routed expert increases the exported weight storage by the expert count. Structural merging enlarges the native FFN; it does not preserve the original parameter count or expert-kernel width.

Reload exported MoE models with `experts_implementation="eager"`. Arbitrary expanded widths may violate optimized expert-kernel alignment requirements. The routed adapter forward groups tokens by expert in Python; it does not provide an optimized fused adapter kernel. Quantized projections, TP/EP shards, and active Accelerate CPU/disk offload hooks are unsupported. Offload is checked before injection and before merging or unmerging, including when enabled after injection. Ordinary CPU residency and `low_cpu_mem_usage=True` adapter checkpoint loading remain supported. Multi-rank FSDP and compiled execution have not been validated. Distributed training support depends on the architecture, dtype, initialization, and engine configuration; tests on small DDP/ZeRO models do not establish support for every combination.

MoE parent forwards must retain their native implementation, either directly or through a supported non-offload Accelerate device hook. Compatibility checks verify the actual Accelerate wrapper and its dispatch to the native forward, rather than trusting copied metadata or the presence of `_old_forward`. Parent-level kernels such as MegaBlocks may read expert projection weights directly and bypass the Cortex wrapper; these replacements are rejected at injection, at merge, and by a parent pre-forward check if installed after injection. A kernel annotation or native fallback alone is allowed, and kernels confined to attention or normalization are not rejected by this check. This restriction also applies to floating-point models; kernel acceleration does not imply quantization.

## CortexConfig

[[autodoc]] tuners.cortex.config.CortexConfig

## CortexModel

[[autodoc]] tuners.cortex.model.CortexModel
    - merge_and_unload
