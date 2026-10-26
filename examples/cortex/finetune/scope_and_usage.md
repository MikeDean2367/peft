# Cortex in PEFT: scope and usage

`CortexConfig` implements the **Cortex-Structural FFN parameterization**. A user can attach it to a supported model, train with an ordinary task loss using `Trainer` or a PyTorch loop, save an adapter, and reload it through PEFT. A teacher model, context-generation procedure, and special optimizer are not required for this use.

The tuner implements the structural parameter update. The [consolidation example](../consolidate/README.md) illustrates a single context-to-parameters update; the complete research workflow, including continual replay, is outside this tuner. Ordinary fine-tuning can use the adapter directly.

The examples below require a PEFT checkout containing Cortex. They explain the integration API and are not a reproduction of the paper's memory-consolidation results.

## 1. Using Cortex as a PEFT tuner

### Which parts of the paper enter PEFT?

The [Cortex paper](https://arxiv.org/abs/2610.20399) separates rehearsal construction (§3.1), FFN parameterization (§3.2), and distillation (§3.3). The integration boundary is:

| Paper component | PEFT scope | Consequence for users |
| --- | --- | --- |
| §3.2 Structural FFN expansion | Included: added channels, initialization, frozen base weights | Select `CortexConfig` and optimize the new parameters |
| §3.2 Structural MoE expansion | Included for supported layouts: shared and expert-specific branches | Choose the branch configuration before training |
| §3.2 Synaptic low-rank updates | Outside this tuner; can build on PEFT's existing LoRA support | Use `LoraConfig` when this parameterization is wanted |
| §3.1 Prospective queries and response generation | Outside the tuner | Training scripts supply the dataset |
| §3.3 Context-conditioned teacher, output KL, hidden-state alignment, and replay | Outside the tuner | A separate training workflow is needed to reproduce those experiments |

Adapter creation, freezing, switching, and checkpointing are PEFT responsibilities. Deciding which past queries to replay, constructing teacher inputs, aligning teacher/student positions, and combining losses remain training-script responsibilities. Multiple adapters provide the building blocks for sequential updates; the tuner does not automatically implement the paper's continual-learning policy.

Ordinary supervised training therefore uses the same trainable structure with a different source of supervision. It does not, by itself, reproduce the paper's consolidation/retention evaluation or establish preservation of prior capabilities. A paper-reproduction workflow can use the same `CortexConfig` adapter and supply the additional data and loss computation externally.

### What changes relative to LoRA?

For a standard gated FFN, the implementation computes:

```text
output = frozen_ffn(x) + down_s(activation(gate_s(x)) * up_s(x))
```

A non-gated FFN uses `down_s(activation(up_s(x)))`. The added intermediate width is `k`. The down projection starts at zero, so attaching the adapter initially preserves the base output. `init_weights="sample"` initializes gate/up channels from existing FFN weights; it does not sample a training dataset or require teacher generation.

| Usage decision | LoRA on FFN projections | Cortex |
| --- | --- | --- |
| Adapter configuration | `LoraConfig(r=..., target_modules=[...])` | `CortexConfig(k=..., target_modules=[...])` |
| Target granularity | Individual projections | Complete supported FFN/MoE blocks; registered layouts can be discovered automatically |
| Capacity parameter | Matrix-update rank `r` | Additional nonlinear channel count `k` |
| Training | Existing loss, optimizer, and training loop | Existing loss, optimizer, and training loop |
| Adapter checkpoint | `save_pretrained` / `PeftModel.from_pretrained` | The same APIs |
| Optional merge | Add the low-rank update to existing weight matrices | Concatenate channels into a wider native FFN |

Cortex targets such as `"mlp"` refer to complete FFNs. LoRA target lists such as `"q_proj"`, `"v_proj"`, or `"all-linear"` cannot simply be copied into `CortexConfig`. The matching learning rate, width, and training budget remain empirical choices; sharing the training API does not imply sharing optimal hyperparameters.

Structural merging increases the deployed FFN width, parameter count, and computation. Export is limited to validated configurations; it is optional for inference with a loaded adapter.

### Example 1: ordinary language-model fine-tuning

This self-contained example uses standard causal-language-model cross-entropy and the unmodified Transformers `Trainer`. The short corpus only demonstrates the API. Run on one visible GPU, or CPU; the first run downloads the model and tokenizer unless `model_id` points to a local checkpoint.

```python
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import CortexConfig, get_peft_model

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

data = Dataset.from_dict({"text": ["The capital of France is Paris.", "The capital of Japan is Tokyo."]})


def tokenize(batch):
    encoded = tokenizer(batch["text"], truncation=True, max_length=64, padding="max_length")
    encoded["labels"] = [
        [token if mask else -100 for token, mask in zip(ids, attention_mask)]
        for ids, attention_mask in zip(encoded["input_ids"], encoded["attention_mask"])
    ]
    return encoded


dataset = data.map(tokenize, batched=True, remove_columns=data.column_names)
base_model = AutoModelForCausalLM.from_pretrained(model_id)
base_model.config.use_cache = False
model = get_peft_model(
    base_model,
    CortexConfig(k=16, init_weights="sample", task_type="CAUSAL_LM"),
)
model.print_trainable_parameters()
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="outputs/cortex-training",
        max_steps=10,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        optim="adamw_torch",
        save_strategy="no",
        report_to="none",
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=default_data_collator,
)
trainer.train()
model.save_pretrained("outputs/cortex-adapter", save_embedding_layers=False)
tokenizer.save_pretrained("outputs/cortex-adapter")
```

The tokenizer, labels, padding mask, optimizer, and training loop are ordinary task-training code. `get_peft_model` creates the additional FFN channels and freezes the base model. There is no separate teacher forward pass or Cortex-specific loss in this example. Prompt/completion SFT can use its usual completion-only label mask instead.

The example uses the default full-precision loading for portability. BF16 training follows the usual model-loading and `TrainingArguments` settings; the tuner follows PEFT's adapter dtype policy. A complete command-line example with dtype options, optional generation, and native export is provided in [finetune.py](finetune.py).

In a separate inference process, reload the same base checkpoint and the saved adapter:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftConfig, PeftModel

adapter_dir = "outputs/cortex-adapter"
config = PeftConfig.from_pretrained(adapter_dir)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, revision=config.revision)
model = PeftModel.from_pretrained(base_model, adapter_dir).eval()
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

For this registered dense layout, an optional native export uses the normal PEFT method name:

```python
merged = model.merge_and_unload(safe_merge=True)
merged.save_pretrained("outputs/cortex-native")
tokenizer.save_pretrained("outputs/cortex-native")
native = AutoModelForCausalLM.from_pretrained("outputs/cortex-native")
```

This saves a full model with wider FFNs. Keep the adapter checkpoint to retain adapter switching.

### Evidence from ordinary supervised training

We ran the experiments below using the upstream MetaMathQA benchmark with `unsloth/Llama-3.2-3B`, standard supervised causal-LM training, and GSM8K evaluation. We used no generated Cortex rehearsal data or teacher-distillation loss.

| Configuration | Optimizer steps | Trainable parameters | GSM8K accuracy ↑ | Forgetting (ΔCE ↓) | Peak GPU memory (GB ↓) |
| --- | ---: | ---: | ---: | ---: | ---: |
| k=16, sample initialization | 5,000 | 4,128,768 | 48.22% | +0.02218 | 21.00 |
| k=32, sample initialization | 5,000 | 8,257,536 | 49.05% | +0.02319 | 21.05 |
| k=64, sample initialization | 5,000 | 16,515,072 | 50.64% | +0.02780 | 21.30 |

Forgetting is the increase in mean per-token cross-entropy on the benchmark's 100 FineWiki texts relative to the unadapted base model (after minus before; lower is better). Peak GPU memory uses the upstream [`accelerator_memory_max`](https://github.com/huggingface/peft/blob/main/method_comparison/MetaMathQA/utils.py) field: peak PyTorch reserved GPU memory minus the initialization baseline, converted to decimal GB (10^9 bytes). This peak covers the benchmark run, including evaluation, rather than isolating training steps.

We ran all three experiments on one NVIDIA H20 each with seed 0, batch size 4, maximum sequence length 768, BF16 base weights, FP32 adapters, AdamW with learning rate 1e-4 and weight decay 0.1, and a cosine schedule with 10% warmup. We evaluated all 1,319 GSM8K test questions. The model revision is `d4446454d87d51aa42e1fb174f25acc5f8762331`; dataset revisions and further records are in `method_comparison/MetaMathQA/experiments/cortex/README.md` at the repository root.

We present these preliminary, single-seed results to demonstrate CE-based fine-tuning in this setting. Our archived run configurations and final metrics all record 5,000 optimizer steps and 20,000 training examples processed at batch size 4, matching the benchmark's [default training budget](https://github.com/huggingface/peft/blob/main/method_comparison/MetaMathQA/default_training_params.json). We used the `unsloth/Llama-3.2-3B` checkpoint rather than the default `meta-llama/Llama-3.2-3B` repository. We are not presenting these runs as official leaderboard entries, controlled comparisons against the existing leaderboard results, or a reproduction of the paper's memory-consolidation results.

### Example 2: a model that is not an LLM

Cortex also supports explicitly selected custom FFNs with ordinary `nn.Linear` projections and the supported elementwise FFN arithmetic. The classification example below was tested locally on CPU in FP32: training produced a finite loss, updated the adapter and classification head, and preserved the frozen base FFN weights. It uses feature vectors and classification labels, without a tokenizer, attention layers, or autoregressive decoding. The random data and base weights demonstrate compatibility, not useful classification performance.

```python
import torch
from torch import nn
from torch.nn import functional as F

from peft import CortexConfig, get_peft_model


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.act_fn = nn.GELU()
        self.down_proj = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = FFN()
        self.head = nn.Linear(16, 3)

    def forward(self, features):
        return self.head(self.ffn(features))


model = get_peft_model(
    Classifier(),
    CortexConfig(
        k=4,
        target_modules=["ffn"],
        modules_to_save=["head"],
    ),
)
optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=1e-3,
)
features = torch.randn(16, 16)
labels = torch.randint(0, 3, (16,))
model.train()
for _ in range(5):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(features), labels)
    loss.backward()
    optimizer.step()
print(f"Classification loss: {loss.item():.4f}")
```

Leaving `task_type` unset uses PEFT's generic model wrapper. `modules_to_save=["head"]` additionally trains and saves the classification head; the original FFN weights remain frozen. A custom model must expose and implement the supported FFN structure, as this example does. Merely having two linear layers with arbitrary intermediate operations is insufficient.

Thus the parameterization is not intrinsically LLM-specific. Native integration and most architecture validation focus on supported Transformers dense/MoE FFNs. Broader architecture support will be documented based on testing. This custom-model example does not establish automatic support for arbitrary encoders, vision backbones, or diffusion models; their native FFN layouts and export paths need explicit integration and tests. It also does not establish task effectiveness outside the language-model experiments.

### MoE usage

The training interface remains the same for supported MoE models. On a layout with no native shared expert, such as GPT-OSS, `CortexConfig(k=16, moe_target="shared", public_weighting="unweighted")` adds a single branch traversed by every token. `moe_target="routed"` adds an independent branch per routed expert. The API default is `"shared"`; use `moe_target=["shared", "routed"]` to combine both extensions. On architectures with a native shared FFN, this option extends that FFN and retains its existing outer gates and normalization.

Router parameters stay frozen, while native routing is still computed from the current hidden states. When a model already has a native shared FFN, `"shared"` expands that FFN and retains any surrounding gates or normalization; this is not generally equivalent to adding an unconditional branch at the MoE output. These choices affect which function is trained, not the need for a special optimizer or loss.

For an unweighted added shared branch, copying it into experts is equivalent when the selected routing weights sum to one and the expert computations are compatible. Such export duplicates its weights across experts. Native merging is therefore checked per layout and configuration. The detailed API document covers the additional routing-weighted and combined modes.

## 2. Using PEFT inside the full Cortex method

We plan to keep the complete paper workflow in `zjunlp/cortex`. In the PEFT-based version, our training scripts would import `CortexConfig` and `get_peft_model` from PEFT to allocate the trainable FFN channels. We would implement data generation, teacher/student input construction, distillation, and the continual-update policy in our research repository. The steps below illustrate this planned integration; the helper functions are conceptual pseudocode, not additional public PEFT APIs.

| Stage | Responsibility | What it does |
| --- | --- | --- |
| Prepare rehearsal | Cortex training code | Generate queries and context-conditioned responses, or ingest task-provided examples |
| Build supervision | Cortex training code | Render teacher/student inputs, align positions, construct masks, and optionally cache teacher outputs |
| Attach adaptation parameters | PEFT | `get_peft_model(base, CortexConfig(...))` creates the channels and freezes base weights |
| Optimize | Cortex training code and ordinary PyTorch/Trainer | Compute the configured KL and hidden-state losses and update the trainable parameters |
| Save, reload, or export | PEFT | Save/load the adapter; optionally merge on a validated configuration |
| Continue to another update | Cortex training code using PEFT adapter operations | Retain earlier memories, freeze older parameters, and select replay queries/targets |

A runnable [SEIS API-learning example](../consolidate/README.md) implements this connection with documentation-generated programming questions, four generic questions, cached teacher supervision, and interchangeable Cortex/LoRA adapters. It evaluates context-free code generation on separate programming tasks with executable tests. Generic teacher prompts receive the source context by default; a command-line option omits it. This example runs independently of the broader research repository.

### Pseudocode for one consolidation update

The two models start from the same checkpoint. The teacher is a separate frozen instance, so attaching an adapter to the student cannot modify the teacher. A single-update illustration is sufficient to show the dependency boundary:

```python
# Pseudocode: cortex_* helpers belong to the external training workflow.
from copy import deepcopy

import torch

from peft import CortexConfig, get_peft_model

base = load_base_checkpoint(model_id)
teacher = deepcopy(base).eval()  # Same checkpoint, independent model instance.
teacher.requires_grad_(False)
records = cortex_generate_rehearsal(teacher, context)  # Questions and responses.
batches = cortex_build_aligned_batches(records, context)

student = get_peft_model(base, CortexConfig(k=16, init_weights="sample", task_type="CAUSAL_LM"))
optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)
student.train()
for batch in batches:
    optimizer.zero_grad()
    with torch.no_grad():
        target = cortex_forward_with_ffn_states(teacher, batch.teacher_inputs)
    prediction = cortex_forward_with_ffn_states(student, batch.student_inputs)
    # Masked, position-aligned KL(teacher || student) + post-FFN state MSE.
    loss = cortex_distillation_loss(prediction, target, batch.alignment)
    loss.backward()
    optimizer.step()

student.save_pretrained("cortex-memory", save_embedding_layers=False)
```

`model_id` and `context` are supplied by the application. `load_base_checkpoint` denotes ordinary model loading. Every `cortex_*` helper is outside the PEFT tuner:

- `cortex_generate_rehearsal` prepares questions and responses. Agent tasks can supply environment queries or trajectories instead.
- `cortex_build_aligned_batches` creates a teacher view containing context, question, and response, and a student view containing the corresponding question and response without the source context. It records token positions and masks rather than assuming equal sequence lengths.
- `cortex_forward_with_ffn_states` calls the model and collects the required post-FFN states using architecture-appropriate output selection or hooks. State collection is training-side logic.
- `cortex_distillation_loss` applies the configured temperature, token masks, logit-position alignment, layer selection, and loss weights. It computes teacher-to-student KL and aligned hidden-state MSE. The pseudocode omits the experiment-specific configuration and is not a numerical reproduction recipe.

Frozen teacher outputs can be cached before optimization; retaining two live model instances is only a convenient way to illustrate the roles. Inference later needs the base model and saved adapter, with the ordinary PEFT loading and generation APIs. It does not need the teacher or source context.

For sequential consolidation, the teacher is the pre-update effective model, including previously learned memories. If those memories remain as named adapters, the training code adds a new adapter, keeps the old ones active but frozen, and optimizes only the new adapter. Globally disabling all adapters would also remove those previous memories, so it would not implement the required teacher in that setting. Historical-query replay, target regeneration, retention data, and the activation/freeze policy remain responsibilities of the external workflow. The one-update pseudocode does not implement this scheduler.

### The Cartridges precedent

[PEFT's Cartridges documentation](https://huggingface.co/docs/peft/main/en/package_reference/cartridges#training) separates its learned KV-cache prefix from the training procedure. The [self-study example directory](https://github.com/huggingface/peft/tree/main/examples/cartridge_self_study) contains `synthesize.py` for data generation and `train_distill.py` for distillation. Those scripts use the adapter without making their data-generation or loss pipeline part of the core library.

The intended Cortex integration follows this organization: core FFN parameterization and adapter lifecycle, a standard task-training example, and a clearly separate explanation or example of the paper-specific workflow. Such an example could live in the PEFT repository without placing the full Cortex workflow inside the tuner. Cortex learns FFN channels rather than a KV-cache prefix; it introduces no virtual-token prefix.
