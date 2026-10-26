# Cortex on MetaMathQA

Use these adapter configurations with the upstream MetaMathQA runner for
ordinary supervised causal-LM training. This benchmark uses neither generated
Cortex rehearsal data nor teacher-distillation loss.

## Configurations

The `k16-random` configuration follows the tuner's initialization and width
defaults; `k16-sample` isolates initialization at the same width. `k64-sample`
is an additional measured configuration with more trainable channels, not an
established optimum. The API does not prescribe an optimizer or task loss.

From `method_comparison/MetaMathQA`:

```bash
python run.py experiments/cortex/llama-3.2-3B-k16-random
python run.py experiments/cortex/llama-3.2-3B-k16-sample
python run.py experiments/cortex/llama-3.2-3B-k64-sample
```

These commands inherit the runner's model and training defaults. The table below
records author-run experiments using `unsloth/Llama-3.2-3B` rather than the
runner's default `meta-llama/Llama-3.2-3B` repository. Reproducing a row requires
matching its model, width, initialization, and training budget. Three adapter
configurations are bundled; the table also records additional widths and budgets.

## Reported results

| Adapter | Optimizer steps | Trainable parameters | GSM8K accuracy ↑ | Forgetting (ΔCE ↓) | Peak GPU memory (GB ↓) |
| --- | ---: | ---: | ---: | ---: | ---: |
| k16-sample | 5,000 | 4,128,768 | 48.22% | +0.02218 | 21.00 |
| k16-random | 5,000 | 4,128,768 | 46.70% | — | — |
| k32-sample | 5,000 | 8,257,536 | 49.05% | +0.02319 | 21.05 |
| k64-sample | 5,000 | 16,515,072 | 50.64% | +0.02780 | 21.30 |
| k32-sample | 20,000 | 8,257,536 | 55.95% | — | — |
| k64-sample | 20,000 | 16,515,072 | 57.77% | — | — |

A dash means the metric is not reported here. Forgetting is the increase in mean
per-token cross-entropy on the benchmark's 100 FineWiki texts relative to the
unadapted base model (after minus before; lower is better). Peak GPU memory uses
the runner's `accelerator_memory_max` field: peak PyTorch reserved GPU memory
minus the initialization baseline, converted to decimal GB (10^9 bytes). This
peak covers the benchmark run, including evaluation.

All runs use seed 0, batch size 4, maximum sequence length 768, BF16 base weights,
FP32 adapters, AdamW with learning rate 1e-4 and weight decay 0.1, and a cosine
schedule with 10% warmup. Gradient checkpointing and compilation are disabled.
Training uses the runner's filtered MetaMathQA dataset; evaluation uses all 1,319
GSM8K test questions. The 5,000-step runs process 20,000 training examples at
batch size four, matching the default training budget. The 20,000-step runs
use a longer budget. Each run starts afresh with its own schedule.

The results are preliminary, single-seed sanity checks, not multi-seed estimates
or official leaderboard entries. Compare methods at the same step budget, model
revision, and evaluation protocol; these rows do not establish a controlled
comparison against existing leaderboard entries or reproduce the paper's
memory-consolidation results.

## Provenance

Model revision: `d4446454d87d51aa42e1fb174f25acc5f8762331`.
MetaMathQA revision: `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`.
GSM8K revision: `740312add88f781978c0658806c59bc2815b9866`.
The runs used PyTorch 2.9.0, Transformers 5.17.0, and one NVIDIA H20 per run, with
the Cortex extension on PEFT commit
`eb74a79a4b6ab7f1e4afe7a54a7d862c36d34ecc`.
