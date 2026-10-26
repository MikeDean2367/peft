# Cortex on MetaMathQA

Use these adapter configurations with the upstream MetaMathQA runner. The
`k16-random` configuration follows the tuner's initialization and width defaults;
`k16-sample` isolates initialization at the same width. `k64-sample` is an
additional measured configuration with more trainable channels, not an
established optimum. The API does not prescribe an optimizer or task loss.

From `method_comparison/MetaMathQA`:

```bash
python run.py experiments/cortex/llama-3.2-3B-k16-random
python run.py experiments/cortex/llama-3.2-3B-k16-sample
python run.py experiments/cortex/llama-3.2-3B-k64-sample
```

These commands inherit the runner's model and training defaults. The table below
records author-run experiments using `unsloth/Llama-3.2-3B`; reproducing it requires
matching the model and training settings below as well as the adapter config.

| Adapter | Optimizer steps | Trainable parameters | GSM8K accuracy |
| --- | ---: | ---: | ---: |
| k16-sample | 5,000 | 4,128,768 | 48.22% |
| k16-random | 5,000 | 4,128,768 | 46.70% |
| k64-sample | 5,000 | 16,515,072 | 50.64% |
| k32-sample | 20,000 | 8,257,536 | 55.95% |
| k64-sample | 20,000 | 16,515,072 | 57.77% |

All runs use seed 0, batch size 4, maximum sequence length 768, BF16 base weights,
FP32 adapters, AdamW with learning rate 1e-4 and weight decay 0.1, and a cosine
schedule with 10% warmup. Gradient checkpointing and compilation are disabled.
Training uses the runner's filtered MetaMathQA dataset; evaluation uses 1,319
GSM8K test questions. Each training budget starts afresh with its own schedule.
The results are single-seed sanity checks, not multi-seed estimates. Compare
methods at the same step budget, model revision, and evaluation protocol.

Model revision: `d4446454d87d51aa42e1fb174f25acc5f8762331`.
MetaMathQA revision: `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`.
GSM8K revision: `740312add88f781978c0658806c59bc2815b9866`.
The runs used PyTorch 2.9.0, Transformers 5.17.0, and an NVIDIA H20, with the
Cortex extension on PEFT commit
`eb74a79a4b6ab7f1e4afe7a54a7d862c36d34ecc`.

The paper's context consolidation pipeline, including query generation and
teacher distillation, is separate from this supervised task benchmark.
