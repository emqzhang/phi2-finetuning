# Phi-2 Dialogue Summarization — QLoRA Ablation Study

**A controlled comparison of parameter-efficient fine-tuning strategies for dialogue summarization.** I adapted Microsoft's Phi-2 language model to summarize conversations, then compared a frozen 4-bit baseline against two Low-Rank Adaptation (LoRA) configurations to study the trade-off between model quality and trainable parameter count.

## Tools and concepts

`Python` · `PyTorch` · `Hugging Face Transformers` · `PEFT/LoRA` · `bitsandbytes` · `Datasets` · `ROUGE` · `Pandas` · `LLM Fine-Tuning` · `Experimental Design`

## Highlights

- **Model:** [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2), loaded in 4-bit NF4 precision.
- **Data:** [`neil-code/dialogsum-test`](https://huggingface.co/datasets/neil-code/dialogsum-test), dialogue-to-summary pairs (1,999 train / 499 validation / 499 test rows).
- **Methods:** 4-bit quantization, LoRA/PEFT, Hugging Face Transformers `Trainer`, ROUGE evaluation.
- **Experiment:** Baseline vs. adapters on **Q/K/V/dense** projections vs. **Q/V only**.
- **Result:** Broader LoRA reached **ROUGE-1 0.434** and **ROUGE-L 0.362**, compared with **0.419** and **0.313** for Q/V-only and **0.280** / **0.202** for the untuned baseline, on a 10-row (4-dialogue) evaluation sample.

## Motivation and research question

Full fine-tuning of large language models is expensive in GPU memory and compute. Quantization reduces base-model storage, while LoRA trains small low-rank updates rather than all model weights. I wanted to understand **whether expanding LoRA beyond query/value projections yields enough summarization improvement to justify additional adapter parameters**.

## Methodology

1. **Prepare the data.** Load dialogue/summary pairs and format them as instruction-following causal-language-model examples (`### Instruct: … ### Output: … ### End`). Tokenize with the Phi-2 tokenizer (max length 2,048) and shuffle with seed 67.
2. **Quantize the base model.** Load Phi-2 using 4-bit NF4 quantization with float16 compute (no double quantization) to fit training in a GPU-constrained environment.
3. **Evaluate the baseline.** Generate summaries from the pretrained model without adapter fine-tuning.
4. **Train two adapters.** Hold the training budget and LoRA hyperparameters constant while varying only the target projections: broader `q_proj`, `k_proj`, `v_proj`, `dense` versus smaller `q_proj`, `v_proj`.
5. **Evaluate and compare.** Generate summaries for the first **10 test rows** and compute ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum with one identical routine for all three models. Record adapter size, training time and final validation loss.

### Configuration

| Setting | Value |
| --- | --- |
| Base model | `microsoft/phi-2` |
| Quantization | 4-bit NF4, float16 compute |
| LoRA rank / alpha / dropout | 32 / 32 / 0.05 |
| Learning rate | 2e-4 (1 warm-up step) |
| Max training steps | 500 (≈ 1 epoch) |
| Micro-batch size / gradient accumulation | 1 / 4 |
| Optimizer | `paged_adamw_8bit` |
| Training seed / data-shuffle seed | 42 / 67 |
| Validation / checkpoint interval | every 50 steps |
| Decoding for evaluation | sampled (temperature 0.1, top-p 0.95, ≤ 100 new tokens), not seeded |
| Evaluation sample | first 10 test rows (4 distinct dialogues) |

## Results

These are the recorded outputs of the notebook in this repository (one run per configuration, run sequentially in the same session). ROUGE values are recomputed from the stored predictions with the same routine for all three models.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Adapter parameters | Training time | Final validation loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla Phi-2 (4-bit baseline) | 0.2797 | 0.0993 | 0.2023 | 0.2139 | 0 | — | — |
| LoRA: Q/K/V/dense | **0.4343** | **0.1546** | **0.3621** | **0.3619** | 20,971,520 | 46.80 min | 1.3261 |
| LoRA: Q/V only | 0.4189 | 0.1270 | 0.3126 | 0.3143 | 10,485,760 | 45.02 min | 1.3293 |

Training time is `trainer.train()` only (not model loading or evaluation).

**Interpretation.**

- Both adapters beat the baseline on all four ROUGE metrics.
- The broader adapter scored higher than Q/V-only on every metric (by 0.015 ROUGE-1, 0.028 ROUGE-2, 0.049 ROUGE-L, 0.048 ROUGE-Lsum). It uses **2× as many trainable adapter parameters** and trained about **1.8 minutes (3.8%) longer**.
- Relative to the baseline gain, Q/V-only keeps about 90% of the ROUGE-1 improvement, but only about 50% of ROUGE-2 and roughly 68–69% of ROUGE-L / ROUGE-Lsum.
- Final validation loss is nearly identical (1.3261 vs 1.3293), so the ROUGE gap between the two adapters should be read as suggestive, not conclusive.
- Part of the improvement over the baseline reflects output style: the fine-tuned models adopt the dataset's `#Person1#` / `#Person2#` conventions and short summaries, which ROUGE rewards.

**Why this matters:** changing *where* LoRA adapters are applied can change the quality/efficiency trade-off even when the underlying model and training recipe are held fixed.

## Engineering decisions and trade-offs

- **4-bit quantization:** reduce model memory requirements without training the full base model.
- **LoRA:** isolate trainable low-rank weights, enabling faster, cheaper experimentation.
- **Target-module ablation:** test whether a smaller attention-only adapter can compete with a broader projection-layer adapter.
- **Fixed training budget:** 500 steps (instead of 1000) and shared hyperparameters improve comparability; they do not guarantee identical training dynamics.
- **ROUGE:** provides an inexpensive, reproducible lexical-overlap signal, but does not directly assess factual accuracy, faithfulness, or summary usefulness.

## Limitations and future work

**This is a small exploratory experiment, not a production-grade benchmark.**

- The 10 scored test rows contain only **4 distinct dialogues** (each dialogue appears with several reference summaries), so the effective sample is small.
- One training seed per configuration, sampled and unseeded decoding, and no confidence intervals, so run-to-run noise could be comparable to the gap between the two adapters.
- Training uses a longer prompt template than the one used at evaluation time (applied equally to all three models).
- No human evaluation or factual-consistency checks.
- The trainable-parameter percentages printed in the notebook (1.36% and 0.68%) use the 4-bit packed parameter count (~1.5 B) as the denominator, so they overstate the share of the full ~2.8 B-parameter model; the absolute adapter sizes above are exact.
- In the saved notebook the baseline cell was executed before the final run of the helper-function cell, so its saved output may come from an earlier helper version (its raw ROUGE differs from the recomputed value in the third decimal). Use **Restart & Run All** for a clean record.
- Training time depends on GPU conditions; peak memory and throughput were not measured.

Next improvements would be to evaluate the full test split, run multiple seeds in a common environment, use deterministic decoding, align the training and evaluation prompts, inspect summaries for hallucinations and omitted facts, and report peak GPU memory and throughput alongside ROUGE.

## Repository contents and reproduction

```text
.
├── README.md
└── Phi2_QLoRA_Comparison.ipynb
```

Open `Phi2_QLoRA_Comparison.ipynb` in a **CUDA GPU** notebook environment and execute cells in order. The notebook installs dependencies (unpinned), prepares the data, evaluates the baseline, trains both LoRA configurations, and recomputes a unified ROUGE comparison. Each experiment writes its checkpoints, final adapter and a `results.json` to `phi2_lora_outputs/<experiment>/`. The notebook already contains the recorded outputs, so results can be inspected without rerunning GPU training. A fresh run can differ due to software versions, hardware, and sampled generation. Avoid committing large checkpoints or private tokens.
