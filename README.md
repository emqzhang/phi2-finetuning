# Phi-2 Dialogue Summarization — QLoRA Ablation Study

**A controlled comparison of parameter-efficient fine-tuning strategies for dialogue summarization.** I adapted Microsoft's Phi-2 language model to summarize conversations, then compared a frozen 4-bit baseline against two Low-Rank Adaptation (LoRA) configurations to study the trade-off between model quality and trainable parameter count.

## Highlights

- **Model:** [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2), loaded in 4-bit NF4 precision.
- **Data:** [`neil-code/dialogsum-test`](https://huggingface.co/datasets/neil-code/dialogsum-test), dialogue-to-summary pairs.
- **Methods:** 4-bit quantization, LoRA/PEFT, Hugging Face Transformers `Trainer`, ROUGE evaluation.
- **Experiment:** Baseline vs. adapters on **Q/K/V/dense** projections vs. **Q/V only**.
- **Result:** Broader LoRA reached **ROUGE-1 0.365** and **ROUGE-L 0.297**, compared with **0.322** and **0.243** for Q/V-only, on the recorded 10-example evaluation.

## Motivation and research question

Full fine-tuning of large language models is expensive in GPU memory and compute. Quantization reduces base-model storage, while LoRA trains small low-rank updates rather than all model weights. I wanted to understand **whether expanding LoRA beyond query/value projections yields enough summarization improvement to justify additional adapter parameters**.

## Methodology

1. **Prepare the data.** Load dialogue/summary pairs and format them as instruction-following causal-language-model examples. Tokenize the examples using the Phi-2 tokenizer and the original notebooks' preprocessing functions.
2. **Quantize the base model.** Load Phi-2 using 4-bit NF4 quantization with float16 compute to fit training in a GPU-constrained environment.
3. **Evaluate the baseline.** Generate summaries from the pretrained model without adapter fine-tuning.
4. **Train two adapters.** Hold the training budget and LoRA hyperparameters constant while varying only the target projections: broader `q_proj`, `k_proj`, `v_proj`, `dense` versus smaller `q_proj`, `v_proj`.
5. **Evaluate and compare.** Generate summaries for the first **10 held-out test examples** and compute ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum. Record adapter size and training time.

### Configuration

| Setting | Value |
| --- | --- |
| Base model | `microsoft/phi-2` |
| Quantization | 4-bit NF4, float16 compute |
| LoRA rank / alpha / dropout | 32 / 32 / 0.05 |
| Learning rate | 2e-4 |
| Max training steps | 500 |
| Micro-batch size / gradient accumulation | 1 / 4 |
| Optimizer | `paged_adamw_8bit` |
| Seed | 42 |
| Evaluation sample | 10 test examples |

## Results

These are **recorded outputs from the two supplied notebooks**, not results of a new execution of the merged notebook. The runs were performed separately, and their independently generated baseline ROUGE values differed slightly.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Adapter parameters | Training time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla Phi-2 (Experiment B baseline) | 0.2884 | 0.1074 | 0.2086 | 0.2198 | 0 | — |
| LoRA: Q/K/V/dense (Experiment A) | **0.3648** | **0.1221** | **0.2974** | **0.2983** | 20,971,520 | 47.8 min |
| LoRA: Q/V only (Experiment B) | 0.3223 | 0.1079 | 0.2433 | 0.2535 | 10,485,760 | 45.0 min |

**Interpretation.** The broader adapter delivered higher ROUGE across all reported metrics, while using **2× as many trainable adapter parameters** and approximately **2.8 additional minutes** of recorded training time. Q/V-only provided a smaller adapter and better scores than its same-run baseline on ROUGE-1, ROUGE-L, and ROUGE-Lsum; ROUGE-2 barely changed. The broader model's within-run baseline in Experiment A had ROUGE-1 **0.2889**, ROUGE-2 **0.1036**, ROUGE-L **0.2096**, and ROUGE-Lsum **0.2202**.

**Why this matters:** changing *where* LoRA adapters are applied can change the quality/efficiency trade-off even when the underlying model and training recipe are held fixed.

## Engineering decisions and trade-offs

- **4-bit quantization:** reduce model memory requirements without training the full base model.
- **LoRA:** isolate trainable low-rank weights, enabling faster, cheaper experimentation.
- **Target-module ablation:** test whether a smaller attention-only adapter can compete with a broader projection-layer adapter.
- **Fixed training budget:** 500 steps and shared hyperparameters improve comparability; they do not guarantee identical training dynamics.
- **ROUGE:** provides an inexpensive, reproducible lexical-overlap signal, but does not directly assess factual accuracy, faithfulness, or summary usefulness.

## Limitations and future work

**This is a small exploratory experiment, not a production-grade benchmark.** Only 10 test examples were scored, the original experiments ran in separate environments/accounts, and the baseline scores differed slightly. Training time depends on GPU conditions; no uncertainty intervals, repeated seeds, human evaluation, or factual-consistency measurements were reported. The original baseline parameter counter also included parameters marked trainable by PyTorch even though **no baseline fine-tuning occurred**; baseline adapter parameters are correctly shown as zero above.

Next improvements would be to evaluate the full held-out split, run multiple seeds in a common environment, standardize deterministic decoding, inspect summaries for hallucinations and omitted facts, and report peak GPU memory and throughput alongside ROUGE.

## Repository contents and reproduction

```text
.
├── README.md
└── Phi2_QLoRA_Comparison.ipynb
```

Open `Phi2_QLoRA_Comparison.ipynb` in a **CUDA GPU** notebook environment and execute cells in order. The notebook installs dependencies, prepares the data, evaluates the baseline, trains both LoRA configurations, and exports comparison CSV files to `phi2_lora_outputs/`. It also includes a separate historical-results table so readers can inspect the originally observed results **without rerunning GPU training**. A fresh run can differ due to software versions, hardware, and generation stochasticity. Avoid committing large checkpoints or private tokens.

## Tools and concepts

`Python` · `PyTorch` · `Hugging Face Transformers` · `PEFT/LoRA` · `bitsandbytes` · `Datasets` · `ROUGE` · `Pandas` · `LLM Fine-Tuning` · `Experimental Design`
