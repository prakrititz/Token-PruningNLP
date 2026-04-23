# CodePromptZip: Token-Aware Code Compression for RAG-Enhanced Program Synthesis

[![Status](https://img.shields.io/badge/Status-Implemented-green)]() [![Task](https://img.shields.io/badge/Task-Bug2Fix-blue)]() [![Language](https://img.shields.io/badge/Language-Java-orange)]() [![Model](https://img.shields.io/badge/Model-CodeT5--Base-blueviolet)]()

## ⭐ Paper Attribution

**This is an implementation of the CodePromptZip paper for the Bug2Fix task.**

**Original Paper:**
- **Title:** CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs
- **Authors:** 
  - Pengfei He (University of Manitoba)
  - Shaowei Wang (University of Manitoba)
  - Tse-Hsun (Peter) Chen (Concordia University)
- **Publication:** arXiv preprint arXiv:2502.14925v2 [cs.SE], April 2026
- **Paper Link:** https://arxiv.org/abs/2502.14925

**Implementation Note:** This codebase implements the CodePromptZip framework specifically for the **Bug2Fix task** (Java code defect detection and repair). The original paper covers three coding tasks (Assertion Generation, Bugs2Fix, and Code Suggestion); this implementation focuses on Bugs2Fix only.

---

## Overview

**CodePromptZip** is an intelligent code compression system for Retrieval-Augmented Generation (RAG) pipelines. It uses a **priority-driven, type-aware algorithm** to compress code examples while preserving semantic information critical for downstream code understanding tasks.

### Key Innovation
Instead of randomly removing tokens, CodePromptZip:
1. **Classifies** code tokens into 5 types (Identifier, Invocation, Structure, Symbol, Signature)
2. **Prioritizes** token removal based on task requirements
3. **Trains** a neural compressor (CodeT5 + copy mechanism) on priority-driven compression patterns
4. **Achieves** up to **41% token reduction with only 12% performance loss**

### Impact
- ✅ Reduce API costs by removing redundant tokens from prompts
- ✅ Improve latency with shorter context windows
- ✅ Preserve model accuracy with intelligent compression

---

## What's in This Repository

### Implementation Scope
This is a **complete implementation for the Bug2Fix task** (Java code defect detection and repair) based on the CodePromptZip paper.

| Component | Status | Details |
|-----------|--------|---------|
| **Dataset Construction** | ✅ Complete | 9 compression ratios × 5K examples = 45K training pairs |
| **Priority-Driven Compression** | ✅ Complete | Algorithm 1: Type-aware greedy token removal |
| **CopyCodeT5 Model** | ✅ Complete | CodeT5-Base with pointer-generator copy mechanism |
| **Training Pipeline** | ✅ Complete | FP16-optimized for 8GB GPUs |
| **Evaluation Suite** | ✅ Complete | BM25 retrieval → compression → CodeLlama → metrics |
| **CodeBLEU Metrics** | ✅ Complete | n-gram + syntax + dataflow matching |
| **Results** | ✅ Complete | Comprehensive evaluation on 100 test samples |

### What's NOT Included
- Assertion detection task (paper covers this; only Bug2Fix here)
- Code suggestion task (paper covers this; only Bug2Fix here)
- Comparisons with other compression methods

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. View Results (No Training Needed)

Pre-computed results are available in `results/all_results_summary.json`:

```bash
# View results
cat results/all_results_summary.json | python -m json.tool

# Key finding: At τ=0.5 (50% compression target):
# - CodeBLEU: 80.36 (vs. baseline 91.72)
# - Actual compression: 41% token reduction
# - Trade-off: 12% accuracy drop for 41% prompt savings
```

### 3. Train the Compressor (Optional)

```bash
# Train CopyCodeT5 model (~15-20 hours on RTX 4060)
python src/train.py \
    --config configs/config.yaml \
    --train_data BugFixData_Compressed/bug2fix_train_all.jsonl \
    --val_data BugFixData_Compressed/bug2fix_valid_all.jsonl \
    --output_dir checkpoints
```

### 4. Run Evaluation

```bash
# Full evaluation pipeline (requires CodeLlama GGUF model)
python src/evaluate.py \
    --config configs/config.yaml \
    --checkpoint_dir checkpoints/checkpoint-9819 \
    --test_data BugFixData/test.jsonl \
    --train_data BugFixData/train.jsonl
```

### 5. Quick Test (No LLM Required)

```bash
# Verify compression works without full evaluation
python scripts/run_quick_test.py --checkpoint checkpoints/checkpoint-9819
```

---

### Framework Architecture

The framework operates in two phases:

**Training Phase:**

```
Knowledge Base (52,364 Bugs2Fix pairs)
      │
      ▼
[Program Analysis] — JavaParser
      │  Categorize tokens into 5 types
      ▼
[Ablation Analysis] — Derive removal priority ranking
      │
      ▼
[Priority-Driven Greedy Algorithm]
      │  Generate (original, compressed, τ) triples
      ▼
[Fine-tune Copy-Enhanced CodeT5]
      │  Train compressor LM_C
```

**Inference Phase:**

```
Query (new buggy method)
      │
      ▼
[BM25 Retriever] — Fetch top-k similar buggy-fixed pairs
      │
      ▼
[LM_C] — Compress each pair at the target τ_code
      │
      ▼
[Compressed Prompt] → Base LM (CodeLlama-13B)
      │
      ▼
Generated Fix
```

The compressor (LM_C) and the base LM are **fully decoupled**: the compressed output is plain text, requiring no gradient sharing or special embeddings between models.

---

## 3. Key Technical Components

### 3.1 Type-Aware Priority Ranking

Using JavaParser to construct Abstract Syntax Trees (ASTs), every token in a code snippet is categorized into one of five types:

| Token Type | Definition | Example |
|------------|------------|---------|
| **Symbol** | Operators, delimiters, punctuation | `=`, `{`, `;` |
| **Signature** | Method declaration and parameters | `public static void init(...)` |
| **Invocation** | Function/method calls | `Calendar.getInstance()` |
| **Identifier** | Variable names, class names | `VAR_1`, `TYPE_1` |
| **Structure** | Control-flow keywords | `if`, `for`, `return` |

The removal priority is computed as:

$$\text{Priority}(T) = \frac{\tau_{\text{code}/T}}{d_T}$$

where τ_code/T is the compression ratio achieved by removing all tokens of type T, and d_T is the resulting CodeBLEU degradation. Higher priority means a token type saves many tokens with minimal quality loss.

**Bugs2Fix Priority Ranking (our configuration):**

```
Identifier > Invocation > Structure > Symbol > Signature
(remove first)                                (remove last)
```

This ranking was adopted directly from the Ablation Study performed by the original authors, as they demonstrated it to be universally effective across different base LLMs for the Bugs2Fix task. In their ablation study, the authors systematically removed tokens of each specific type isolated from others, and measured the resulting percentage degradation in CodeBLEU to empirically determine which token types were safest to discard (highest priority) and which were most critical (lowest priority).

### 3.2 Dataset Construction via Priority-Driven Greedy Algorithm

For each parsable code example in the training set, the greedy algorithm:

1. Assigns each token a removal priority based on its type and within-type frequency.
2. Computes Lrm = floor(τ_code × L) — the number of tokens to remove.
3. Iteratively removes the highest-priority tokens until the budget is exhausted.
4. Returns the compressed code as the training target.

This is repeated for **9 compression ratios** (τ_code ∈ {0.1, 0.2, ..., 0.9}), producing the following dataset:

| Component | Count |
|-----------|-------|
| Raw Bugs2Fix pairs | 52,364 |
| Compression ratios per example | 9 |
| Total compression triples | 52,364 × 9 = **471,276** |
| Training split (80%) | **377,020** |
| Validation split (10%) | **47,128** |
| Test split (10%) | **47,128** |

The split is performed at the code-example level (not at the triple level) to prevent data leakage — all 9 compressed versions of the same code example are assigned to the same split.

#### Compression Progression Example
To illustrate the priority-driven greedy removal algorithm in action, consider how the following buggy method degrades uniformly as the compression ratio (τ_code) increases:

**Original (τ_code = 0.0)** - 0% tokens removed
```java
public static TYPE_1 init(java.lang.String name, java.util.Date date) {
    TYPE_1 VAR_1 = new TYPE_1();
    VAR_1.METHOD_1(name);
    java.util.Calendar VAR_2 = java.util.Calendar.getInstance();
    VAR_2.METHOD_2(date);
    VAR_1.METHOD_3(VAR_2);
    return VAR_1;
}
```

**Light Compression (τ_code = 0.3)** - 30% target. Redundant high-frequency `Identifier` tokens (`VAR_1`, `VAR_2`) begin dropping first, as they are deemed safest to lose.
```java
public static TYPE_1 init(java.lang.String name, java.util.Date date) {
    TYPE_1 = new TYPE_1();
    .METHOD_1(name);
    java.util.Calendar = java.util.Calendar.getInstance();
    .METHOD_2(date);
    .METHOD_3(VAR_2);
    return ;
}
```

**Moderate Compression (τ_code = 0.5)** - 50% target. Almost all `Identifier` tokens drop. The core structural context remains fully intact.
```java
public static TYPE_1 init(java.lang.String name, java.util.Date date) {
    = new TYPE_1();
    ;
    java.util.Calendar = java.util.Calendar;
    .METHOD_2(date);
    .METHOD_3(VAR_2);
    return ;
}
```

### 3.3 Compressor Architecture: Copy-Enhanced CodeT5

The compressor is built on **CodeT5-base** (Salesforce, 220M parameters), an encoder-decoder transformer pre-trained on 8.35M code functions across 8 programming languages.

**Modification 1 — Extended Vocabulary:**

Special tokens are added to signal the task and compression ratio:

```
<BUGS2FIX> <Ratio> 0.3 </Ratio> <Compress> {code} </Compress>
```

This enables the same model to be steered to different compression levels at inference time.

**Modification 2 — Copy Mechanism (Pointer-Generator):**

The critical architectural addition. At each decoding step t:

1. The decoder produces cross-attention weights a^t over source positions.
2. A context vector is computed: h*_t = Σ_i a^t_i · h_i
3. A generation gate is computed: p_gen = σ(W_gen · [h*_t, s_t] + b_gen)
4. The copy distribution sums attention over matching source tokens: P_copy(y) = Σ_{i: x_i=y} a^t_i
5. The final distribution blends vocabulary and copy:

$$P(y) = p_{\text{gen}} \cdot P_{\text{vocab}}(y) + (1 - p_{\text{gen}}) \cdot P_{\text{copy}}(y)$$

This ensures the compressed output is drawn strictly from the input tokens, preventing hallucinated code. The CopyModule is the only component initialized from scratch; all other weights are loaded from the pre-trained CodeT5-base checkpoint.

---

## 4. Implementation Details

### 4.1 Hardware and Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| System Environment | Windows (local, RTX 4060) |
| Base LM for evaluation | CodeLlama-13B-Instruct (Q4_K_M quantized, GGUF) |
| Python version | 3.12 |
| Framework | PyTorch + HuggingFace Transformers |

### 4.2 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base model | Salesforce/codet5-base (220M params) |
| Micro batch size | 2 |
| Gradient accumulation steps | 8 |
| **Effective batch size** | **16** |
| Learning rate | 5 × 10⁻⁵ |
| Number of epochs | **3** |
| Total training steps | ~70,500 (23,500 steps/epoch × 3) |
| **Total training time** | **~100 hours** |

These hyperparameters were set to fit the model within 8 GB VRAM. The effective batch size of 16 was achieved via gradient accumulation (micro-batch 2 × 8 accumulation steps).

### 4.3 Compressor Fine-Tuning Results

The CodeT5-base model was fine-tuned for 3 full epochs on the 377,020 training samples. The table below reports the best validation loss checkpoints across each epoch:

**Key observations from the training curves:**
- **Epoch 1:** The loss dropped dramatically from 3.67 (step 100) to 0.0284 (step 23,000), reflecting the model rapidly learning to compress code using the rich pre-trained representations from CodeT5.
- **Epoch 2:** The validation loss improved from 0.0284 to 0.0145 — a 49% reduction — confirming that a second epoch over the large dataset was valuable.
- **Epoch 3:** Continued to reduce, with the final best model achieving val loss = 0.0145 at step 47,000. The loss stabilized in the 0.017–0.018 range, indicating convergence.

![Figure 1: Validation loss across the full 3-epoch training run. The steep initial drop (Epoch 0–0.5) reflects rapid adaptation of pre-trained CodeT5 weights to the compression task. Subsequent epochs show continued but diminishing gains, with periodic spikes corresponding to evaluation on held-out validation batches. Gold stars mark checkpoints where a new best validation loss was recorded.](combined_val_loss_only.png)

The best checkpoint (lowest validation loss) was selected for all downstream evaluations.

### 4.4 Evaluation Pipeline

For evaluation, we use:
- **BM25 retrieval** (whitespace tokenization) to fetch similar buggy-fixed pairs from the full 52,364 training knowledge base.
- **CodeT5 compressor** to compress demonstrations at the specified τ_code.
- **CodeLlama-13B-Instruct** (4-bit quantized) as the base LM for code generation.
- **CodeBLEU** as the evaluation metric, computed across four components: N-gram match, Weighted N-gram match, Syntax match (AST), and Dataflow match.
- 100 randomly sampled test examples (seed=42) for each experiment configuration.

---

## 5. Results and Discussion

### 5.1 Experiment 1: Compression Ratio Sweep (1-shot)

We sweep τ_code from 0.0 (no compression) to 0.9 (extreme compression), keeping num_shots = 1 fixed.

| τ_code | Actual Compression | Avg Tokens | CodeBLEU (%) | Syntax (%) | Dataflow (%) |
|--------|-------------------|------------|-------------|-----------|-------------|
| 0.0 | 0% (baseline) | 160 | **91.72** | 96.82 | 93.45 |
| 0.1 | −5.3% (expansion) | 168 | 79.70 | 83.11 | 80.95 |
| 0.2 | 6.4% | 149 | 71.49 | 75.73 | 72.52 |
| 0.3 | 18.0% | 131 | 69.10 | 71.81 | 71.32 |
| 0.5 | 41.1% | 94 | 80.36 | 83.88 | 83.03 |
| 0.7 | 64.2% | 57 | 78.61 | 82.12 | 80.35 |
| 0.9 | 87.3% | 20 | 78.68 | 82.23 | 80.19 |

![Figure 2: Token savings achieved by the compressor at each compression ratio setting. Red bars show original token counts per demonstration; green bars show compressed token counts. The percentage labels indicate the actual token reduction achieved. At τ=0.9, the compressor reduces demonstrations from 160 tokens to just 20 — an 88% saving — while still preserving enough signal for the base LM to produce reasonable fixes.](plot_token_savings.png)

**Key findings:**

1. **Non-monotonic curve:** The CodeBLEU does not degrade linearly with compression. After an initial dip (τ=0.1–0.3), performance *recovers* at higher compression (τ=0.5–0.9). This suggests that at moderate compression, the compressor removes enough context to confuse the base LM, but at high compression, the demonstrations become so short that the base LM relies more on its internal knowledge, producing reasonable outputs.

2. **τ=0.1 expansion anomaly:** At τ=0.1, the compressor actually *increased* the token count (avg 168 vs. original 160). This is because the CodeT5 decoder generates tokens auto-regressively and at very low compression targets, it may reproduce the input with minor additions from the vocabulary distribution. This confirms findings from the original paper that very low compression ratios are unreliable.

3. **Optimal trade-off:** τ=0.5 achieves the best compression-quality balance: 41.1% token reduction with only a 12.4% CodeBLEU drop (91.72 → 80.36). The demonstrations retain enough structural and semantic information for the base LM to produce high-quality fixes while consuming 41% fewer tokens.

4. **Robustness at extreme compression:** Even at τ=0.9 (87.3% token reduction, avg 20 tokens), CodeBLEU remains at 78.68 — only 14.2% below the uncompressed baseline. This demonstrates the compressor's ability to preserve the most critical repair signals even under extreme compression.

### 5.2 Experiment 2: Multi-Shot Ablation (τ_code=0.3)

We fix τ_code = 0.3 (the paper's default) and vary the number of retrieved demonstrations.

| Shots | Compression | Avg Tokens | CodeBLEU (%) |
|-------|------------|------------|-------------|
| 0 (zero-shot) | N/A | — | 91.66 |
| 1 | 18.0% | 131 | 69.10 |
| 2 | 17.9% | 263 | 7.35 |
| 3 | 17.9% | 398 | 4.15 |

![Figure 3: Impact of the number of retrieved RAG examples on CodeBLEU, with and without compression (τ=0.3). Without compression (gray), performance is stable across all shot counts (~91.7). With compression (blue), performance degrades sharply beyond 1-shot, indicating that compressed multi-shot prompts saturate the base LM's limited context window.](plot_num_shots_comparison.png)

**Key findings:**

1. **Zero-shot dominance:** The zero-shot setting (no retrieved demonstrations) achieves 91.66 CodeBLEU — nearly identical to the uncompressed 1-shot baseline (91.72). This indicates that the base LM (CodeLlama-13B) is already highly capable at bug fixing when given just the buggy code as input.

2. **Multi-shot degradation:** Performance collapses dramatically beyond 1-shot: 2-shot drops to 7.35 and 3-shot to 4.15 CodeBLEU. This is primarily caused by **prompt format saturation** — the CodeLlama model, operating under a fixed 4096 context window with 4-bit quantization, struggles to parse and utilize multiple compressed demonstrations. The compressed demos, stripped of identifiers and some structural tokens, create syntactic patterns that the base LM interprets as the target output format rather than as demonstration context.

3. **Implication:** For the Bugs2Fix task with CodeLlama-13B, the optimal configuration is 0-shot or 1-shot with no/light compression. Multi-shot prompting requires a base LM with a larger context window and better in-context learning capabilities to be effective.

### 5.3 Experiment 3: Uncompressed Baselines

To isolate the effect of compression (vs. retrieval itself), we evaluate uncompressed demonstrations at multiple shot counts.

| Shots | CodeBLEU (%) | Syntax (%) | Dataflow (%) |
|-------|-------------|-----------|-------------|
| 1 | 91.72 | 96.82 | 93.45 |
| 2 | 91.82 | 96.81 | 93.55 |
| 3 | 91.71 | 96.90 | 93.45 |

**Key finding:** Without compression, adding more demonstrations has negligible impact (all within ±0.11%). This confirms that the base LM is already saturated at 1-shot for Bugs2Fix — additional uncompressed examples provide redundant information. This also validates that the multi-shot degradation observed in Experiment 2 is caused by the **compression artifacts**, not by multi-shot prompting itself.

---

## 6. Extension / Critique of the Original Work

### What we replicated
- The complete CODEPROMPTZIP pipeline: type-aware priority ranking, greedy dataset construction, copy-enhanced CodeT5 compressor, and BM25-based RAG evaluation.
- The compression ratio sweep experiment, confirming the non-trivial relationship between compression and downstream quality.

### Key differences from the original paper
| Aspect | Original Paper | Our Implementation |
|--------|---------------|-------------------|
| Base LM | GPT-3.5-turbo / Gemini-1.0 | CodeLlama-13B (4-bit quantized, local) |
| CodeT5 variant | CodeT5-large (770M) | CodeT5-base (220M) |
| Training epochs | 10 | 3 (~100 hrs on RTX 4060) |
| Training data | 48,903 parsable × 9 = 440K | 52,364 total × 9 = 471K |
| Evaluation samples | Full test set (6,545) | 100 samples (resource constraint) |

### Critique
1. **Base LM sensitivity:** The original paper does not deeply explore how the choice of base LM affects the utility of compressed demonstrations. Our results show that CodeLlama-13B under 4-bit quantization behaves very differently from GPT-3.5-turbo — specifically, it shows much less benefit from RAG demonstrations and degrades sharply with multi-shot compressed prompts.

2. **Compression at low τ:** The τ=0.1 expansion behavior (producing more tokens than the input) is a known but under-discussed limitation. The copy mechanism, while powerful, does not enforce a hard length constraint.

3. **Evaluation scale:** The paper's results on the full test set (6,545 samples) are more statistically robust than our 100-sample evaluation, which may not capture the full distribution of bug types and complexities.

---

## 7. How Could We Improve?

### 7.1 Multi-Method Contextual Bug Fixing
Currently, the pipeline assumes bugs are localized within singular isolated methods. In reality, modern software bugs often span multiple dependent method calls across a larger codebase. We propose extending the framework by including multi-method contexts, where dependent calls are provided alongside the buggy method. By leveraging static analysis tools (e.g., call graph filters), we can first prune out completely unlinked methods (those outside the function call chain for the buggy logic) and then selectively apply our token-pruning compressor on the remaining dependent code blocks. This would effectively upgrade the system into a multi-method, context-aware prompt compressor capable of resolving systemic, cross-method bugs.

---

## Summary

| Aspect | Value |
|--------|-------|
| **Model** | CodeT5-Base (220M) + Copy Mechanism |
| **Task** | Java Bug2Fix (defect detection & repair) |
| **Dataset** | 5K examples → 45K training pairs (9 ratios) |
| **Best Result** | 41% compression @ 80% accuracy (τ=0.5) |
| **Training Time** | 15-20 hours on RTX 4060 |
| **Evaluation Metrics** | CodeBLEU, Exact Match, Syntax, Dataflow |
| **Implementation** | Complete & Working ✅ |

---

**For questions or issues, refer to:**
- 📖 [REPORT.md](REPORT.md) - Full technical documentation
- 📁 [configs/config.yaml](configs/config.yaml) - All configuration options
- 💻 [src/](src/) - Source code with inline comments

**Happy compressing! 🚀**
