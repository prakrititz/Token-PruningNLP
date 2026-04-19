"""
train.py — Fine-tune CopyEnhancedCodeT5 on compressed Bugs2Fix data.
Custom NLL loss (model outputs P(y), not logits).
Memory-optimized for RTX 5060 (16 GB VRAM).
"""

import os
import json
import glob
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no window needed)
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AddedToken
import transformers.tokenization_utils_tokenizers as _tut

from model import CopyEnhancedCodeT5

# ─── MONKEYPATCH: transformers 5.5.4 + tokenizers 0.22.x bug ───
# CodeT5's special_tokens_map.json stores 100 <extra_id_*> tokens as JSON
# dicts.  transformers deserializes them but never converts them to AddedToken
# objects before passing them to the Rust tokenizers backend, which then raises
# "TypeError: Input must be a List[Union[str, AddedToken]]".
# This patch intercepts _add_tokens to do that conversion.
_orig_add_tokens = _tut.TokenizersBackend._add_tokens
def _patched_add_tokens(self, new_tokens, special_tokens=False):
    fixed = []
    for t in new_tokens:
        if isinstance(t, dict):
            t = AddedToken(
                t["content"],
                single_word=t.get("single_word", False),
                lstrip=t.get("lstrip", False),
                rstrip=t.get("rstrip", False),
                normalized=t.get("normalized", True),
                special=t.get("special", False),
            )
        fixed.append(t)
    return _orig_add_tokens(self, fixed, special_tokens)
_tut.TokenizersBackend._add_tokens = _patched_add_tokens

# ─────────────────────────────────────────────────────────────
# 0. HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
DATA_DIR          = "BugFixData_Compressed"
RATIOS            = [0.5]      # start with 1 ratio (~52K samples); add more later
MODEL_NAME        = "Salesforce/codet5-base"
MAX_LEN           = 512
BATCH_SIZE        = 4          # try 4 now that the forward pass is faster
GRAD_ACCUM_STEPS  = 8          # effective batch = 2 * 8 = 16
LEARNING_RATE     = 5e-5
WEIGHT_DECAY      = 0.01
EPOCHS            = 3
SAVE_DIR          = "checkpoints"
LOG_EVERY         = 20         # print loss every N optimizer-steps
SAVE_EVERY_EPOCH  = True
USE_AMP           = True       # mixed-precision (fp16) — saves ~40 % VRAM

SPECIAL_TOKENS = [
    AddedToken("<BUGS2FIX>", lstrip=False, rstrip=False),
    AddedToken("<Ratio>", lstrip=False, rstrip=False),
    AddedToken("</Ratio>", lstrip=False, rstrip=False),
    AddedToken("<Compress>", lstrip=False, rstrip=False),
    AddedToken("</Compress>", lstrip=False, rstrip=False),
]


# ─────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────
class Bugs2FixCompressedDataset(Dataset):
    """
    Reads one or more bug2fix_train_compress_{ratio}.jsonl files.
    Each sample becomes:
        encoder input : "<BUGS2FIX> <Ratio> {ratio} </Ratio> <Compress> {buggy} </Compress>"
        decoder target: "{fixed}"
    """

    def __init__(self, data_dir: str, split: str, ratios: list,
                 tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []  # list of (input_str, target_str)

        for ratio in ratios:
            path = os.path.join(data_dir, f"bug2fix_{split}_compress_{ratio}.jsonl")
            if not os.path.exists(path):
                print(f"[!] Skipping missing file: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    # ── build encoder input string ──
                    inp = (
                        f"<BUGS2FIX> <Ratio> {rec['compression_ratio']} </Ratio> "
                        f"<Compress> {rec['buggy']} </Compress>"
                    )
                    tgt = rec["fixed"]
                    self.samples.append((inp, tgt))

        print(f"[+] Loaded {len(self.samples)} samples ({split}, ratios={ratios})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp_str, tgt_str = self.samples[idx]

        # Tokenize encoder input
        enc = self.tokenizer(
            inp_str,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Tokenize decoder target
        dec = self.tokenizer(
            tgt_str,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze(0)            # [max_len]
        attention_mask  = enc["attention_mask"].squeeze(0)       # [max_len]
        labels         = dec["input_ids"].squeeze(0)             # [max_len]
        labels_mask    = dec["attention_mask"].squeeze(0)        # [max_len]

        # Replace padding token ids in labels with -100 so they are ignored in loss
        labels[labels_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask":  attention_mask,
            "labels":         labels,
            "labels_mask":    labels_mask,
        }


# ─────────────────────────────────────────────────────────────
# 2. CUSTOM NLL LOSS
# ─────────────────────────────────────────────────────────────
def compute_nll_loss(p_final, labels):
    """
    Negative-log-likelihood loss over the pointer-generator distribution.

    Why not CrossEntropyLoss?
    ─────────────────────────
    CrossEntropyLoss expects raw logits and internally applies log-softmax.
    Our CopyEnhancedCodeT5 returns P_final — a *probability* distribution
    (p_gen * P_vocab + (1-p_gen) * P_copy) that is already normalized.
    Passing probabilities to CE would double-softmax and produce garbage
    gradients.  Instead we compute NLL manually:

        NLL = -mean( log P_final(y_true) )    over non-padding tokens.

    Args:
        p_final : [batch, tgt_len, vocab_size]  — probabilities (sums to ~1).
        labels  : [batch, tgt_len]              — target token ids.
                  Padding positions are set to -100.

    Returns:
        Scalar loss (averaged over unmasked tokens).
    """

    # ── Step 1: Build a boolean mask for real (non-padding) tokens ──
    # labels == -100 marks positions that should NOT contribute to the loss
    # (padding injected by the tokenizer).
    mask = (labels != -100)                        # [batch, tgt_len]

    # ── Step 2: Replace -100 with 0 so gather doesn't index out of range ──
    # The values at these positions won't affect the loss because we mask them.
    safe_labels = labels.clone()
    safe_labels[~mask] = 0                         # [batch, tgt_len]

    # ── Step 3: Gather the probability assigned to each ground-truth token ──
    # p_final: [B, T, V]  →  target_probs: [B, T]
    # We index into the vocab dimension (dim=-1) with the target token ids.
    target_probs = p_final.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1)            # [B, T, 1]
    ).squeeze(-1)                                  # [B, T]

    # ── Step 4: Compute token-level NLL ──
    # Add epsilon (1e-8) to prevent log(0) → -inf which would cause NaN.
    token_nll = -torch.log(target_probs + 1e-8)    # [B, T]

    # ── Step 5: Zero out padding positions so they don't affect the mean ──
    token_nll = token_nll * mask.float()            # [B, T]

    # ── Step 6: Average over the real (unmasked) tokens only ──
    loss = token_nll.sum() / mask.float().sum().clamp(min=1.0)

    return loss


# ─────────────────────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")
    if device.type == "cuda":
        print(f"    GPU : {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 3a. Tokenizer ──
    # use_fast=False avoids a Rust tokenizers library bug that crashes during
    # __init__ when loading CodeT5's RoBERTa-based tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    # add_special_tokens({"additional_special_tokens": ...}) fails on the
    # RoBERTa-based CodeT5 tokenizer backend.  add_tokens(..., special_tokens=True)
    # achieves the same result without going through the broken codepath.
    tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    print(f"[+] Tokenizer vocab size (after special tokens): {len(tokenizer)}")

    # ── 3b. Model ──
    model = CopyEnhancedCodeT5(model_name=MODEL_NAME)

    # CRITICAL: resize embeddings so the 5 new special tokens get their own vectors
    model.model.resize_token_embeddings(len(tokenizer))
    model.vocab_size = len(tokenizer)

    # ── Memory optimization 1: Gradient checkpointing ──
    # Trades ~30 % more compute for ~40 % less activation memory.
    model.model.gradient_checkpointing_enable()

    model.to(device)

    # ── 3c. Dataset & DataLoader ──
    train_ds = Bugs2FixCompressedDataset(
        DATA_DIR, "train", RATIOS, tokenizer, MAX_LEN,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,       # 0 avoids Windows multiprocessing issues
        pin_memory=True,
    )

    # ── 3d. Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    # ── Memory optimization 2: Mixed-precision scaler ──
    # Keeps master weights in fp32 but runs forward/backward in fp16
    # ≈ halves activation + gradient memory on Ampere / Ada GPUs.
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    total_steps = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * EPOCHS
    print(f"[+] Training: {EPOCHS} epochs, {len(train_loader)} batches/epoch, "
          f"~{total_steps} optimizer steps")
    print(f"    Effective batch size = {BATCH_SIZE} × {GRAD_ACCUM_STEPS} = "
          f"{BATCH_SIZE * GRAD_ACCUM_STEPS}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Main loop ──
    global_step = 0
    loss_history = []          # (global_step, loss) for plotting

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        running_loss = 0.0      # smoothed loss for the progress bar
        running_count = 0
        t0 = time.time()

        # ── tqdm progress bar: shows loss + GPU memory live ──
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{EPOCHS}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch_idx, batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # ── Prepare decoder_input_ids ──
            # Standard T5 teacher-forcing: shift labels right.
            # decoder_input_ids[t] = labels[t-1], with BOS (pad_token_id) prepended.
            decoder_input_ids = labels.clone()
            decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id
            decoder_input_ids = torch.cat(
                [torch.full((labels.size(0), 1), tokenizer.pad_token_id,
                            device=device, dtype=labels.dtype),
                 decoder_input_ids[:, :-1]],
                dim=1,
            )

            # ── Memory optimization 3: Autocast (fp16 forward + backward) ──
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                # Forward → P_final: [B, T, V]  (probabilities, not logits)
                p_final = model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                )
                loss = compute_nll_loss(p_final, labels)
                # Scale loss by accumulation steps so the gradient magnitude
                # stays consistent regardless of how many micro-batches we use.
                loss = loss / GRAD_ACCUM_STEPS

            # ── Backward (scaled for mixed-precision) ──
            scaler.scale(loss).backward()

            # Track metrics (un-scaled loss)
            real_loss = loss.item() * GRAD_ACCUM_STEPS
            n_tokens  = (labels != -100).sum().item()
            epoch_loss   += real_loss * n_tokens
            epoch_tokens += n_tokens
            running_loss += real_loss
            running_count += 1

            # ── Memory optimization 4: Free the huge P_final tensor ASAP ──
            del p_final

            # ── Update progress bar every batch ──
            if running_count > 0:
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                pbar.set_postfix(
                    loss=f"{running_loss / running_count:.4f}",
                    gpu=f"{gpu_mem:.1f}G",
                    step=global_step,
                    refresh=False,
                )

            # ── Gradient accumulation boundary ──
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # ── Memory optimization 5: Gradient clipping ──
                # Prevents exploding gradients and the resulting spike in
                # optimizer state memory.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory
                global_step += 1

                # Record loss for plotting
                step_loss = running_loss / max(running_count, 1)
                loss_history.append((global_step, step_loss))
                running_loss = 0.0
                running_count = 0

        pbar.close()

        # ── Epoch summary ──
        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        elapsed = time.time() - t0
        print(f"[epoch {epoch}/{EPOCHS}] avg loss = {avg_epoch_loss:.4f}  "
              f"({elapsed:.0f}s)")

        # ── Save checkpoint ──
        if SAVE_EVERY_EPOCH:
            ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch}.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scaler":      scaler.state_dict(),
                "loss":        avg_epoch_loss,
            }, ckpt_path)
            print(f"    ✓ Saved {ckpt_path}")

        # ── Save loss curve plot ──
        if loss_history:
            steps, losses = zip(*loss_history)
            plt.figure(figsize=(10, 4))
            plt.plot(steps, losses, linewidth=0.5, alpha=0.8)
            plt.xlabel("Optimizer Step")
            plt.ylabel("NLL Loss")
            plt.title(f"Training Loss — Epoch {epoch}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(SAVE_DIR, "loss_curve.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"    ✓ Loss curve saved to {plot_path}")

        # ── Memory optimization 6: Clear cache between epochs ──
        torch.cuda.empty_cache()

    print("\n[✓] Training complete.")


if __name__ == "__main__":
    train()
