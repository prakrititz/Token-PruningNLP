"""
train.py - Training Pipeline for CodePromptZip Compressor

Trains the CopyCodeT5 model on the compression dataset.
Optimized for RTX 4060 (8GB VRAM) with:
- Mixed precision (FP16)
- Gradient checkpointing
- Gradient accumulation
- Small batch size

Hyperparameters from paper (Section 6):
    Optimizer: AdamW
    LR: 5e-5
    Warmup: 1000 steps
    Epochs: 10
    Batch size: 16 (effective, via accumulation)
"""

import os
import json
import math
import time
import yaml
import argparse
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.amp import GradScaler
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm

from src.tokenizer_utils import get_extended_tokenizer
from src.model.copy_codet5 import create_model


class CompressionDataset(Dataset):
    """
    PyTorch Dataset for code compression training.
    Each sample maps (input_text -> target_text) where:
        input_text:  <TASK> <Ratio> tau </Ratio> <Compress> original_code </Compress>
        target_text: <Compress> compressed_code </Compress>
    """

    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 512,
    ):
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        print(f"  Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize input
        source = self.tokenizer(
            item["input_text"],
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize target
        target = self.tokenizer(
            item["target_text"],
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Labels: replace padding token IDs with -100 for loss masking
        labels = target["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
            "decoder_input_ids": self._shift_right(target["input_ids"].squeeze(0)),
        }

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shift input IDs right for decoder input (teacher forcing)."""
        decoder_start_token_id = self.tokenizer.pad_token_id
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[1:] = input_ids[:-1].clone()
        shifted[0] = decoder_start_token_id
        return shifted


class Trainer:
    """
    Training loop for the CopyCodeT5 compressor.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: CompressionDataset,
        val_dataset: Optional[CompressionDataset],
        config: Dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Scheduler
        num_training_steps = (
            len(self.train_loader)
            * config["training"]["num_epochs"]
            // config["training"]["gradient_accumulation_steps"]
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["training"]["warmup_steps"],
            num_training_steps=num_training_steps,
        )

        # Mixed precision
        self.scaler = GradScaler('cuda', enabled=config["training"]["fp16"])
        self.fp16 = config["training"]["fp16"]

        # Gradient accumulation
        self.grad_accum_steps = config["training"]["gradient_accumulation_steps"]

        # Logging
        self.output_dir = config["training"]["output_dir"]
        self.log_dir = config["training"]["log_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        print(f"\n[Trainer] Device: {self.device}")
        print(f"[Trainer] Batch size: {config['training']['batch_size']}")
        print(f"[Trainer] Gradient accumulation: {self.grad_accum_steps}")
        print(f"[Trainer] Effective batch size: {config['training']['batch_size'] * self.grad_accum_steps}")
        print(f"[Trainer] Total training steps: {num_training_steps}")
        print(f"[Trainer] FP16: {self.fp16}")

    def train(self):
        """Main training loop."""
        num_epochs = self.config["training"]["num_epochs"]
        logging_steps = self.config["training"]["logging_steps"]
        save_steps = self.config["training"]["save_steps"]
        eval_steps = self.config["training"]["eval_steps"]

        print(f"\n{'='*60}")
        print(f"Starting Training for {num_epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.time()

            progress = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch+1}/{num_epochs}",
            )

            for batch_idx, batch in progress:
                loss = self._training_step(batch)
                epoch_loss += loss

                # Update weights after accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    if self.fp16:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        # Only step scheduler if optimizer actually stepped (no inf/nan grads)
                        if scale_before <= scale_after:
                            self.scheduler.step()
                    else:
                        self.optimizer.step()
                        self.scheduler.step()

                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % logging_steps == 0:
                        avg_loss = epoch_loss / (batch_idx + 1)
                        lr = self.scheduler.get_last_lr()[0]
                        progress.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "step": self.global_step,
                        })
                        self.train_losses.append({
                            "step": self.global_step,
                            "loss": avg_loss,
                            "lr": lr,
                        })

                        # Save text log
                        frac_epoch = epoch + (batch_idx + 1) / len(self.train_loader)
                        with open(os.path.join(self.log_dir, "training_output.txt"), "a") as f:
                            f.write(f"Epoch {frac_epoch:.4f} | Step {self.global_step} | Loss {avg_loss:.4f} | LR {lr:.2e}\n")

                    # Save checkpoint
                    if self.global_step % save_steps == 0:
                        self._save_checkpoint(f"checkpoint-{self.global_step}")

                    # Evaluation
                    if self.val_loader and self.global_step % eval_steps == 0:
                        val_loss = self._evaluate()
                        self.val_losses.append({
                            "step": self.global_step,
                            "val_loss": val_loss,
                        })
                        
                        frac_epoch = epoch + (batch_idx + 1) / len(self.train_loader)
                        log_msg = f"Epoch {frac_epoch:.4f} | Step {self.global_step} | Validation Loss: {val_loss:.4f}"
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_checkpoint("best_model")
                            print(f"\n  ★ New best val loss: {val_loss:.4f}")
                            log_msg += " (New Best!)"
                            
                        with open(os.path.join(self.log_dir, "training_output.txt"), "a") as f:
                            f.write(f"\n{log_msg}\n\n")
                        self.model.train()

            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"\n  Epoch {epoch+1} complete: avg_loss={avg_epoch_loss:.4f}, "
                  f"time={epoch_time:.1f}s")

            # Evaluate at end of each epoch
            if self.val_loader:
                val_loss = self._evaluate()
                print(f"  Validation loss: {val_loss:.4f}")
                
                log_msg = f"Epoch {epoch+1.0:.4f} | End of Epoch | Validation Loss: {val_loss:.4f}"
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best_model")
                    print(f"  ★ New best val loss: {val_loss:.4f}")
                    log_msg += " (New Best!)"
                    
                with open(os.path.join(self.log_dir, "training_output.txt"), "a") as f:
                    f.write(f"\n{log_msg}\n\n")

            # Save epoch checkpoint
            self._save_checkpoint(f"epoch-{epoch+1}")

        # Save final model
        self._save_checkpoint("final_model")
        self._save_training_log()
        print(f"\n{'='*60}")
        print(f"Training complete! Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")

    def _training_step(self, batch: Dict) -> float:
        """Single training step."""
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=self.fp16):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                labels=batch["labels"],
            )
            loss = outputs["loss"] / self.grad_accum_steps

        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.grad_accum_steps

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=self.fp16):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    labels=batch["labels"],
                )

            total_loss += outputs["loss"].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_dir = os.path.join(self.output_dir, name)
        print(f"\n  Saving checkpoint to {save_dir}...")
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, os.path.join(save_dir, "training_state.pt"))

    def _save_training_log(self):
        """Save training metrics to JSON."""
        log_path = os.path.join(self.log_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss,
                "global_step": self.global_step,
            }, f, indent=2)
        print(f"  Training log saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CodePromptZip Compressor")
    parser.add_argument("--config", type=str, default="./configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--task", type=str, default=None,
                        help="Override task from config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.task:
        config["data"]["task"] = args.task

    task = config["data"]["task"]
    print(f"\n{'='*60}")
    print(f"Training CodePromptZip Compressor for: {task}")
    print(f"{'='*60}")

    # Initialize tokenizer
    print("\n[1/4] Loading tokenizer...")
    model_name = config["model"]["compressor_name"]
    tokenizer = get_extended_tokenizer(model_name)

    # Load datasets
    print("\n[2/4] Loading datasets...")
    data_dir = os.path.join(config["data"]["compression_dataset_dir"], task)
    train_dataset = CompressionDataset(
        os.path.join(data_dir, "train.json"),
        tokenizer,
        config["model"]["max_source_length"],
        config["model"]["max_target_length"],
    )
    val_dataset = CompressionDataset(
        os.path.join(data_dir, "validation.json"),
        tokenizer,
        config["model"]["max_source_length"],
        config["model"]["max_target_length"],
    )

    # Create model
    print("\n[3/4] Creating model...")
    if args.resume:
        from src.model.copy_codet5 import CopyCodeT5
        model = CopyCodeT5.from_pretrained(args.resume, tokenizer=tokenizer)
        print(f"  Resumed from {args.resume}")
    else:
        model = create_model(
            model_name=model_name,
            use_copy=config["model"]["use_copy_mechanism"],
            tokenizer=tokenizer,
            gradient_checkpointing=config["training"]["gradient_checkpointing"],
        )

    # Train
    print("\n[4/4] Starting training...")
    trainer = Trainer(model, tokenizer, train_dataset, val_dataset, config)
    trainer.train()


if __name__ == "__main__":
    main()
