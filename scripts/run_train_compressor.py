"""
run_train_compressor.py - Script to run the full training pipeline

Steps:
1. Download Bugs2Fix dataset
2. Build compression training dataset (Algorithm 1)
3. Train CopyCodeT5 compressor

Usage:
    python scripts/run_train_compressor.py
    python scripts/run_train_compressor.py --task bugs2fix --max_examples 1000
"""

import os
import sys
import argparse
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Full training pipeline")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--task", type=str, default="bugs2fix")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit examples for debugging (e.g., 100)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip dataset download")
    parser.add_argument("--skip_dataset_construction", action="store_true",
                        help="Skip compression dataset construction")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("CodePromptZip: Full Training Pipeline")
    print(f"Task: {args.task}")
    print("=" * 60)

    # ============================================================
    # Step 1: Download dataset
    # ============================================================
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 1: Download Dataset")
        print("=" * 60)
        from data.download_datasets import download_bugs2fix
        download_bugs2fix(config["data"]["raw_dir"])
    else:
        print("\n[Skipping dataset download]")

    # ============================================================
    # Step 2: Build compression dataset
    # ============================================================
    if not args.skip_dataset_construction:
        print("\n" + "=" * 60)
        print("STEP 2: Build Compression Training Dataset")
        print("=" * 60)
        from src.dataset_construction import (
            load_raw_data, extract_code_examples,
            build_compression_dataset, split_dataset, save_dataset,
        )

        # Load raw data
        raw_data = load_raw_data(config["data"]["raw_dir"], args.task)

        # Extract parsable code examples from training set
        code_examples = extract_code_examples(raw_data["train"], args.task)
        print(f"  Extracted {len(code_examples)} code examples from training set")

        # Build compression dataset
        dataset = build_compression_dataset(
            code_examples, args.task,
            max_examples=args.max_examples,
        )

        # Split and save
        output_dir = os.path.join(config["data"]["compression_dataset_dir"], args.task)
        splits = split_dataset(dataset, seed=config["training"]["seed"])
        save_dataset(splits, output_dir)
    else:
        print("\n[Skipping dataset construction]")

    # ============================================================
    # Step 3: Train compressor
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Train CopyCodeT5 Compressor")
    print("=" * 60)

    from src.tokenizer_utils import get_extended_tokenizer
    from src.model.copy_codet5 import create_model
    from src.train import CompressionDataset, Trainer

    model_name = config["model"]["compressor_name"]

    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = get_extended_tokenizer(model_name)

    # Load compression datasets
    data_dir = os.path.join(config["data"]["compression_dataset_dir"], args.task)

    print("\nLoading training data...")
    train_dataset = CompressionDataset(
        os.path.join(data_dir, "train.json"),
        tokenizer,
        config["model"]["max_source_length"],
        config["model"]["max_target_length"],
    )

    print("Loading validation data...")
    val_dataset = CompressionDataset(
        os.path.join(data_dir, "validation.json"),
        tokenizer,
        config["model"]["max_source_length"],
        config["model"]["max_target_length"],
    )

    # Create model
    print("\nCreating model...")
    if args.resume:
        from src.model.copy_codet5 import CopyCodeT5
        model = CopyCodeT5.from_pretrained(args.resume, tokenizer=tokenizer)
        print(f"  Resumed from checkpoint: {args.resume}")
    else:
        model = create_model(
            model_name=model_name,
            use_copy=config["model"]["use_copy_mechanism"],
            tokenizer=tokenizer,
            gradient_checkpointing=config["training"]["gradient_checkpointing"],
        )

    # Train!
    print("\nStarting training...")
    trainer = Trainer(model, tokenizer, train_dataset, val_dataset, config)
    trainer.train()

    print("\n" + "=" * 60)
    print("Full Training Pipeline Complete!")
    print("=" * 60)
    print(f"\nBest model saved to: {config['training']['output_dir']}/best_model")
    print(f"Final model saved to: {config['training']['output_dir']}/final_model")


if __name__ == "__main__":
    main()
