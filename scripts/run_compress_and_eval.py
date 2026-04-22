"""
run_compress_and_eval.py - Script to run compression and evaluation

Steps:
1. Load trained compressor
2. Load test data
3. Retrieve code examples via BM25
4. Compress retrieved examples
5. Generate with Base LM (CodeLlama-13B)
6. Evaluate with CodeBLEU / Exact Match
7. Generate result tables

Usage:
    python scripts/run_compress_and_eval.py --checkpoint ./checkpoints/best_model
    python scripts/run_compress_and_eval.py --checkpoint ./checkpoints/best_model --tau_code 0.3 --num_shots 1
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Compression + Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained compressor checkpoint")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--tau_code", type=float, default=0.3)
    parser.add_argument("--num_shots", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=2000,
                        help="Max eval samples (paper uses 2000)")
    parser.add_argument("--run_baselines", action="store_true",
                        help="Also run baseline comparisons")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.task:
        config["data"]["task"] = args.task

    task = config["data"]["task"]

    print("=" * 60)
    print("CodePromptZip: Compression + Evaluation Pipeline")
    print(f"  Task: {task}")
    print(f"  tau_code: {args.tau_code}")
    print(f"  Shots: {args.num_shots}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # Load test and train data
    raw_dir = Path(config["data"]["raw_dir"]) / task
    print("\nLoading data...")

    with open(raw_dir / "test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(raw_dir / "train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    # Initialize compressor
    print("\nLoading compressor...")
    from src.compress import CodeCompressor
    compressor = CodeCompressor(args.checkpoint)

    # Initialize evaluator
    print("\nInitializing evaluator...")
    from src.evaluate import Evaluator
    evaluator = Evaluator(config, compressor=compressor)

    all_results = {}

    # ============================================================
    # Main evaluation: CodePromptZip with compression
    # ============================================================
    print("\n" + "=" * 60)
    print("Running: CodePromptZip (with compression)")
    print("=" * 60)

    results_codepromptzip = evaluator.run_evaluation(
        test_data=test_data,
        train_data=train_data,
        tau_code=args.tau_code,
        num_shots=args.num_shots,
        max_eval_samples=args.max_eval_samples,
        use_compression=True,
    )
    all_results["CodePromptZip"] = results_codepromptzip

    # ============================================================
    # Baselines (optional)
    # ============================================================
    if args.run_baselines:
        # Without compression baseline
        print("\n" + "=" * 60)
        print("Running: Without Compression Baseline")
        print("=" * 60)

        results_no_comp = evaluator.run_no_compression_baseline(
            test_data=test_data,
            train_data=train_data,
            num_shots=args.num_shots,
            max_eval_samples=args.max_eval_samples,
        )
        all_results["w/o Compression"] = results_no_comp

        # Without retrieval baseline
        print("\n" + "=" * 60)
        print("Running: Without Retrieval Baseline")
        print("=" * 60)

        results_no_ret = evaluator.run_no_retrieval_baseline(
            test_data=test_data,
            max_eval_samples=args.max_eval_samples,
        )
        all_results["w/o Retrieval"] = results_no_ret

    # ============================================================
    # Save and print results
    # ============================================================
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"results_{task}_tau{args.tau_code}_shots{args.num_shots}.json"
    )

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    # Print table
    print(f"\n{'Approach':<30} ", end="")
    if task == "assertion":
        print(f"{'tau(%)':>8} {'Exact Match(%)':>15}")
    else:
        print(f"{'tau(%)':>8} {'CodeBLEU(%)':>12}")

    print("-" * 60)

    for name, result in all_results.items():
        tau = result.get("actual_tau", result.get("tau_code", 0) * 100)
        if task == "assertion":
            metric = result.get("exact_match", 0)
            print(f"{name:<30} {tau:>8.1f} {metric:>15.1f}")
        else:
            metric = result.get("codebleu", 0)
            print(f"{name:<30} {tau:>8.1f} {metric:>12.1f}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
