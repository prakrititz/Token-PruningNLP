"""
run_all_evaluations_linux.py - Run all evaluation configurations (Linux)

Identical to run_all_evaluations.py but calls src/evaluate_linux.py
which patches the tokenizer for Linux compatibility.

Usage:
    python scripts/run_all_evaluations_linux.py
    python scripts/run_all_evaluations_linux.py --max_eval_samples 200
    python scripts/run_all_evaluations_linux.py --skip_existing
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================
# Each experiment is a dict with a name, description, and CLI args.
# Results are saved to ./results/<experiment_name>.json

EXPERIMENTS = [
    # --------------------------------------------------------
    # GROUP 1: Compression Ratio Sweep (tau_code)
    # Purpose: Show how different compression levels affect quality
    # --------------------------------------------------------
    {
        "name": "tau_0.0_shots_1",
        "group": "compression_ratio",
        "description": "No compression baseline (tau=0.0, 1-shot)",
        "args": "--tau_code 0.0 --num_shots 1",
        "use_compression": False,  # tau=0 means no compression
    },
    {
        "name": "tau_0.1_shots_1",
        "group": "compression_ratio",
        "description": "Light compression (tau=0.1, 1-shot)",
        "args": "--tau_code 0.1 --num_shots 1",
    },
    {
        "name": "tau_0.2_shots_1",
        "group": "compression_ratio",
        "description": "Mild compression (tau=0.2, 1-shot)",
        "args": "--tau_code 0.2 --num_shots 1",
    },
    {
        "name": "tau_0.3_shots_1",
        "group": "compression_ratio",
        "description": "Moderate compression (tau=0.3, 1-shot) [Paper default]",
        "args": "--tau_code 0.3 --num_shots 1",
    },
    {
        "name": "tau_0.5_shots_1",
        "group": "compression_ratio",
        "description": "Heavy compression (tau=0.5, 1-shot)",
        "args": "--tau_code 0.5 --num_shots 1",
    },
    {
        "name": "tau_0.7_shots_1",
        "group": "compression_ratio",
        "description": "Aggressive compression (tau=0.7, 1-shot)",
        "args": "--tau_code 0.7 --num_shots 1",
    },
    {
        "name": "tau_0.9_shots_1",
        "group": "compression_ratio",
        "description": "Extreme compression (tau=0.9, 1-shot)",
        "args": "--tau_code 0.9 --num_shots 1",
    },

    # --------------------------------------------------------
    # GROUP 2: Number of Shots Sweep (num_shots)
    # Purpose: Show how more retrieved examples affect quality
    # Fixed tau=0.3 (the paper's recommended default)
    # --------------------------------------------------------
    {
        "name": "tau_0.3_shots_0",
        "group": "num_shots",
        "description": "Zero-shot (no retrieval, no compression)",
        "args": "--tau_code 0.3 --num_shots 0",
        "use_compression": False,
    },
    # tau_0.3_shots_1 already covered above
    {
        "name": "tau_0.3_shots_2",
        "group": "num_shots",
        "description": "2-shot with compression (tau=0.3)",
        "args": "--tau_code 0.3 --num_shots 2",
    },
    {
        "name": "tau_0.3_shots_3",
        "group": "num_shots",
        "description": "3-shot with compression (tau=0.3)",
        "args": "--tau_code 0.3 --num_shots 3",
    },

    # --------------------------------------------------------
    # GROUP 3: Compression ON vs OFF (with same num_shots)
    # Purpose: Directly prove compression helps
    # --------------------------------------------------------
    {
        "name": "no_compress_shots_1",
        "group": "ablation",
        "description": "1-shot WITHOUT compression",
        "args": "--tau_code 0.0 --num_shots 1",
        "use_compression": False,
    },
    {
        "name": "no_compress_shots_2",
        "group": "ablation",
        "description": "2-shot WITHOUT compression",
        "args": "--tau_code 0.0 --num_shots 2",
        "use_compression": False,
    },
    {
        "name": "no_compress_shots_3",
        "group": "ablation",
        "description": "3-shot WITHOUT compression",
        "args": "--tau_code 0.0 --num_shots 3",
        "use_compression": False,
    },
]


def run_experiment(exp, config_path, checkpoint_path, max_eval_samples, skip_existing=False):
    """Run a single evaluation experiment."""
    output_file = f"./results/{exp['name']}.json"

    # Skip if results already exist
    if skip_existing and os.path.exists(output_file):
        print(f"\n  [SKIP] {exp['name']} - results already exist")
        with open(output_file, "r") as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {exp['name']}")
    print(f"  {exp['description']}")
    print(f"{'='*60}")

    # Build the command — uses evaluate_linux.py instead of evaluate.py
    cmd = [
        sys.executable, "src/evaluate_linux.py",
        "--config", config_path,
        "--output_file", output_file,
        "--max_eval_samples", str(max_eval_samples),
    ]

    # Add checkpoint only if compression is used
    use_compression = exp.get("use_compression", True)
    if use_compression and checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])

    # Parse and add experiment-specific args
    extra_args = exp["args"].split()
    cmd.extend(extra_args)

    print(f"  Command: {' '.join(cmd)}")
    print()

    # Run the evaluation
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=False,
            timeout=7200,  # 2 hour timeout per experiment
        )

        if result.returncode != 0:
            print(f"  [FAIL] Experiment {exp['name']} failed with exit code {result.returncode}")
            return None

        # Load and return results
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                results = json.load(f)
            results["experiment_name"] = exp["name"]
            results["experiment_group"] = exp["group"]
            results["description"] = exp["description"]
            # Re-save with metadata
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            return results

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Experiment {exp['name']} timed out after 2 hours")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    return None


def main():
    parser = argparse.ArgumentParser(description="Run all evaluation experiments (Linux)")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model",
                        help="Path to trained compressor checkpoint")
    parser.add_argument("--max_eval_samples", type=int, default=100,
                        help="Max samples per experiment (default: 100 for quick run)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments that already have results")
    args = parser.parse_args()

    os.makedirs("./results", exist_ok=True)

    print("=" * 60)
    print("CodePromptZip: Full Evaluation Suite (Linux)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples per experiment: {args.max_eval_samples}")
    print(f"  Total experiments: {len(EXPERIMENTS)}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {}
    successful = 0
    failed = 0

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] Running: {exp['name']}")
        result = run_experiment(
            exp, args.config, args.checkpoint,
            args.max_eval_samples, args.skip_existing,
        )
        if result:
            all_results[exp["name"]] = result
            successful += 1
            # Print quick summary
            codebleu = result.get("codebleu", result.get("exact_match", "N/A"))
            print(f"  -> CodeBLEU: {codebleu}")
        else:
            failed += 1

    # Save consolidated results
    summary_file = "./results/all_results_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final summary table
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'CodeBLEU':>10} {'Tau':>8} {'Shots':>6} {'Compression':>12}")
    print("-" * 80)

    for name, result in all_results.items():
        codebleu = result.get("codebleu", result.get("exact_match", 0))
        tau = result.get("tau_code", "N/A")
        shots = result.get("num_shots", "N/A")
        compressed = "YES" if result.get("use_compression", False) else "NO"
        print(f"{name:<30} {codebleu:>10.2f} {tau:>8} {shots:>6} {compressed:>12}")

    print(f"\n  Successful: {successful}/{len(EXPERIMENTS)}")
    print(f"  Failed: {failed}/{len(EXPERIMENTS)}")
    print(f"  Results saved to: {summary_file}")
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  Next step: python scripts/plot_results.py")


if __name__ == "__main__":
    main()
