"""
plot_results.py - Generate publication-quality plots for CodePromptZip report

Reads results from ./results/ and generates:
  1. Compression Ratio vs CodeBLEU (line chart)
  2. Number of Shots vs CodeBLEU (grouped bar chart)
  3. Compression ON vs OFF comparison (bar chart)
  4. Token Savings visualization (bar chart)
  5. Combined summary dashboard

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results_dir ./results --output_dir ./results/plots
"""

import os
import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Style Configuration (Publication Quality)
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette (professional, colorblind-friendly)
COLORS = {
    "primary": "#2563EB",       # Blue
    "secondary": "#7C3AED",     # Purple
    "success": "#059669",       # Green
    "warning": "#D97706",       # Amber
    "danger": "#DC2626",        # Red
    "neutral": "#6B7280",       # Gray
    "accent": "#EC4899",        # Pink
}

GRADIENT_COLORS = ["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE", "#EFF6FF"]


def load_all_results(results_dir):
    """Load all JSON result files from the results directory."""
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        print("  Run: python scripts/run_all_evaluations.py first!")
        sys.exit(1)

    for json_file in sorted(results_path.glob("*.json")):
        if json_file.name == "all_results_summary.json":
            continue
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            results[json_file.stem] = data
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [WARN] Skipping {json_file.name}: {e}")

    print(f"Loaded {len(results)} result files from {results_dir}")
    return results


# ============================================================
# PLOT 1: Compression Ratio vs CodeBLEU
# ============================================================
def plot_compression_ratio_sweep(results, output_dir):
    """Line chart showing how CodeBLEU changes with compression ratio."""
    tau_values = []
    codebleu_values = []
    actual_tau_values = []

    # Collect data for tau sweep experiments
    for name, data in sorted(results.items()):
        if name.startswith("tau_") and name.endswith("_shots_1"):
            tau = data.get("tau_code", 0)
            codebleu = data.get("codebleu", data.get("exact_match", 0))
            actual_tau = data.get("actual_tau", 0)
            tau_values.append(tau)
            codebleu_values.append(codebleu)
            actual_tau_values.append(actual_tau)

    if len(tau_values) < 2:
        print("  [SKIP] Not enough tau sweep data for compression ratio plot")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis: CodeBLEU
    line1 = ax1.plot(tau_values, codebleu_values, 'o-', color=COLORS["primary"],
                     linewidth=2.5, markersize=8, label="CodeBLEU Score", zorder=5)
    ax1.fill_between(tau_values, codebleu_values, alpha=0.1, color=COLORS["primary"])
    ax1.set_xlabel("Target Compression Ratio (τ_code)")
    ax1.set_ylabel("CodeBLEU Score", color=COLORS["primary"])
    ax1.tick_params(axis="y", labelcolor=COLORS["primary"])

    # Secondary axis: Actual compression achieved
    if any(v > 0 for v in actual_tau_values):
        ax2 = ax1.twinx()
        line2 = ax2.plot(tau_values, actual_tau_values, 's--', color=COLORS["warning"],
                         linewidth=2, markersize=7, label="Actual Compression %", zorder=4)
        ax2.set_ylabel("Actual Compression Achieved (%)", color=COLORS["warning"])
        ax2.tick_params(axis="y", labelcolor=COLORS["warning"])

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", framealpha=0.9)
    else:
        ax1.legend(loc="upper right", framealpha=0.9)

    ax1.set_title("Effect of Compression Ratio on Code Generation Quality\n(Bugs2Fix, 1-shot RAG)")

    # Annotate key points
    if codebleu_values:
        best_idx = np.argmax(codebleu_values)
        ax1.annotate(
            f"Best: {codebleu_values[best_idx]:.1f}",
            xy=(tau_values[best_idx], codebleu_values[best_idx]),
            xytext=(10, 15), textcoords="offset points",
            fontsize=10, fontweight="bold", color=COLORS["success"],
            arrowprops=dict(arrowstyle="->", color=COLORS["success"]),
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "plot_compression_ratio_sweep.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# PLOT 2: Number of Shots vs CodeBLEU
# ============================================================
def plot_num_shots_comparison(results, output_dir):
    """Grouped bar chart comparing different shot counts."""

    # Collect compressed results
    compressed_data = {}
    uncompressed_data = {}

    for name, data in results.items():
        shots = data.get("num_shots", None)
        codebleu = data.get("codebleu", data.get("exact_match", 0))

        if name.startswith("tau_0.3_shots_"):
            compressed_data[shots] = codebleu
        elif name.startswith("no_compress_shots_"):
            uncompressed_data[shots] = codebleu

    # Also check for tau_0.0 as uncompressed
    if "tau_0.0_shots_1" in results:
        uncompressed_data[1] = results["tau_0.0_shots_1"].get("codebleu",
            results["tau_0.0_shots_1"].get("exact_match", 0))

    all_shots = sorted(set(list(compressed_data.keys()) + list(uncompressed_data.keys())))

    if len(all_shots) < 2:
        print("  [SKIP] Not enough shot variations for comparison plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(all_shots))
    width = 0.35

    comp_vals = [compressed_data.get(s, 0) for s in all_shots]
    uncomp_vals = [uncompressed_data.get(s, 0) for s in all_shots]

    bars1 = ax.bar(x - width/2, uncomp_vals, width, label="Without Compression",
                   color=COLORS["neutral"], edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, comp_vals, width, label="With Compression (τ=0.3)",
                   color=COLORS["primary"], edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold')

    ax.set_xlabel("Number of Retrieved Examples (k-shot)")
    ax.set_ylabel("CodeBLEU Score")
    ax.set_title("Impact of RAG Examples on Code Generation\n(With vs Without Compression)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}-shot" for s in all_shots])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "plot_num_shots_comparison.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# PLOT 3: Token Savings Visualization
# ============================================================
def plot_token_savings(results, output_dir):
    """Bar chart showing original vs compressed token counts."""
    experiments = []
    orig_tokens = []
    comp_tokens = []

    for name, data in sorted(results.items()):
        avg_orig = data.get("avg_orig_tokens", 0)
        avg_comp = data.get("avg_comp_tokens", 0)
        if avg_orig > 0 and avg_comp > 0 and avg_comp < avg_orig:
            tau = data.get("tau_code", 0)
            experiments.append(f"τ={tau}")
            orig_tokens.append(avg_orig)
            comp_tokens.append(avg_comp)

    if len(experiments) < 1:
        print("  [SKIP] Not enough token data for savings plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width/2, orig_tokens, width, label="Original Tokens",
                   color=COLORS["danger"], alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width/2, comp_tokens, width, label="Compressed Tokens",
                   color=COLORS["success"], alpha=0.8, edgecolor="white")

    # Add savings percentage
    for i, (orig, comp) in enumerate(zip(orig_tokens, comp_tokens)):
        savings = (1 - comp/orig) * 100
        ax.annotate(
            f"-{savings:.0f}%",
            xy=(x[i] + width/2, comp),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=11, fontweight="bold", color=COLORS["success"],
        )

    ax.set_xlabel("Compression Ratio Setting")
    ax.set_ylabel("Average Tokens per Demonstration")
    ax.set_title("Token Savings from Prompt Compression\n(Average tokens per retrieved example)")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "plot_token_savings.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# PLOT 4: Ablation Study Summary
# ============================================================
def plot_ablation_summary(results, output_dir):
    """Horizontal bar chart comparing all configurations."""
    names = []
    scores = []
    colors = []

    # Sort by CodeBLEU score
    scored_results = []
    for name, data in results.items():
        codebleu = data.get("codebleu", data.get("exact_match", 0))
        desc = data.get("description", name)
        is_compressed = data.get("use_compression", False)
        scored_results.append((name, codebleu, desc, is_compressed))

    scored_results.sort(key=lambda x: x[1], reverse=False)  # ascending for horizontal bars

    for name, score, desc, is_compressed in scored_results:
        # Use a cleaner label
        label = desc if desc else name
        names.append(label)
        scores.append(score)
        if is_compressed:
            colors.append(COLORS["primary"])
        else:
            colors.append(COLORS["neutral"])

    if len(names) < 2:
        print("  [SKIP] Not enough data for ablation summary")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.6)))

    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{score:.1f}', ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("CodeBLEU Score")
    ax.set_title("Ablation Study: All Configuration Results\n(Blue = With Compression, Gray = Without Compression)")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["primary"], label='With Compression'),
        Patch(facecolor=COLORS["neutral"], label='Without Compression'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "plot_ablation_summary.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# PLOT 5: Training Loss Curve
# ============================================================
def plot_training_curve(output_dir):
    """Parse training_output.txt and plot the loss curve."""
    log_file = "./logs/training_output.txt"
    if not os.path.exists(log_file):
        print("  [SKIP] No training log found at ./logs/training_output.txt")
        return

    epochs = []
    train_losses = []
    val_epochs = []
    val_losses = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):
                continue

            # Parse training loss lines
            if "Loss" in line and "Validation" not in line and "Epoch" in line:
                try:
                    parts = line.split("|")
                    epoch_part = parts[0].strip()
                    loss_part = [p for p in parts if "Loss" in p][0]
                    epoch = float(epoch_part.replace("Epoch", "").strip())
                    loss = float(loss_part.replace("Loss", "").strip())
                    epochs.append(epoch)
                    train_losses.append(loss)
                except (ValueError, IndexError):
                    continue

            # Parse validation loss lines
            elif "Validation Loss" in line and "Epoch" in line:
                try:
                    parts = line.split("|")
                    epoch_part = parts[0].strip()
                    val_part = [p for p in parts if "Validation" in p][0]
                    epoch = float(epoch_part.replace("Epoch", "").strip())
                    val_loss_str = val_part.replace("Validation Loss:", "").strip()
                    # Remove "(New Best!)" if present
                    val_loss_str = val_loss_str.replace("(New Best!)", "").strip()
                    val_loss = float(val_loss_str)
                    val_epochs.append(epoch)
                    val_losses.append(val_loss)
                except (ValueError, IndexError):
                    continue

    if len(epochs) < 5:
        print("  [SKIP] Not enough training data points to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot training loss (use a lighter line since there are many points)
    ax.plot(epochs, train_losses, '-', color=COLORS["primary"], alpha=0.6,
            linewidth=1, label="Training Loss")

    # Plot validation loss with markers
    if val_epochs:
        ax.plot(val_epochs, val_losses, 'o-', color=COLORS["danger"],
                linewidth=2, markersize=6, label="Validation Loss", zorder=5)

        # Mark best validation loss
        best_val_idx = np.argmin(val_losses)
        ax.annotate(
            f"Best: {val_losses[best_val_idx]:.4f}",
            xy=(val_epochs[best_val_idx], val_losses[best_val_idx]),
            xytext=(15, 15), textcoords="offset points",
            fontsize=10, fontweight="bold", color=COLORS["success"],
            arrowprops=dict(arrowstyle="->", color=COLORS["success"], lw=1.5),
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CopyCodeT5 Training & Validation Loss Curve\n(Bugs2Fix Compression Task)")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "plot_training_curve.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Generate Results Table (LaTeX & Markdown)
# ============================================================
def generate_results_table(results, output_dir):
    """Generate a formatted results table for the report."""

    # Markdown table
    md_lines = [
        "# CodePromptZip Evaluation Results\n",
        "## Results Table\n",
        "| Configuration | τ_code | Shots | Compression | CodeBLEU | Actual τ% | Avg Tokens (Orig→Comp) |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, data in sorted(results.items()):
        tau = data.get("tau_code", "N/A")
        shots = data.get("num_shots", "N/A")
        compressed = "✓" if data.get("use_compression", False) else "✗"
        codebleu = data.get("codebleu", data.get("exact_match", 0))
        actual_tau = data.get("actual_tau", "N/A")
        avg_orig = data.get("avg_orig_tokens", "N/A")
        avg_comp = data.get("avg_comp_tokens", "N/A")
        tokens_str = f"{avg_orig}→{avg_comp}" if avg_orig != "N/A" else "N/A"

        md_lines.append(
            f"| {name} | {tau} | {shots} | {compressed} | {codebleu:.1f} | {actual_tau} | {tokens_str} |"
        )

    # LaTeX table
    md_lines.extend([
        "\n## LaTeX Table (copy-paste into your report)\n",
        "```latex",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{CodePromptZip Evaluation Results on Bugs2Fix}",
        "\\label{tab:results}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Configuration & $\\tau_{code}$ & Shots & Compression & CodeBLEU & Token Savings \\\\",
        "\\midrule",
    ])

    for name, data in sorted(results.items()):
        tau = data.get("tau_code", "N/A")
        shots = data.get("num_shots", "N/A")
        compressed = "\\checkmark" if data.get("use_compression", False) else "\\texttimes"
        codebleu = data.get("codebleu", data.get("exact_match", 0))
        actual_tau = data.get("actual_tau", "N/A")
        actual_tau_str = f"{actual_tau}\\%" if actual_tau != "N/A" else "N/A"

        md_lines.append(
            f"{name} & {tau} & {shots} & {compressed} & {codebleu:.1f} & {actual_tau_str} \\\\"
        )

    md_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "```",
    ])

    save_path = os.path.join(output_dir, "results_table.md")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory containing result JSON files")
    parser.add_argument("--output_dir", type=str, default="./results/plots",
                        help="Directory to save plot images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CodePromptZip: Generating Report Plots")
    print("=" * 60)

    # Load all results
    results = load_all_results(args.results_dir)

    if not results:
        print("\n[ERROR] No results found! Run evaluations first:")
        print("  python scripts/run_all_evaluations.py")
        return

    # Generate all plots
    print("\n--- Generating Plots ---")

    print("\n[1/6] Training Loss Curve...")
    plot_training_curve(args.output_dir)

    print("\n[2/6] Compression Ratio Sweep...")
    plot_compression_ratio_sweep(results, args.output_dir)

    print("\n[3/6] Number of Shots Comparison...")
    plot_num_shots_comparison(results, args.output_dir)

    print("\n[4/6] Token Savings...")
    plot_token_savings(results, args.output_dir)

    print("\n[5/6] Ablation Summary...")
    plot_ablation_summary(results, args.output_dir)

    print("\n[6/6] Results Table (Markdown + LaTeX)...")
    generate_results_table(results, args.output_dir)

    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
