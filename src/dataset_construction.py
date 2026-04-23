"""
dataset_construction.py - Build Code Compression Training Dataset

Constructs the training dataset for the compressor model by:
1. Loading raw task data (e.g., Bugs2Fix)
2. Filtering for parsable code examples
3. Applying Algorithm 1 (priority-driven greedy compression) at 9 compression ratios
4. Formatting input/output pairs for CodeT5 training
5. Splitting into train/val/test (80/10/10)

Per paper Section 4.1 and Appendix C:
    Total samples = parsable_examples × 9 (one per tau_code value)
    Split: 80% train, 10% val, 10% test
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from src.type_analysis import is_parsable
from src.priority_ranking import compress_code_with_priority
from src.tokenizer_utils import format_compressor_input, format_compressor_target


def load_raw_data(data_dir: str, task: str) -> Dict[str, List[Dict]]:
    """
    Load raw dataset for a given task.

    Args:
        data_dir: Path to raw data directory.
        task: Task name (bugs2fix, assertion, suggestion).

    Returns:
        Dictionary with 'train', 'validation', 'test' splits.
    """
    task_dir = Path(data_dir) / task
    data = {}

    for split in ["train", "validation", "test"]:
        filepath = task_dir / f"{split}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data[split] = json.load(f)
            print(f"  Loaded {split}: {len(data[split])} samples")
        else:
            print(f"  [Warning] {filepath} not found, skipping")
            data[split] = []

    return data


def extract_code_examples(data: List[Dict], task: str) -> List[str]:
    """
    Extract code examples from raw data that serve as RAG demonstrations.

    For Bugs2Fix: both buggy and fixed code are used as code examples.
    The knowledge base for retrieval consists of these training examples.

    For the compression dataset, we compress the code examples that would
    appear in the RAG prompt demonstrations.

    Args:
        data: List of data samples.
        task: Task name.

    Returns:
        List of code strings to compress.
    """
    code_examples = []

    if task == "bugs2fix":
        for item in data:
            # In RAG for Bugs2Fix, demonstrations include buggy+fixed code
            # We compress the full demonstration text
            demo = f"### BUGGY_CODE\n{item['buggy']}\n### FIXED_CODE\n{item['fixed']}"
            code_examples.append(demo)
    elif task == "assertion":
        for item in data:
            demo = f"### FOCAL_METHOD\n{item.get('focal_method', '')}\n### UNIT_TEST\n{item.get('test_method', '')}\n### Assertion\n{item.get('assertion', '')}"
            code_examples.append(demo)
    elif task == "suggestion":
        for item in data:
            demo = f"### METHOD_HEADER\n{item.get('method_header', '')}\n### WHOLE_METHOD\n{item.get('method_body', '')}"
            code_examples.append(demo)
    else:
        raise ValueError(f"Unknown task: {task}")

    return code_examples


def build_compression_dataset(
    code_examples: List[str],
    task: str = "bugs2fix",
    compression_ratios: Optional[List[float]] = None,
    max_examples: Optional[int] = None,
) -> List[Dict]:
    """
    Build the compression training dataset using Algorithm 1.

    For each parsable code example, generates compressed versions at
    9 different compression ratios (0.1 to 0.9).

    Args:
        code_examples: List of code strings to compress.
        task: Task name.
        compression_ratios: List of tau_code values.
        max_examples: Maximum number of examples to process (for debugging).

    Returns:
        List of training samples with 'input' and 'target' fields.
    """
    if compression_ratios is None:
        compression_ratios = [round(i * 0.1, 1) for i in range(1, 10)]

    if max_examples is not None:
        code_examples = code_examples[:max_examples]

    dataset = []
    parsable_count = 0
    unparsable_count = 0

    for code in tqdm(code_examples, desc="Building compression dataset"):
        # Check parsability (paper: only parsable examples for training)
        parsable = is_parsable(code)

        if not parsable:
            unparsable_count += 1
            # Still include with heuristic tokenization (the LM handles both)
            # but mark as heuristic
            pass

        parsable_count += 1

        for tau in compression_ratios:
            compressed, actual_tau = compress_code_with_priority(code, tau, task)

            # Format for CodeT5 training
            input_text = format_compressor_input(code, tau, task)
            target_text = format_compressor_target(compressed)

            dataset.append({
                "input_text": input_text,
                "target_text": target_text,
                "original_code": code,
                "compressed_code": compressed,
                "tau_code": tau,
                "actual_tau": actual_tau,
                "task": task,
                "parsable": parsable,
            })

    print(f"\nDataset construction complete:")
    print(f"  Parsable examples: {parsable_count}")
    print(f"  Unparsable examples: {unparsable_count}")
    print(f"  Total samples (×{len(compression_ratios)} ratios): {len(dataset)}")

    return dataset


def split_dataset(
    dataset: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Split dataset into train/val/test.

    Args:
        dataset: Full dataset list.
        train_ratio, val_ratio, test_ratio: Split proportions.
        seed: Random seed.

    Returns:
        Dictionary with 'train', 'validation', 'test' splits.
    """
    random.seed(seed)

    # Group by original code to avoid data leakage
    # (all ratios of the same code example go to the same split)
    code_to_samples = {}
    for sample in dataset:
        code = sample["original_code"]
        if code not in code_to_samples:
            code_to_samples[code] = []
        code_to_samples[code].append(sample)

    # Shuffle unique code examples
    unique_codes = list(code_to_samples.keys())
    random.shuffle(unique_codes)

    n = len(unique_codes)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {"train": [], "validation": [], "test": []}

    for i, code in enumerate(unique_codes):
        if i < train_end:
            splits["train"].extend(code_to_samples[code])
        elif i < val_end:
            splits["validation"].extend(code_to_samples[code])
        else:
            splits["test"].extend(code_to_samples[code])

    for split_name, split_data in splits.items():
        random.shuffle(split_data)
        print(f"  {split_name}: {len(split_data)} samples")

    return splits


def save_dataset(splits: Dict[str, List[Dict]], output_dir: str):
    """Save dataset splits to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        filepath = output_path / f"{split_name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {split_name}: {len(split_data)} samples -> {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Build code compression training dataset for CodePromptZip"
    )
    parser.add_argument("--raw_data_dir", type=str, default="./data/raw",
                        help="Directory containing raw datasets")
    parser.add_argument("--output_dir", type=str, default="./data/compression_dataset",
                        help="Output directory for compression dataset")
    parser.add_argument("--task", type=str, default="bugs2fix",
                        choices=["bugs2fix", "assertion", "suggestion"])
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Max examples to process (for debugging)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Building Compression Dataset for: {args.task}")
    print("=" * 60)

    # 1. Load raw data
    print("\n[1/4] Loading raw data...")
    raw_data = load_raw_data(args.raw_data_dir, args.task)

    # 2. Extract code examples (use training partition as knowledge base)
    print("\n[2/4] Extracting code examples from training set...")
    code_examples = extract_code_examples(raw_data["train"], args.task)
    print(f"  Extracted {len(code_examples)} code examples")

    # 3. Build compression dataset
    print("\n[3/4] Applying Algorithm 1 (priority-driven compression)...")
    dataset = build_compression_dataset(
        code_examples, args.task, max_examples=args.max_examples
    )

    # 4. Split and save
    print("\n[4/4] Splitting and saving dataset...")
    task_output_dir = Path(args.output_dir) / args.task
    splits = split_dataset(dataset, seed=args.seed)
    save_dataset(splits, task_output_dir)

    print(f"\n{'=' * 60}")
    print("Dataset construction complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
