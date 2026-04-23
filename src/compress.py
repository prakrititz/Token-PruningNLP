"""
compress.py - Inference Pipeline for Code Compression

Uses the trained CopyCodeT5 compressor to compress code examples
for RAG-based downstream tasks.

Workflow:
1. Load trained compressor model
2. Accept original code + target compression ratio
3. Generate compressed code
4. Return compressed examples for RAG prompt construction
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.tokenizer_utils import (
    get_extended_tokenizer,
    format_compressor_input,
    decode_compressed_output,
    TASK_TOKEN_MAP,
)
from src.model.copy_codet5 import CopyCodeT5


class CodeCompressor:
    """
    Inference wrapper for the trained compressor model.

    Compresses code examples to a specified compression ratio
    using the trained CopyCodeT5 model.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        max_length: int = 512,
    ):
        """
        Args:
            checkpoint_dir: Path to saved model checkpoint.
            device: Device to run inference on.
            max_length: Maximum generation length.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        print(f"[Compressor] Loading from {checkpoint_dir}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        # Load model
        self.model = CopyCodeT5.from_pretrained(
            checkpoint_dir, tokenizer=self.tokenizer
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"[Compressor] Loaded on {self.device}")

    @torch.no_grad()
    def compress(
        self,
        code: str,
        tau_code: float = 0.3,
        task: str = "bugs2fix",
    ) -> str:
        """
        Compress a single code snippet.

        Args:
            code: Original code string.
            tau_code: Target compression ratio.
            task: Task name.

        Returns:
            Compressed code string.
        """
        # Format input
        input_text = format_compressor_input(code, tau_code, task)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.max_length,
        )

        # Decode
        output_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=False
        )

        # Extract compressed code
        compressed = decode_compressed_output(output_text)

        return compressed

    def compress_batch(
        self,
        codes: List[str],
        tau_code: float = 0.3,
        task: str = "bugs2fix",
    ) -> List[str]:
        """Compress multiple code snippets."""
        results = []
        for code in tqdm(codes, desc=f"Compressing (tau={tau_code})"):
            compressed = self.compress(code, tau_code, task)
            results.append(compressed)
        return results

    def compress_demonstration(
        self,
        demo: Dict,
        tau_code: float = 0.3,
        task: str = "bugs2fix",
    ) -> Dict:
        """
        Compress a RAG demonstration example.

        For Bugs2Fix: compresses the buggy+fixed code representation.
        The compressed demo can be used directly in the RAG prompt.

        Args:
            demo: Demonstration dictionary.
            tau_code: Target compression ratio.
            task: Task name.

        Returns:
            Modified demo dict with compressed code fields.
        """
        compressed_demo = demo.copy()

        if task == "bugs2fix":
            # Compress the full demonstration (buggy + fixed together)
            full_demo = f"### BUGGY_CODE\n{demo['buggy']}\n### FIXED_CODE\n{demo['fixed']}"
            compressed = self.compress(full_demo, tau_code, task)

            # Parse back into buggy/fixed parts
            if "### FIXED_CODE" in compressed:
                parts = compressed.split("### FIXED_CODE")
                compressed_demo["buggy"] = parts[0].replace("### BUGGY_CODE", "").strip()
                compressed_demo["fixed"] = parts[1].strip() if len(parts) > 1 else ""
            else:
                compressed_demo["buggy"] = compressed
                compressed_demo["fixed"] = ""

        elif task == "assertion":
            full_demo = (
                f"### FOCAL_METHOD\n{demo.get('focal_method', '')}\n"
                f"### UNIT_TEST\n{demo.get('test_method', '')}\n"
                f"### Assertion\n{demo.get('assertion', '')}"
            )
            compressed = self.compress(full_demo, tau_code, task)
            compressed_demo["compressed"] = compressed

        elif task == "suggestion":
            full_demo = (
                f"### METHOD_HEADER\n{demo.get('method_header', '')}\n"
                f"### WHOLE_METHOD\n{demo.get('method_body', '')}"
            )
            compressed = self.compress(full_demo, tau_code, task)
            compressed_demo["compressed"] = compressed

        compressed_demo["tau_code"] = tau_code
        return compressed_demo

    def get_compression_stats(
        self,
        original: str,
        compressed: str,
    ) -> Dict:
        """Compute compression statistics."""
        orig_tokens = len(original.split())
        comp_tokens = len(compressed.split())
        ratio = 1 - comp_tokens / orig_tokens if orig_tokens > 0 else 0

        return {
            "original_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "actual_ratio": round(ratio, 4),
        }


def main():
    parser = argparse.ArgumentParser(description="Compress code with CodePromptZip")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained compressor checkpoint")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON file with code examples")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file for compressed examples")
    parser.add_argument("--tau_code", type=float, default=0.3,
                        help="Target compression ratio")
    parser.add_argument("--task", type=str, default="bugs2fix")
    args = parser.parse_args()

    # Load compressor
    compressor = CodeCompressor(args.checkpoint)

    # Load input data
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nCompressing {len(data)} examples with tau_code={args.tau_code}")

    # Compress each example
    results = []
    for item in tqdm(data, desc="Compressing"):
        compressed_demo = compressor.compress_demonstration(
            item, args.tau_code, args.task
        )
        stats = compressor.get_compression_stats(
            str(item), str(compressed_demo)
        )
        compressed_demo.update(stats)
        results.append(compressed_demo)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nCompressed examples saved to {args.output_file}")

    # Print summary statistics
    avg_ratio = sum(r["actual_ratio"] for r in results) / len(results)
    avg_orig = sum(r["original_tokens"] for r in results) / len(results)
    avg_comp = sum(r["compressed_tokens"] for r in results) / len(results)
    print(f"Average compression ratio: {avg_ratio:.4f}")
    print(f"Average original tokens: {avg_orig:.1f}")
    print(f"Average compressed tokens: {avg_comp:.1f}")


if __name__ == "__main__":
    main()
