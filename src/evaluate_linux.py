"""
evaluate_linux.py - Linux-compatible Evaluation Pipeline for CodePromptZip

Identical to evaluate.py but patches the tokenizer import chain so that
compress.py (and any other module) uses tokenizer_utils_linux instead of
tokenizer_utils.  This avoids modifying the original files.

Usage (from project root):
    python src/evaluate_linux.py --config ./configs/config.yaml \
        --checkpoint ./checkpoints/best_model --tau_code 0.3
"""

import os
import sys

# Add project root to path so 'src' module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------------------------------------------------
# LINUX PATCH: Redirect src.tokenizer_utils -> src.tokenizer_utils_linux
# This MUST happen before importing compress.py or anything that
# depends on src.tokenizer_utils.
# ----------------------------------------------------------------
import src.tokenizer_utils_linux as _linux_tokenizer
sys.modules["src.tokenizer_utils"] = _linux_tokenizer
print("[Linux] Patched src.tokenizer_utils -> src.tokenizer_utils_linux")

# ----------------------------------------------------------------
# From here on, everything is identical to evaluate.py
# ----------------------------------------------------------------
import json
import time
import yaml
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional

import torch
from tqdm import tqdm

from src.retrieval import BM25Retriever, format_rag_prompt
from src.compress import CodeCompressor
from src.metrics.exact_match import exact_match_score
from src.metrics.codebleu_metric_linux import compute_codebleu


class BaseLMInference:
    """
    Wrapper for Base LM inference using llama-cpp-python.
    Uses CodeLlama-13B-Instruct (quantized GGUF) for local inference.
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Base LM configuration dict.
        """
        self.config = config
        self.model = None

        model_path = config.get("model_path", "")

        if not os.path.exists(model_path):
            print(f"\n[BaseLM] WARNING: Model file not found: {model_path}")
            print("[BaseLM] Please download CodeLlama-13B-Instruct GGUF:")
            print("  Download from: https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF")
            print("  Recommended file: codellama-13b-instruct.Q4_K_M.gguf")
            print(f"  Place it at: {model_path}")
            print("[BaseLM] Falling back to dummy mode for testing.\n")
            self.dummy_mode = True
            return

        self.dummy_mode = False

        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=model_path,
                n_ctx=config.get("n_ctx", 4096),
                n_gpu_layers=config.get("n_gpu_layers", 40),
                verbose=False,
            )
            print(f"[BaseLM] Loaded CodeLlama from {model_path}")
        except ImportError:
            print("[BaseLM] llama-cpp-python not installed. Using dummy mode.")
            self.dummy_mode = True

    def generate(self, prompt: str) -> str:
        """
        Generate output from the Base LM given a RAG prompt.

        Args:
            prompt: Full RAG prompt including demonstrations and query.

        Returns:
            Generated text.
        """
        if self.dummy_mode:
            return self._dummy_generate(prompt)

        output = self.model(
            prompt,
            max_tokens=self.config.get("max_tokens", 256),
            temperature=self.config.get("temperature", 0.0),
            stop=["\n[END]", "\n\n", "###"],
            echo=False,
        )

        return output["choices"][0]["text"].strip()

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate for multiple prompts."""
        results = []
        for prompt in tqdm(prompts, desc="Generating with Base LM"):
            results.append(self.generate(prompt))
        return results

    def _dummy_generate(self, prompt: str) -> str:
        """Dummy generation for testing when model is not available."""
        # Extract some context from the prompt to create a plausible dummy output
        if "FIXED_CODE" in prompt:
            # For Bugs2Fix, return a modified version of the buggy code
            lines = prompt.split("\n")
            for line in lines:
                if "BUGGY_CODE" in line:
                    continue
                if line.strip() and not line.startswith("#"):
                    return line.strip()
        return "// generated code placeholder"


class Evaluator:
    """
    Main evaluation pipeline that orchestrates retrieval, compression,
    generation, and metric computation.
    """

    def __init__(
        self,
        config: Dict,
        compressor: Optional[CodeCompressor] = None,
        base_lm: Optional[BaseLMInference] = None,
    ):
        self.config = config
        self.task = config["data"]["task"]
        self.compressor = compressor
        self.base_lm = base_lm or BaseLMInference(config["base_lm"])
        self.retriever = BM25Retriever()

    def run_evaluation(
        self,
        test_data: List[Dict],
        train_data: List[Dict],
        tau_code: float = 0.3,
        num_shots: int = 1,
        max_eval_samples: Optional[int] = None,
        use_compression: bool = True,
    ) -> Dict:
        """
        Run the full evaluation pipeline.

        Args:
            test_data: Test set samples.
            train_data: Training set (knowledge base for retrieval).
            tau_code: Compression ratio.
            num_shots: Number of retrieved examples.
            max_eval_samples: Max samples to evaluate (paper uses 2000).
            use_compression: Whether to use the compressor.

        Returns:
            Evaluation results dictionary.
        """
        print(f"\n{'='*60}")
        print(f"Evaluation: {self.task}")
        print(f"  tau_code: {tau_code}")
        print(f"  num_shots: {num_shots}")
        print(f"  compression: {'ON' if use_compression else 'OFF'}")
        print(f"{'='*60}")

        # Subsample test data
        if max_eval_samples and len(test_data) > max_eval_samples:
            random.seed(42)
            test_data = random.sample(test_data, max_eval_samples)
            print(f"\nSubsampled to {max_eval_samples} test examples")

        # Step 1: Build retrieval index
        print("\n[1/5] Building BM25 index...")
        self.retriever.build_index(train_data, self.task)

        # Step 2: Retrieve demonstrations for each test query
        print("\n[2/5] Retrieving demonstrations...")
        all_demos = self.retriever.retrieve_batch(
            test_data, self.task, top_k=num_shots
        )

        # Step 3: Compress demonstrations (optional)
        print("\n[3/5] Compressing demonstrations...")
        compressed_demos = []
        total_orig_tokens = 0
        total_comp_tokens = 0

        for demos in tqdm(all_demos, desc="Compressing demos"):
            compressed = []
            for demo in demos:
                if use_compression and self.compressor is not None:
                    comp_demo = self.compressor.compress_demonstration(
                        demo, tau_code, self.task
                    )
                    compressed.append(comp_demo)
                    # Track token counts
                    orig = " ".join(str(v) for v in demo.values())
                    comp = " ".join(str(v) for v in comp_demo.values())
                    total_orig_tokens += len(orig.split())
                    total_comp_tokens += len(comp.split())
                else:
                    compressed.append(demo)
            compressed_demos.append(compressed)

        # Step 4: Construct RAG prompts and generate
        print("\n[4/5] Generating with Base LM...")
        predictions = []
        references = []

        for query, demos in tqdm(
            zip(test_data, compressed_demos),
            total=len(test_data),
            desc="Generating",
        ):
            prompt = format_rag_prompt(query, demos, self.task)
            prediction = self.base_lm.generate(prompt)
            predictions.append(prediction)

            # Get reference
            ref = self._get_reference(query)
            references.append(ref)

        # Step 5: Compute metrics
        print("\n[5/5] Computing metrics...")
        results = self._compute_metrics(predictions, references)

        # Add token statistics
        if use_compression and total_orig_tokens > 0:
            actual_tau = 1 - (total_comp_tokens / total_orig_tokens)
            results["actual_tau"] = round(actual_tau * 100, 1)
            results["avg_orig_tokens"] = total_orig_tokens // len(test_data)
            results["avg_comp_tokens"] = total_comp_tokens // len(test_data)

        results["tau_code"] = tau_code
        results["num_shots"] = num_shots
        results["num_samples"] = len(test_data)
        results["use_compression"] = use_compression

        return results

    def _get_reference(self, query: Dict) -> str:
        """Extract the reference/ground-truth from a query."""
        if self.task == "bugs2fix":
            return query.get("fixed", "")
        elif self.task == "assertion":
            return query.get("assertion", "")
        elif self.task == "suggestion":
            return query.get("method_body", "")
        return ""

    def _compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict:
        """Compute task-specific metrics."""
        if self.task == "assertion":
            em = exact_match_score(predictions, references)
            return {"exact_match": round(em * 100, 1)}
        else:
            # Bugs2Fix and Code Suggestion use CodeBLEU
            codebleu_result = compute_codebleu(predictions, references)
            return codebleu_result

    def run_no_compression_baseline(
        self,
        test_data: List[Dict],
        train_data: List[Dict],
        num_shots: int = 1,
        max_eval_samples: Optional[int] = None,
    ) -> Dict:
        """Run evaluation without compression (w/o compression baseline)."""
        return self.run_evaluation(
            test_data, train_data,
            tau_code=0.0,
            num_shots=num_shots,
            max_eval_samples=max_eval_samples,
            use_compression=False,
        )

    def run_no_retrieval_baseline(
        self,
        test_data: List[Dict],
        max_eval_samples: Optional[int] = None,
    ) -> Dict:
        """Run evaluation without retrieval (w/o retrieval baseline)."""
        if max_eval_samples and len(test_data) > max_eval_samples:
            random.seed(42)
            test_data = random.sample(test_data, max_eval_samples)

        predictions = []
        references = []

        for query in tqdm(test_data, desc="No-retrieval generation"):
            # No demonstrations, just the query
            prompt = format_rag_prompt(query, [], self.task)
            prediction = self.base_lm.generate(prompt)
            predictions.append(prediction)
            references.append(self._get_reference(query))

        results = self._compute_metrics(predictions, references)
        results["num_samples"] = len(test_data)
        results["use_compression"] = False
        results["num_shots"] = 0
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CodePromptZip (Linux)")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained compressor checkpoint")
    parser.add_argument("--tau_code", type=float, default=0.3)
    parser.add_argument("--num_shots", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=100,
                        help="Max samples for evaluation (default: 100 for quick test)")
    parser.add_argument("--output_file", type=str, default="./results/eval_results.json")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task = config["data"]["task"]

    # Load test data
    raw_dir = Path(config["data"]["raw_dir"]) / task
    test_file = raw_dir / "test.json"
    train_file = raw_dir / "train.json"

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Loaded: {len(train_data)} train, {len(test_data)} test samples")

    # Load compressor (optional)
    compressor = None
    if args.checkpoint:
        compressor = CodeCompressor(args.checkpoint)

    # Initialize evaluator
    evaluator = Evaluator(config, compressor=compressor)

    # Run evaluation
    results = evaluator.run_evaluation(
        test_data=test_data,
        train_data=train_data,
        tau_code=args.tau_code,
        num_shots=args.num_shots,
        max_eval_samples=args.max_eval_samples,
    )

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    for key, value in results.items():
        print(f"  {key}: {value}")

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
