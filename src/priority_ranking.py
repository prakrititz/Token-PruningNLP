"""
priority_ranking.py - Type-aware Priority-Driven Token Removal

Implements Algorithm 1 from the CodePromptZip paper:
    Priority-driven Greedy Algorithm for Dataset Construction

Key idea: Tokens of higher-priority types are removed before lower-priority types.
Within the same type, higher term-frequency tokens are removed first (redundant tokens).
"""

import heapq
from typing import List, Tuple, Dict, Optional
from collections import Counter

from src.type_analysis import categorize_code_tokens, tokenize_java_code


# Default priority rankings from the paper's ablation study (Figure 1).
# Index 0 = highest removal priority (removed first).
DEFAULT_PRIORITIES = {
    "bugs2fix": ["Identifier", "Invocation", "Structure", "Symbol", "Signature"],
    "assertion": ["Invocation", "Symbol", "Identifier", "Structure", "Signature"],
    "suggestion": ["Invocation", "Structure", "Symbol", "Identifier", "Signature"],
}


class TokenWithPriority:
    """Token wrapper for priority queue ordering."""

    def __init__(self, value: str, token_type: str, position: int,
                 type_priority: int, term_frequency: int):
        self.value = value
        self.token_type = token_type
        self.position = position  # Original position in sequence
        self.type_priority = type_priority  # Lower number = removed first
        self.term_frequency = term_frequency  # Higher freq = removed first

    def __lt__(self, other):
        """
        Comparison for min-heap (highest removal priority = smallest value).
        Priority order:
            1. Lower type_priority value = higher removal priority
            2. Higher term_frequency = higher removal priority (tie-breaker)
            3. Higher position = higher removal priority (tie-breaker)
        """
        if self.type_priority != other.type_priority:
            return self.type_priority < other.type_priority
        if self.term_frequency != other.term_frequency:
            return self.term_frequency > other.term_frequency
        return self.position > other.position

    def __repr__(self):
        return (f"Token('{self.value}', type={self.token_type}, "
                f"pos={self.position}, pri={self.type_priority}, "
                f"tf={self.term_frequency})")


def compute_term_frequencies(tokens: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Compute term frequency for each token value within the code example.

    Args:
        tokens: List of (token_value, token_type) tuples.

    Returns:
        Dictionary mapping token_value -> frequency count.
    """
    return Counter(value for value, _ in tokens)


def compress_code_with_priority(
    code: str,
    tau_code: float,
    task: str = "bugs2fix",
    priority_order: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """
    Algorithm 1: Priority-driven Greedy Token Removal.

    Compresses code by iteratively removing tokens based on type-aware priorities.
    Higher-priority types are removed first. Within a type, higher-frequency tokens
    are removed first (as they are more likely redundant).

    Args:
        code: Original Java code string.
        tau_code: Target compression ratio (fraction of tokens to remove, 0.0-1.0).
        task: Task name to determine priority ranking.
        priority_order: Optional custom priority order (index 0 = removed first).

    Returns:
        Tuple of (compressed_code, actual_compression_ratio).
    """
    if priority_order is None:
        priority_order = DEFAULT_PRIORITIES.get(task, DEFAULT_PRIORITIES["bugs2fix"])

    # Step 1: Categorize tokens by type
    categorized_tokens = categorize_code_tokens(code)

    if len(categorized_tokens) == 0:
        return code, 0.0

    L = len(categorized_tokens)
    L_rm = int(tau_code * L)  # Number of tokens to remove

    if L_rm == 0:
        # Reconstruct from categorized tokens
        return " ".join(value for value, _ in categorized_tokens), 0.0

    # Step 2: Compute term frequencies
    tf = compute_term_frequencies(categorized_tokens)

    # Step 3: Build priority queue
    # Map type name -> priority value (lower = removed first)
    type_to_priority = {}
    for idx, ttype in enumerate(priority_order):
        type_to_priority[ttype] = idx

    priority_queue = []
    for pos, (value, token_type) in enumerate(categorized_tokens):
        # Tokens not in the priority order get lowest removal priority
        type_pri = type_to_priority.get(token_type, len(priority_order))
        token_wp = TokenWithPriority(
            value=value,
            token_type=token_type,
            position=pos,
            type_priority=type_pri,
            term_frequency=tf[value],
        )
        heapq.heappush(priority_queue, token_wp)

    # Step 4: Iteratively remove highest-priority tokens
    removed_positions = set()
    removed_count = 0

    while removed_count < L_rm and priority_queue:
        token = heapq.heappop(priority_queue)
        removed_positions.add(token.position)
        removed_count += 1

    # Step 5: Construct compressed code (keep tokens not in removed set)
    compressed_tokens = []
    for pos, (value, _) in enumerate(categorized_tokens):
        if pos not in removed_positions:
            compressed_tokens.append(value)

    compressed_code = " ".join(compressed_tokens)
    actual_ratio = removed_count / L if L > 0 else 0.0

    return compressed_code, actual_ratio


def generate_compression_samples(
    code: str,
    task: str = "bugs2fix",
    ratios: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Generate multiple compressed versions of a code snippet at different ratios.
    Used for constructing the compression training dataset.

    Args:
        code: Original Java code string.
        task: Task name.
        ratios: List of compression ratios to generate. Defaults to 0.1-0.9.

    Returns:
        List of dicts with keys: original, compressed, tau_code, actual_tau, task.
    """
    if ratios is None:
        ratios = [round(i * 0.1, 1) for i in range(1, 10)]

    samples = []
    for tau in ratios:
        compressed, actual_tau = compress_code_with_priority(code, tau, task)
        samples.append({
            "original": code,
            "compressed": compressed,
            "tau_code": tau,
            "actual_tau": actual_tau,
            "task": task,
        })

    return samples


if __name__ == "__main__":
    # Test with sample Bugs2Fix code
    sample_code = (
        "public static TYPE_1 init ( java.lang.String name , java.util.Date date ) { "
        "TYPE_1 VAR_1 = new TYPE_1 ( ) ; VAR_1 . METHOD_1 ( name ) ; "
        "java.util.Calendar VAR_2 = java.util.Calendar . getInstance ( ) ; "
        "VAR_2 . METHOD_2 ( date ) ; VAR_1 . METHOD_3 ( VAR_2 ) ; return VAR_1 ; }"
    )

    print("=== Priority-Driven Compression ===\n")
    print(f"Original ({len(tokenize_java_code(sample_code))} tokens):")
    print(f"  {sample_code[:100]}...\n")

    for tau in [0.1, 0.3, 0.5, 0.7]:
        compressed, actual = compress_code_with_priority(sample_code, tau, "bugs2fix")
        orig_len = len(tokenize_java_code(sample_code))
        comp_len = len(compressed.split())
        print(f"tau_code={tau:.1f} (actual={actual:.2f}): {comp_len} tokens")
        print(f"  {compressed[:100]}...\n")
