"""
exact_match.py - Exact Match Metric for Assertion Generation

The Exact Match rate measures the percentage of generated assertions
that exactly match the reference assertion (after normalization).
"""

from typing import List


def normalize_code(code: str) -> str:
    """Normalize code for comparison (strip whitespace, lowercase)."""
    return " ".join(code.strip().split()).lower()


def exact_match_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute Exact Match rate.

    Args:
        predictions: List of predicted assertion strings.
        references: List of reference assertion strings.

    Returns:
        Exact Match rate (0.0 to 1.0).
    """
    assert len(predictions) == len(references), \
        f"Length mismatch: {len(predictions)} vs {len(references)}"

    if len(predictions) == 0:
        return 0.0

    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if normalize_code(pred) == normalize_code(ref)
    )

    return matches / len(predictions)


def exact_match_per_sample(
    predictions: List[str],
    references: List[str],
) -> List[bool]:
    """
    Compute per-sample exact match.

    Returns:
        List of booleans indicating match for each sample.
    """
    return [
        normalize_code(pred) == normalize_code(ref)
        for pred, ref in zip(predictions, references)
    ]
