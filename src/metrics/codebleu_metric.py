"""
codebleu_metric.py - CodeBLEU Metric Wrapper

CodeBLEU is a composite metric for code generation that considers:
- N-gram match (BLEU)
- Weighted n-gram match (keywords)
- Syntax match (AST)
- Semantic match (data flow)

Used for Bugs2Fix and Code Suggestion evaluation.
Reference: Ren et al., 2020
"""

from typing import List, Optional, Dict

try:
    from codebleu import calc_codebleu
    CODEBLEU_AVAILABLE = True
except ImportError:
    CODEBLEU_AVAILABLE = False
    print("[Warning] codebleu package not installed. Install with: pip install codebleu")


def compute_codebleu(
    predictions: List[str],
    references: List[str],
    lang: str = "java",
    weights: tuple = (0.25, 0.25, 0.25, 0.25),
) -> Dict[str, float]:
    """
    Compute CodeBLEU score.

    Args:
        predictions: List of predicted code strings.
        references: List of reference code strings.
        lang: Programming language.
        weights: Weights for (ngram, weighted_ngram, syntax, dataflow).

    Returns:
        Dictionary with CodeBLEU and component scores.
    """
    if not CODEBLEU_AVAILABLE:
        # Fallback: compute simple BLEU-like score
        print("[Warning] Using fallback BLEU (codebleu package not available)")
        return _fallback_bleu(predictions, references)

    try:
        result = calc_codebleu(
            references=[[ref] for ref in references],
            predictions=predictions,
            lang=lang,
            weights=weights,
        )
        return {
            "codebleu": result["codebleu"] * 100,  # Percentage
            "ngram_match": result.get("ngram_match_score", 0) * 100,
            "weighted_ngram": result.get("weighted_ngram_match_score", 0) * 100,
            "syntax_match": result.get("syntax_match_score", 0) * 100,
            "dataflow_match": result.get("dataflow_match_score", 0) * 100,
        }
    except Exception as e:
        print(f"[Warning] CodeBLEU computation failed: {e}")
        return _fallback_bleu(predictions, references)


def _fallback_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Simple BLEU-4 fallback when codebleu package is unavailable."""
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    refs = [[ref.split()] for ref in references]
    preds = [pred.split() for pred in predictions]

    smoother = SmoothingFunction().method1
    score = corpus_bleu(refs, preds, smoothing_function=smoother)

    return {
        "codebleu": score * 100,
        "ngram_match": score * 100,
        "weighted_ngram": 0.0,
        "syntax_match": 0.0,
        "dataflow_match": 0.0,
    }


if __name__ == "__main__":
    # Test
    preds = [
        "public int add ( int a , int b ) { return a + b ; }",
        "public void print ( String s ) { System.out.println ( s ) ; }",
    ]
    refs = [
        "public int add ( int a , int b ) { return a + b ; }",
        "public void print ( String msg ) { System.out.println ( msg ) ; }",
    ]

    result = compute_codebleu(preds, refs)
    print("CodeBLEU Results:")
    for k, v in result.items():
        print(f"  {k}: {v:.2f}%")
