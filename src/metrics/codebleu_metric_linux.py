"""
codebleu_metric_linux.py - CodeBLEU Metric (Linux-compatible)

Implements CodeBLEU from scratch to bypass the broken tree-sitter
version deadlock in the 'codebleu' pip package.

CodeBLEU = α × BLEU + β × BLEU_weight + γ × syntax_match + δ × dataflow_match
Default weights: α = β = γ = δ = 0.25

Components:
  1. N-gram BLEU (NLTK corpus_bleu)
  2. Weighted N-gram BLEU (Java keyword weighting)
  3. Syntax match (AST subtree matching via tree-sitter-java)
  4. Dataflow match (simplified variable-flow matching)

Reference: Ren et al., 2020 - "CodeBLEU: a Method for Automatic Evaluation
           of Code Synthesis"
"""

import re
import math
from typing import List, Dict, Tuple
from collections import Counter

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# ============================================================
# Java Keywords for Weighted N-gram
# ============================================================
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private",
    "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while",
    # Common built-in types/methods treated as keywords
    "String", "System", "null", "true", "false",
}


# ============================================================
# Component 1: Standard N-gram BLEU
# ============================================================
def compute_ngram_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> float:
    """Standard corpus BLEU-4 score."""
    refs = [[ref.split()] for ref in references]
    preds = [pred.split() for pred in predictions]

    smoother = SmoothingFunction().method1
    try:
        score = corpus_bleu(
            refs, preds,
            weights=tuple([1.0 / max_n] * max_n),
            smoothing_function=smoother,
        )
    except (ZeroDivisionError, ValueError):
        score = 0.0

    return score


# ============================================================
# Component 2: Weighted N-gram BLEU (keyword-weighted)
# ============================================================
def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from a token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_weighted_ngram(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> float:
    """
    Weighted n-gram match: gives higher weight to n-grams
    containing language keywords.
    """
    total_score = 0.0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        if not pred_tokens or not ref_tokens:
            continue

        ngram_scores = []
        for n in range(1, max_n + 1):
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                ngram_scores.append(0.0)
                continue

            # Weighted matching: keyword n-grams count more
            weighted_match = 0.0
            weighted_total = 0.0

            for ngram, count in pred_ngrams.items():
                # Weight = 1 + (number of keywords in this n-gram)
                keyword_count = sum(1 for token in ngram if token in JAVA_KEYWORDS)
                weight = 1.0 + keyword_count

                weighted_total += count * weight
                matched = min(count, ref_ngrams.get(ngram, 0))
                weighted_match += matched * weight

            if weighted_total > 0:
                ngram_scores.append(weighted_match / weighted_total)
            else:
                ngram_scores.append(0.0)

        # Geometric mean of n-gram precisions
        if ngram_scores and all(s > 0 for s in ngram_scores):
            log_avg = sum(math.log(s) for s in ngram_scores) / len(ngram_scores)
            total_score += math.exp(log_avg)

    return total_score / len(predictions) if predictions else 0.0


# ============================================================
# Component 3: Syntax Match (AST subtree matching)
# ============================================================
def _parse_to_subtrees(code: str) -> List[str]:
    """
    Parse Java code into AST-like subtrees using tree-sitter-java.
    Falls back to a bracket-based heuristic if tree-sitter is unavailable.
    """
    try:
        import tree_sitter_java as tsjava
        from tree_sitter import Parser, Language

        # Try tree-sitter-java 0.23.x native API
        try:
            lang = Language(tsjava.language())
            parser = Parser(lang)
        except TypeError:
            try:
                lang = Language(tsjava.language(), "java")
                parser = Parser()
                parser.set_language(lang)
            except Exception:
                return _heuristic_subtrees(code)

        tree = parser.parse(bytes(code, "utf-8"))
        subtrees = []
        _extract_subtrees(tree.root_node, subtrees, max_depth=3)
        return subtrees

    except (ImportError, Exception):
        return _heuristic_subtrees(code)


def _extract_subtrees(node, subtrees: List[str], max_depth: int, current_depth: int = 0):
    """Recursively extract subtree signatures from a tree-sitter node."""
    if current_depth > max_depth:
        return

    # Build a subtree signature: node_type(child_type, child_type, ...)
    if node.child_count > 0:
        child_types = [child.type for child in node.children if child.type.strip()]
        signature = f"{node.type}({','.join(child_types)})"
        subtrees.append(signature)

        for child in node.children:
            _extract_subtrees(child, subtrees, max_depth, current_depth + 1)
    else:
        subtrees.append(node.type)


def _heuristic_subtrees(code: str) -> List[str]:
    """
    Fallback: extract structural patterns from Java code heuristically.
    Uses bracket matching and statement patterns as pseudo-AST subtrees.
    """
    subtrees = []
    tokens = code.split()

    # Extract statement-level patterns
    i = 0
    while i < len(tokens):
        # Method declarations
        if tokens[i] in ("public", "private", "protected", "static"):
            pattern = []
            while i < len(tokens) and tokens[i] != "{" and tokens[i] != ";":
                pattern.append(tokens[i])
                i += 1
            if pattern:
                subtrees.append("method_decl(" + ",".join(pattern[:5]) + ")")

        # Control flow
        elif tokens[i] in ("if", "for", "while", "switch", "try", "catch"):
            subtrees.append(f"control({tokens[i]})")
            i += 1

        # Return statements
        elif tokens[i] == "return":
            subtrees.append("return_stmt")
            i += 1

        # Variable declarations with types
        elif tokens[i] in ("int", "String", "boolean", "double", "float", "long", "char", "byte"):
            subtrees.append(f"var_decl({tokens[i]})")
            i += 1

        # Braces for block structure
        elif tokens[i] == "{":
            subtrees.append("block_start")
            i += 1
        elif tokens[i] == "}":
            subtrees.append("block_end")
            i += 1
        else:
            i += 1

    return subtrees


def compute_syntax_match(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Syntax match score: measures AST subtree overlap between
    predictions and references.
    """
    total_score = 0.0
    valid_count = 0

    for pred, ref in zip(predictions, references):
        pred_subtrees = Counter(_parse_to_subtrees(pred))
        ref_subtrees = Counter(_parse_to_subtrees(ref))

        if not ref_subtrees:
            continue

        # Count matching subtrees
        matched = sum(
            min(pred_subtrees.get(st, 0), count)
            for st, count in ref_subtrees.items()
        )
        total_ref = sum(ref_subtrees.values())

        if total_ref > 0:
            total_score += matched / total_ref
            valid_count += 1

    return total_score / valid_count if valid_count > 0 else 0.0


# ============================================================
# Component 4: Dataflow Match (simplified)
# ============================================================
def _extract_dataflow(code: str) -> List[Tuple[str, str, str]]:
    """
    Extract simplified data-flow edges: (variable, operation, target).
    Captures assignment, usage, and return flows.
    """
    flows = []
    tokens = code.split()

    for i, token in enumerate(tokens):
        # Assignment: var = expr
        if token == "=" and i > 0 and i < len(tokens) - 1:
            var = tokens[i - 1].rstrip(".")
            val = tokens[i + 1].rstrip(";").rstrip(")")
            if var.isidentifier():
                flows.append((var, "assign", val))

        # Return flow
        elif token == "return" and i < len(tokens) - 1:
            val = tokens[i + 1].rstrip(";")
            flows.append(("return", "value", val))

        # Method calls: obj.method(
        elif "." in token and "(" in token:
            parts = token.split(".")
            if len(parts) >= 2:
                obj = parts[0]
                method = parts[1].split("(")[0]
                flows.append((obj, "call", method))

    return flows


def compute_dataflow_match(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Dataflow match score: measures overlap of data-flow edges
    between predictions and references.
    """
    total_score = 0.0
    valid_count = 0

    for pred, ref in zip(predictions, references):
        pred_flows = Counter(str(f) for f in _extract_dataflow(pred))
        ref_flows = Counter(str(f) for f in _extract_dataflow(ref))

        if not ref_flows:
            continue

        matched = sum(
            min(pred_flows.get(f, 0), count)
            for f, count in ref_flows.items()
        )
        total_ref = sum(ref_flows.values())

        if total_ref > 0:
            total_score += matched / total_ref
            valid_count += 1

    return total_score / valid_count if valid_count > 0 else 0.0


# ============================================================
# Full CodeBLEU
# ============================================================
def compute_codebleu(
    predictions: List[str],
    references: List[str],
    lang: str = "java",
    weights: tuple = (0.25, 0.25, 0.25, 0.25),
) -> Dict[str, float]:
    """
    Compute CodeBLEU score with all four components.

    Args:
        predictions: List of predicted code strings.
        references: List of reference code strings.
        lang: Programming language (currently supports Java).
        weights: Weights for (ngram, weighted_ngram, syntax, dataflow).

    Returns:
        Dictionary with CodeBLEU and component scores.
    """
    alpha, beta, gamma, delta = weights

    print("  [CodeBLEU] Computing n-gram match...")
    ngram = compute_ngram_bleu(predictions, references)

    print("  [CodeBLEU] Computing weighted n-gram match...")
    weighted = compute_weighted_ngram(predictions, references)

    print("  [CodeBLEU] Computing syntax match...")
    syntax = compute_syntax_match(predictions, references)

    print("  [CodeBLEU] Computing dataflow match...")
    dataflow = compute_dataflow_match(predictions, references)

    codebleu = alpha * ngram + beta * weighted + gamma * syntax + delta * dataflow

    return {
        "codebleu": codebleu * 100,
        "ngram_match": ngram * 100,
        "weighted_ngram": weighted * 100,
        "syntax_match": syntax * 100,
        "dataflow_match": dataflow * 100,
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
    print("\nCodeBLEU Results:")
    for k, v in result.items():
        print(f"  {k}: {v:.2f}%")
