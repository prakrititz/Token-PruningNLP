"""
type_analysis.py - Java Token Type Categorization

Uses the `javalang` library to parse Java code and categorize tokens into
5 types defined in the paper (Appendix B):
    1. Symbol       - operators, delimiters (=, {, ;, ,)
    2. Signature    - method declarations & parameters
    3. Invocation   - function/method calls
    4. Identifier   - variable names, class names
    5. Structure    - control flow keywords (if, for, class, return)

For unparsable code, falls back to a heuristic-based approach.
"""

import re
from typing import List, Tuple, Dict, Optional
from collections import Counter

try:
    import javalang
    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False
    print("[Warning] javalang not installed. Using heuristic tokenizer.")


# Token type constants
SYMBOL = "Symbol"
SIGNATURE = "Signature"
INVOCATION = "Invocation"
IDENTIFIER = "Identifier"
STRUCTURE = "Structure"
UNKNOWN = "Unknown"

# Java keywords that are structural
STRUCTURE_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "break", "continue", "return", "try", "catch", "finally", "throw",
    "throws", "class", "interface", "extends", "implements", "new",
    "import", "package", "public", "private", "protected", "static",
    "final", "abstract", "synchronized", "volatile", "transient",
    "native", "strictfp", "assert", "enum", "instanceof", "super",
    "this", "void", "boolean", "byte", "char", "short", "int",
    "long", "float", "double",
}

# Symbols/operators/delimiters
SYMBOL_CHARS = set("={}()[];,.<>+-*/&|^~!?:%@#")


def tokenize_java_code(code: str) -> List[str]:
    """
    Simple tokenization of Java code preserving meaningful tokens.
    Splits on whitespace and separates symbols from identifiers.
    """
    tokens = []
    # Split by whitespace first
    for part in code.split():
        # Further split symbols from text
        current = ""
        for ch in part:
            if ch in SYMBOL_CHARS:
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(ch)
            else:
                current += ch
        if current:
            tokens.append(current)
    return tokens


def categorize_tokens_javalang(code: str) -> List[Tuple[str, str]]:
    """
    Use javalang to parse code and categorize each token.

    Returns:
        List of (token_value, token_type) tuples.
        Returns None if parsing fails.
    """
    if not JAVALANG_AVAILABLE:
        return None

    try:
        # Try to tokenize with javalang
        java_tokens = list(javalang.tokenizer.tokenize(code))
    except (javalang.tokenizer.LexerError, Exception):
        return None

    categorized = []
    for tok in java_tokens:
        value = tok.value
        tok_type_name = type(tok).__name__

        if tok_type_name in ("Separator", "Operator"):
            categorized.append((value, SYMBOL))
        elif tok_type_name == "Keyword":
            if value in STRUCTURE_KEYWORDS:
                categorized.append((value, STRUCTURE))
            else:
                categorized.append((value, STRUCTURE))
        elif tok_type_name == "Identifier":
            categorized.append((value, IDENTIFIER))
        elif tok_type_name in ("DecimalInteger", "HexInteger", "OctalInteger",
                                "BinaryInteger", "DecimalFloatingPoint",
                                "HexFloatingPoint", "String", "Character",
                                "Boolean", "Null"):
            # Literals are treated as identifiers
            categorized.append((value, IDENTIFIER))
        elif tok_type_name == "Modifier":
            categorized.append((value, SIGNATURE))
        elif tok_type_name == "BasicType":
            categorized.append((value, SIGNATURE))
        elif tok_type_name == "Annotation":
            categorized.append((value, IDENTIFIER))
        else:
            categorized.append((value, IDENTIFIER))

    return categorized


def _is_method_call(tokens: List[str], idx: int) -> bool:
    """Check if a token at idx is followed by '(' suggesting a method call."""
    if idx + 1 < len(tokens) and tokens[idx + 1] == "(":
        return True
    return False


def _is_preceded_by_dot(tokens: List[str], idx: int) -> bool:
    """Check if a token at idx is preceded by '.' suggesting member access."""
    if idx > 0 and tokens[idx - 1] == ".":
        return True
    return False


def categorize_tokens_heuristic(code: str) -> List[Tuple[str, str]]:
    """
    Heuristic-based token categorization for when javalang fails.
    Less accurate but handles incomplete/unparsable code.

    Returns:
        List of (token_value, token_type) tuples.
    """
    tokens = tokenize_java_code(code)
    categorized = []

    for i, token in enumerate(tokens):
        if not token.strip():
            continue

        # Check if it's a symbol
        if all(ch in SYMBOL_CHARS for ch in token):
            categorized.append((token, SYMBOL))
        # Check if it's a structure keyword
        elif token in STRUCTURE_KEYWORDS:
            categorized.append((token, STRUCTURE))
        # Check if it's a method invocation (identifier followed by '(')
        elif _is_method_call(tokens, i):
            categorized.append((token, INVOCATION))
        # Check if preceded by dot (member access / invocation chain)
        elif _is_preceded_by_dot(tokens, i) and _is_method_call(tokens, i):
            categorized.append((token, INVOCATION))
        else:
            categorized.append((token, IDENTIFIER))

    return categorized


def categorize_code_tokens(code: str) -> List[Tuple[str, str]]:
    """
    Main function: Categorize all tokens in a Java code snippet.
    Tries javalang first, falls back to heuristic approach.

    Args:
        code: Java source code string.

    Returns:
        List of (token_value, token_type) tuples.
    """
    # Try javalang parser first
    result = categorize_tokens_javalang(code)
    if result is not None:
        # Post-process: detect invocations (identifiers followed by '(')
        processed = []
        for i, (value, ttype) in enumerate(result):
            if ttype == IDENTIFIER:
                # Check if next token is '(' -> invocation
                if i + 1 < len(result) and result[i + 1][0] == "(":
                    processed.append((value, INVOCATION))
                    continue
            processed.append((value, ttype))
        return processed

    # Fallback to heuristic
    return categorize_tokens_heuristic(code)


def get_token_type_distribution(code: str) -> Dict[str, int]:
    """
    Get the distribution of token types in a code snippet.

    Args:
        code: Java source code.

    Returns:
        Dictionary mapping token_type -> count.
    """
    categorized = categorize_code_tokens(code)
    counter = Counter(ttype for _, ttype in categorized)
    return dict(counter)


def is_parsable(code: str) -> bool:
    """
    Check if a Java code snippet can be parsed by javalang.

    Args:
        code: Java source code.

    Returns:
        True if parsable by javalang tokenizer.
    """
    if not JAVALANG_AVAILABLE:
        return False
    try:
        list(javalang.tokenizer.tokenize(code))
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Test with sample Java code
    sample_code = """
    public static TYPE_1 init(java.lang.String name, java.util.Date date) {
        TYPE_1 VAR_1 = new TYPE_1();
        VAR_1.METHOD_1(name);
        java.util.Calendar VAR_2 = java.util.Calendar.getInstance();
        VAR_2.METHOD_2(date);
        VAR_1.METHOD_3(VAR_2);
        return VAR_1;
    }
    """

    print("=== Token Type Analysis ===\n")
    tokens = categorize_code_tokens(sample_code)
    for value, ttype in tokens:
        print(f"  {ttype:12s} | {value}")

    print("\n=== Distribution ===")
    dist = get_token_type_distribution(sample_code)
    for ttype, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {ttype:12s}: {count}")

    print(f"\nParsable: {is_parsable(sample_code)}")
