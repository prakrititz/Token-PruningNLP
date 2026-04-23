import os
from dotenv import load_dotenv
load_dotenv()
"""
tokenizer_utils.py - Extended Tokenizer for CodePromptZip

Extends the CodeT5 tokenizer with special tokens for:
- Task indication: <ASSERTION>, <BUGS2FIX>, <SUGGESTION>
- Compression ratio control: <Ratio>, </Ratio>
- Compression markers: <Compress>, </Compress>
"""

from transformers import AutoTokenizer, RobertaTokenizer
from typing import List, Dict, Optional


# All special tokens added by CodePromptZip
SPECIAL_TOKENS = {
    "task_tokens": ["<ASSERTION>", "<BUGS2FIX>", "<SUGGESTION>"],
    "ratio_tokens": ["<Ratio>", "</Ratio>"],
    "compress_tokens": ["<Compress>", "</Compress>"],
}

# Map task name -> special token
TASK_TOKEN_MAP = {
    "assertion": "<ASSERTION>",
    "bugs2fix": "<BUGS2FIX>",
    "suggestion": "<SUGGESTION>",
}


def get_extended_tokenizer(model_name: str = "Salesforce/codet5-base"):
    """
    Load the CodeT5 tokenizer and extend it with CodePromptZip special tokens.

    Args:
        model_name: HuggingFace model name for CodeT5.

    Returns:
        Extended tokenizer with additional special tokens.
    """
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Collect all new special tokens
    new_tokens = []
    for token_group in SPECIAL_TOKENS.values():
        new_tokens.extend(token_group)

    # Get existing additional special tokens (safely handle None)
    existing = tokenizer.additional_special_tokens
    if existing is None:
        existing = []

    # Only add tokens that aren't already present (converting existing AddedTokens to string if needed)
    existing_strs = [str(t) for t in existing]
    tokens_to_add = [t for t in new_tokens if t not in existing_strs]

    if tokens_to_add:
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": tokens_to_add
        })
        print(f"[Tokenizer] Added {num_added} special tokens to {model_name}")
    else:
        print(f"[Tokenizer] All special tokens already present in {model_name}")

    print(f"[Tokenizer] Vocab size: {len(tokenizer)}")

    return tokenizer


def format_compressor_input(
    code: str,
    tau_code: float,
    task: str,
    tokenizer: Optional[RobertaTokenizer] = None,
) -> str:
    """
    Format the input to the compressor model.

    Format: <TASK> <Ratio> {tau_code} </Ratio> <Compress> {code} </Compress>

    Args:
        code: Original code snippet to compress.
        tau_code: Target compression ratio (0.0 to 1.0).
        task: Task name (assertion, bugs2fix, suggestion).
        tokenizer: Optional tokenizer (unused, kept for API consistency).

    Returns:
        Formatted input string for the compressor.
    """
    task_token = TASK_TOKEN_MAP.get(task, "<BUGS2FIX>")
    return f"{task_token} <Ratio> {tau_code:.1f} </Ratio> <Compress> {code} </Compress>"


def format_compressor_target(compressed_code: str) -> str:
    """
    Format the target output for compressor training.

    Format: <Compress> {compressed_code} </Compress>

    Args:
        compressed_code: The compressed version of the code.

    Returns:
        Formatted target string.
    """
    return f"<Compress> {compressed_code} </Compress>"


def decode_compressed_output(output: str) -> str:
    """
    Extract the compressed code from the compressor's output.

    Args:
        output: Raw model output string.

    Returns:
        Extracted compressed code.
    """
    # Remove <Compress> and </Compress> markers
    output = output.strip()
    if "<Compress>" in output:
        output = output.split("<Compress>", 1)[1]
    if "</Compress>" in output:
        output = output.split("</Compress>", 1)[0]
    return output.strip()


if __name__ == "__main__":
    # Quick test
    tokenizer = get_extended_tokenizer("Salesforce/codet5-large")

    sample_code = "public static void main(String[] args) { System.out.println(\"Hello\"); }"
    formatted = format_compressor_input(sample_code, 0.3, "bugs2fix")
    print(f"\nFormatted input:\n{formatted}")

    target = format_compressor_target("public static void main(String[] args) { ; }")
    print(f"\nFormatted target:\n{target}")

    # Tokenize and verify
    tokens = tokenizer(formatted, return_tensors="pt")
    print(f"\nInput token IDs shape: {tokens['input_ids'].shape}")
    decoded = tokenizer.decode(tokens['input_ids'][0])
    print(f"Decoded back: {decoded}")
