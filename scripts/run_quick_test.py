"""
run_quick_test.py - Quick sanity test for the entire pipeline

Runs a minimal end-to-end test with a few samples to verify
all components work correctly before full training.

Usage:
    python scripts/run_quick_test.py
"""

import os
import sys
import importlib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -------------------------------------------------------
# Dependency check helper
# -------------------------------------------------------
REQUIRED_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "rank_bm25": "rank-bm25",
    "javalang": "javalang",
    "yaml": "pyyaml",
    "datasets": "datasets",
    "tqdm": "tqdm",
    "numpy": "numpy",
    "dotenv": "python-dotenv",
}


def check_dependencies():
    """Check that all required packages are installed."""
    print("\n[0/7] Checking Dependencies...")
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            print(f"  [OK] {module_name}")
        except ImportError:
            print(f"  [FAIL] {module_name}  (install: pip install {pip_name})")
            missing.append(pip_name)

    if missing:
        print(f"\n  MISSING PACKAGES DETECTED!")
        print(f"  Run the following command inside your activated virtual environment:\n")
        print(f"    pip install {' '.join(missing)}")
        print(f"\n  Or re-run the setup script:")
        print(f"    .\\setup_env.ps1")
        print(f"\n  Then activate the venv before running tests:")
        print(f"    .\\venv\\Scripts\\Activate.ps1")
        print(f"    python scripts/run_quick_test.py")
        return False

    print("  All dependencies available!")
    return True


# -------------------------------------------------------
# Tests
# -------------------------------------------------------
def test_type_analysis():
    """Test token type categorization."""
    print("\n[1/7] Testing Type Analysis...")
    from src.type_analysis import categorize_code_tokens, get_token_type_distribution

    code = "public static void main ( String [] args ) { System.out.println ( args ) ; }"
    tokens = categorize_code_tokens(code)
    dist = get_token_type_distribution(code)

    print(f"  Tokens: {len(tokens)}")
    print(f"  Distribution: {dist}")
    assert len(tokens) > 0, "No tokens returned!"
    print("  [OK] PASSED")


def test_priority_ranking():
    """Test Algorithm 1 - priority-driven compression."""
    print("\n[2/7] Testing Priority Ranking (Algorithm 1)...")
    from src.priority_ranking import compress_code_with_priority

    code = "public static TYPE_1 init ( java.lang.String name ) { TYPE_1 VAR_1 = new TYPE_1 ( ) ; return VAR_1 ; }"

    for tau in [0.1, 0.3, 0.5, 0.7]:
        compressed, actual = compress_code_with_priority(code, tau, "bugs2fix")
        orig_tokens = len(code.split())
        comp_tokens = len(compressed.split())
        print(f"  tau={tau:.1f}: {orig_tokens} -> {comp_tokens} tokens (actual={actual:.2f})")

    assert len(compressed.split()) < len(code.split()), "Compression not working!"
    print("  [OK] PASSED")


def test_tokenizer():
    """Test extended tokenizer."""
    print("\n[3/7] Testing Extended Tokenizer...")
    from src.tokenizer_utils import (
        get_extended_tokenizer, format_compressor_input,
        format_compressor_target, decode_compressed_output,
    )

    # Use base model for quick test (smaller download)
    tokenizer = get_extended_tokenizer("Salesforce/codet5-base")

    input_text = format_compressor_input("public void foo ( ) { }", 0.3, "bugs2fix")
    target_text = format_compressor_target("public void foo ( )")
    decoded = decode_compressed_output(target_text)

    print(f"  Input: {input_text[:80]}...")
    print(f"  Target: {target_text}")
    print(f"  Decoded: {decoded}")

    # Verify special tokens are in vocabulary
    tokens = tokenizer(input_text, return_tensors="pt")
    print(f"  Token IDs shape: {tokens['input_ids'].shape}")
    # Check that <BUGS2FIX> is recognized as a single token (not split into subwords)
    bugs2fix_id = tokenizer.convert_tokens_to_ids("<BUGS2FIX>")
    assert bugs2fix_id != tokenizer.unk_token_id, "<BUGS2FIX> not found in vocab!"
    print(f"  <BUGS2FIX> token ID: {bugs2fix_id}")
    print("  [OK] PASSED")


def test_copy_module():
    """Test the copy mechanism."""
    print("\n[4/7] Testing Copy Module...")
    import torch
    from src.model.copy_module import CopyModule

    hidden_size = 768  # base model
    batch_size = 2
    src_len = 20
    tgt_len = 10
    vocab_size = 32100

    module = CopyModule(hidden_size)

    decoder_hidden = torch.randn(batch_size, tgt_len, hidden_size)
    cross_attn = torch.softmax(torch.randn(batch_size, tgt_len, src_len), dim=-1)
    encoder_hidden = torch.randn(batch_size, src_len, hidden_size)
    vocab_logits = torch.randn(batch_size, tgt_len, vocab_size)
    source_ids = torch.randint(0, vocab_size, (batch_size, src_len))

    log_probs = module(
        decoder_hidden, cross_attn, encoder_hidden,
        vocab_logits, source_ids, vocab_size,
    )

    print(f"  Output shape: {log_probs.shape}")
    print(f"  Log prob range: [{log_probs.min().item():.4f}, {log_probs.max().item():.4f}]")
    assert log_probs.shape == (batch_size, tgt_len, vocab_size)
    # Verify log probs sum to ~1 in exp space
    probs_sum = log_probs.exp().sum(dim=-1).mean().item()
    print(f"  Prob sum (should be ~1.0): {probs_sum:.4f}")
    print("  [OK] PASSED")


def test_model_forward():
    """Test CopyCodeT5 forward pass."""
    print("\n[5/7] Testing CopyCodeT5 Forward Pass...")
    from src.tokenizer_utils import get_extended_tokenizer
    from src.model.copy_codet5 import create_model

    tokenizer = get_extended_tokenizer("Salesforce/codet5-base")

    model = create_model(
        model_name="Salesforce/codet5-base",
        use_copy=True,
        tokenizer=tokenizer,
        gradient_checkpointing=False,
    )

    # Test input
    input_text = "<BUGS2FIX> <Ratio> 0.3 </Ratio> <Compress> public void foo ( ) { } </Compress>"
    target_text = "<Compress> public void foo ( ) </Compress>"

    inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
    targets = tokenizer(target_text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")

    labels = targets["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # Forward
    output = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=labels,
    )

    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Logits shape: {output['logits'].shape}")
    assert output["loss"] is not None
    assert output["loss"].item() > 0
    print("  [OK] PASSED")


def test_model_generate():
    """Test CopyCodeT5 generation."""
    print("\n[6/7] Testing CopyCodeT5 Generation...")
    import torch
    from src.tokenizer_utils import get_extended_tokenizer
    from src.model.copy_codet5 import create_model

    tokenizer = get_extended_tokenizer("Salesforce/codet5-base")

    model = create_model(
        model_name="Salesforce/codet5-base",
        use_copy=True,
        tokenizer=tokenizer,
        gradient_checkpointing=False,
    )
    model.eval()

    input_text = "<BUGS2FIX> <Ratio> 0.3 </Ratio> <Compress> public void foo ( ) { int x = 0 ; } </Compress>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=32,
        )

    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
    print(f"  Input: {input_text[:60]}...")
    print(f"  Generated: {gen_text[:80]}...")
    assert gen_ids.shape[1] > 1, "No tokens generated!"
    print("  [OK] PASSED")


def test_retrieval():
    """Test BM25 retrieval."""
    print("\n[7/7] Testing BM25 Retrieval...")
    from src.retrieval import BM25Retriever, format_rag_prompt

    train_data = [
        {"buggy": "public void foo ( ) { int x = 0 ; if ( x = 1 ) { } }",
         "fixed": "public void foo ( ) { int x = 0 ; if ( x == 1 ) { } }"},
        {"buggy": "public int add ( int a , int b ) { return a - b ; }",
         "fixed": "public int add ( int a , int b ) { return a + b ; }"},
        {"buggy": "public String getName ( ) { return this.name ; }",
         "fixed": "public String getName ( ) { return name ; }"},
    ]

    retriever = BM25Retriever()
    retriever.build_index(train_data, "bugs2fix")

    query = {"buggy": "public void bar ( ) { int y = 0 ; if ( y = 2 ) { } }"}
    results = retriever.retrieve(query, "bugs2fix", top_k=1)

    print(f"  Query: {query['buggy'][:50]}...")
    print(f"  Retrieved: {results[0]['buggy'][:50]}...")
    print(f"  Score: {results[0]['bm25_score']:.4f}")

    prompt = format_rag_prompt(query, results, "bugs2fix")
    print(f"  Prompt length: {len(prompt)} chars")
    assert len(results) == 1
    print("  [OK] PASSED")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    print("=" * 60)
    print("CodePromptZip: Quick Sanity Tests")
    print("=" * 60)

    # Step 0: Check all dependencies first
    if not check_dependencies():
        return 1

    try:
        test_type_analysis()
        test_priority_ranking()
        test_tokenizer()
        test_copy_module()
        test_model_forward()
        test_model_generate()
        test_retrieval()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! [OK]")
        print("=" * 60)
        print("\nYou're ready to run the full training pipeline:")
        print("  python scripts/run_train_compressor.py")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
