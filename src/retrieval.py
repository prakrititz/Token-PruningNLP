"""
retrieval.py - BM25 Retrieval for RAG

Implements BM25-based retrieval to select relevant code examples
from the knowledge base (training set) for each query.

The paper uses BM25 (He et al., 2024) as the retriever.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from rank_bm25 import BM25Okapi
from tqdm import tqdm


class BM25Retriever:
    """
    BM25-based retriever for RAG code examples.

    Indexes the training set and retrieves the most similar
    code examples for each test query.
    """

    def __init__(self):
        self.bm25 = None
        self.corpus = []       # Raw corpus items (dicts)
        self.tokenized = []    # Tokenized corpus for BM25

    def build_index(self, data: List[Dict], task: str = "bugs2fix"):
        """
        Build BM25 index from training data.

        Args:
            data: List of training samples.
            task: Task name (determines which field to index on).
        """
        print(f"[BM25] Building index on {len(data)} documents...")

        self.corpus = data
        self.tokenized = []

        for item in tqdm(data, desc="Tokenizing"):
            text = self._get_query_text(item, task)
            tokens = self._tokenize(text)
            self.tokenized.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized)
        print(f"[BM25] Index built with {len(self.corpus)} documents")

    def retrieve(
        self,
        query: Dict,
        task: str = "bugs2fix",
        top_k: int = 1,
    ) -> List[Dict]:
        """
        Retrieve top-k most similar examples for a query.

        Args:
            query: Query sample dict.
            task: Task name.
            top_k: Number of examples to retrieve.

        Returns:
            List of top-k most similar training examples.
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index first.")

        query_text = self._get_query_text(query, task)
        query_tokens = self._tokenize(query_text)

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices (excluding exact matches)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            item = self.corpus[idx].copy()
            item["bm25_score"] = float(scores[idx])
            results.append(item)

        return results

    def retrieve_batch(
        self,
        queries: List[Dict],
        task: str = "bugs2fix",
        top_k: int = 1,
    ) -> List[List[Dict]]:
        """Retrieve for multiple queries."""
        results = []
        for query in tqdm(queries, desc="Retrieving"):
            results.append(self.retrieve(query, task, top_k))
        return results

    def _get_query_text(self, item: Dict, task: str) -> str:
        """
        Extract the query-relevant text from a data item.

        For retrieval, we use the input part (not the target).
        """
        if task == "bugs2fix":
            return item.get("buggy", "")
        elif task == "assertion":
            return f"{item.get('focal_method', '')} {item.get('test_method', '')}"
        elif task == "suggestion":
            return item.get("method_header", "")
        else:
            return str(item)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization for BM25."""
        return text.lower().split()

    def save(self, filepath: str):
        """Save the BM25 index to disk."""
        save_data = {
            "corpus": self.corpus,
            "tokenized": self.tokenized,
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        print(f"[BM25] Index saved to {filepath}")

    def load(self, filepath: str):
        """Load a saved BM25 index."""
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        self.corpus = save_data["corpus"]
        self.tokenized = save_data["tokenized"]
        self.bm25 = BM25Okapi(self.tokenized)
        print(f"[BM25] Index loaded from {filepath} ({len(self.corpus)} documents)")


def format_rag_prompt(
    query: Dict,
    demonstrations: List[Dict],
    task: str = "bugs2fix",
) -> str:
    """
    Format the full RAG prompt with demonstrations and query.
    Follows the prompt templates from Figure 7 of the paper.

    Args:
        query: The test query dict.
        demonstrations: List of retrieved (possibly compressed) demonstration dicts.
        task: Task name.

    Returns:
        Formatted prompt string.
    """
    if task == "bugs2fix":
        return _format_bugs2fix_prompt(query, demonstrations)
    elif task == "assertion":
        return _format_assertion_prompt(query, demonstrations)
    elif task == "suggestion":
        return _format_suggestion_prompt(query, demonstrations)
    else:
        raise ValueError(f"Unknown task: {task}")


def _format_bugs2fix_prompt(query: Dict, demos: List[Dict]) -> str:
    """Format Bugs2Fix RAG prompt (Figure 7b)."""
    parts = ["Demonstrations:"]

    for demo in demos:
        buggy = demo.get("buggy", demo.get("compressed_buggy", ""))
        fixed = demo.get("fixed", demo.get("compressed_fixed", ""))
        parts.append("[START]")
        parts.append(f"### BUGGY_CODE:\n{buggy}")
        parts.append(f"\n### FIXED_CODE:\n{fixed}")

    parts.append("\n...\n[END]")
    parts.append("Query:")
    parts.append("[START]")
    parts.append(f"### BUGGY_CODE:\n{query['buggy']}")
    parts.append("\n### FIXED_CODE:")

    return "\n".join(parts)


def _format_assertion_prompt(query: Dict, demos: List[Dict]) -> str:
    """Format Assertion Generation RAG prompt (Figure 7a)."""
    parts = ["Demonstrations:"]

    for demo in demos:
        parts.append("[START]")
        parts.append(f"### FOCAL_METHOD:\n{demo.get('focal_method', '')}")
        parts.append(f"### UNIT_TEST:\n{demo.get('test_method', '')}")
        parts.append(f"### Assertion:\n{demo.get('assertion', '')}")

    parts.append("\n...\n[END]")
    parts.append("Query:")
    parts.append("[START]")
    parts.append(f"### FOCAL_METHOD:\n{query.get('focal_method', '')}")
    parts.append(f"### UNIT_TEST:\n{query.get('test_method', '')}")
    parts.append("### Assertion:")

    return "\n".join(parts)


def _format_suggestion_prompt(query: Dict, demos: List[Dict]) -> str:
    """Format Code Suggestion RAG prompt (Figure 7c)."""
    parts = ["Demonstrations:"]

    for demo in demos:
        parts.append("[START]")
        parts.append(f"### METHOD_HEADER:\n{demo.get('method_header', '')}")
        parts.append(f"### WHOLE_METHOD:\n{demo.get('method_body', '')}")

    parts.append("\n...\n[END]")
    parts.append("Query:")
    parts.append("[START]")
    parts.append(f"### METHOD_HEADER:\n{query.get('method_header', '')}")
    parts.append("### WHOLE_METHOD:")

    return "\n".join(parts)


if __name__ == "__main__":
    # Test with sample data
    train_data = [
        {"buggy": "public void foo ( ) { int x = 0 ; if ( x = 1 ) { } }",
         "fixed": "public void foo ( ) { int x = 0 ; if ( x == 1 ) { } }"},
        {"buggy": "public int add ( int a , int b ) { return a - b ; }",
         "fixed": "public int add ( int a , int b ) { return a + b ; }"},
    ]

    retriever = BM25Retriever()
    retriever.build_index(train_data, "bugs2fix")

    query = {"buggy": "public void bar ( ) { int y = 0 ; if ( y = 2 ) { } }"}
    results = retriever.retrieve(query, "bugs2fix", top_k=1)

    print("\nQuery:", query["buggy"][:60])
    print("Retrieved:", results[0]["buggy"][:60])
    print("Score:", results[0]["bm25_score"])

    prompt = format_rag_prompt(query, results, "bugs2fix")
    print(f"\nFull RAG Prompt:\n{prompt}")
