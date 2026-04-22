"""
Java Code Parser for BUG2FIX using javalang
Implements code compression with varying compression ratios


Dataset Construction (What this script does): You do NOT need to run a BLM to score things right now. 
This Python script represents Algorithm 1. Its job is to apply those pre-calculated priorities to the AST tokens and generate the <original_code, compressed_code> training pairs.
Later, you will train CodeT5 on this data.


They discovered that for the Bugs2Fix task, the LLM doesn't really care about variable names. It can figure out how to fix a bug just by looking at the structure of the code and the method signatures.

So, the authors created a Removal Priority List (published in Figure 1 of the paper). You do not need to use an AI to figure this out; the authors already gave you the cheat sheet.

For Bugs2Fix, the hit list from the paper is exactly this:

    Identifiers (Variables like VAR_1, date — Safest to delete, kill these first)

    Invocations (Function calls like .getInstance())

    Structure (Keywords like if, return)

    Symbols (Operators like =, {)

    Signatures (The method definition public static TYPE_1 init(...) — Protect at all costs)
"""

import javalang
import json
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import os


class TypeDictionary:
    """Maps code element types to priority values"""
    
    def __init__(self, types: List[str]):
        self.type_map = {}
        for idx, type_name in enumerate(types, start=1):
            self.type_map[type_name] = idx
    
    def get_type_value(self, type_name: str) -> int:
        return self.type_map.get(type_name, -1)


class SpanContent:
    """Represents a span of code content"""
    
    def __init__(self, start_word: int, end_word: int, code_splits: List[str]):
        self.start_word = start_word
        self.end_word = end_word
        self.tokens = code_splits[start_word:end_word]
    
    def __repr__(self):
        return f"SpanContent({self.start_word}, {self.end_word}, {self.tokens})"


class JavaCodeVisitor:
    """Extracts code spans from Java code using javalang"""
    
    def __init__(self, code: str):
        self.code = code
        self.code_splits = code.split()
        self.identifier_spans = []
        self.invocation_spans = []
        self.structure_spans = []
        self.statement_spans = []
        
        self._extract_spans()
    
    def _find_token_range(self, target_token: str, start_pos: int = 0) -> Tuple[int, int]:
        """Find word indices for a token in code_splits"""
        reconstructed = " ".join(self.code_splits)
        
        # Handle multi-token matches
        token_list = target_token.split()
        if len(token_list) == 1:
            for idx, token in enumerate(self.code_splits[start_pos:], start=start_pos):
                if token == target_token:
                    return idx, idx + 1
        else:
            for idx in range(start_pos, len(self.code_splits) - len(token_list) + 1):
                if self.code_splits[idx:idx + len(token_list)] == token_list:
                    return idx, idx + len(token_list)
        
        return -1, -1
    
    def _extract_spans(self):
        """Extract code spans using javalang"""
        try:
            # Wrap code to make it valid Java
            wrapped_code = f"public class Wrapper {{ {self.code} }}"
            tree = javalang.parse.parse(wrapped_code)
            
            # Extract method calls
            for _, node in tree.filter(javalang.tree.MethodInvocation):
                method_name = node.member
                start_idx, end_idx = self._find_token_range(method_name)
                if start_idx != -1:
                    self.invocation_spans.append(SpanContent(start_idx, end_idx, self.code_splits))
            
            # Extract identifiers (variable names)
            for _, node in tree.filter(javalang.tree.VariableDeclarator):
                var_name = node.name
                start_idx, end_idx = self._find_token_range(var_name)
                if start_idx != -1:
                    self.identifier_spans.append(SpanContent(start_idx, end_idx, self.code_splits))
            
            # Extract control structure keywords
            keywords = {"if", "else", "try", "catch", "finally", "for", "while", "do", "switch", "case"}
            for idx, token in enumerate(self.code_splits):
                if token in keywords:
                    self.structure_spans.append(SpanContent(idx, idx + 1, self.code_splits))
        
        except Exception as e:
            # If parsing fails, still collect basic keywords
            keywords = {"if", "else", "try", "catch", "finally", "for", "while", "do", "switch", "case"}
            for idx, token in enumerate(self.code_splits):
                if token in keywords:
                    self.structure_spans.append(SpanContent(idx, idx + 1, self.code_splits))
    
    def get_map(self) -> Dict[str, List[SpanContent]]:
        """Return map of all extracted spans"""
        return {
            "identifiers": self.identifier_spans,
            "function_invocation": self.invocation_spans,
            "function_structure": self.structure_spans
        }


class CodeCompressor:
    """Compresses Java code based on token importance"""
    
    SIMPLE_SYMBOLS = {
        "=", "+", "-", "*", "/", "%", "!", ">", "<", "|", "?", ":", "~", "&", "^",
        "(", "{", ")", "}", "[", ".", "]", ";", "\"", ",", "==", "++", "--", "!=",
        ">=", "<=", "&&", "||", "<<", ">>", ">>>", "'"
    }
    
    def __init__(self, type_dict: TypeDictionary):
        self.type_dict = type_dict
    
    def mark_flag(self, code_flag: List[int], span: SpanContent, flag: int):
        """Mark tokens in span with given flag"""
        for i in range(span.start_word, span.end_word):
            if i < len(code_flag):
                code_flag[i] = flag
    
    def get_removed_indices(self, code_splits: List[str], code_flag: List[int],
                          target_length: int, freq_flag: List[int]) -> List[int]:
        """Select indices to remove to reach target length"""
        removed = []
        remove_count = len(code_splits) - target_length
        
        if remove_count <= 0:
            return removed
        
        # Priority: lower flag values are removed first
        for priority in range(7):
            candidates = []
            for idx in range(len(code_splits) - 1, -1, -1):
                if code_flag[idx] == priority:
                    candidates.append(idx)
            
            # Sort by frequency (ascending)
            candidates.sort(key=lambda x: freq_flag[x])
            
            for idx in candidates:
                if len(removed) >= remove_count:
                    return removed
                removed.append(idx)
        
        return removed
    
    def compress(self, code: str, target_length: int, spans_map: Dict) -> str:
        """Compress code by removing low-priority tokens"""
        code_splits = code.split()
        
        if len(code_splits) <= target_length:
            return code
        
        # Initialize flags
        code_flag = [0] * len(code_splits)
        
        # Count token frequencies
        freq_flag = [0] * len(code_splits)
        freq_map = defaultdict(int)
        for token in code_splits:
            freq_map[token] += 1
        for i, token in enumerate(code_splits):
            freq_flag[i] = freq_map[token]
        
        # Mark identifiers
        for span in spans_map.get("identifiers", []):
            self.mark_flag(code_flag, span, self.type_dict.get_type_value("identifiers"))
        
        # Mark function structures
        for span in spans_map.get("function_structure", []):
            self.mark_flag(code_flag, span, self.type_dict.get_type_value("function_structure"))
        
        # Mark function invocations
        for span in spans_map.get("function_invocation", []):
            self.mark_flag(code_flag, span, self.type_dict.get_type_value("function_invocation"))
        
        # Mark symbols
        for i, token in enumerate(code_splits):
            if token in self.SIMPLE_SYMBOLS:
                code_flag[i] = self.type_dict.get_type_value("symbols")
        
        # Mark method signature (tokens before first '{')
        bracket_idx = -1
        for i, token in enumerate(code_splits):
            if token == "{":
                bracket_idx = i
                break
        
        if bracket_idx != -1:
            for i in range(bracket_idx):
                code_flag[i] = self.type_dict.get_type_value("method_signature")
        
        # Mark remaining unmapped tokens
        for i in range(len(code_flag)):
            if code_flag[i] == 0:
                code_flag[i] = 6  # Other
        
        # Get indices to remove
        remove_indices = self.get_removed_indices(code_splits, code_flag, target_length, freq_flag)
        
        # Remove selected tokens
        for idx in remove_indices:
            code_splits[idx] = ""
        
        # Reconstruct code
        result = " ".join(token for token in code_splits if token)
        return result


def process_bugfix_data(data_dir: str, output_dir: str, compression_ratios: List[float]):
    """Process BUG2FIX data with multiple compression ratios"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize type dictionary and compressor
    types = ["method_signature", "function_invocation", "identifiers", "function_structure", "symbols"]
    type_dict = TypeDictionary(types)
    compressor = CodeCompressor(type_dict)
    
    # Process each dataset file
    for split in ["train", "valid", "test"]:
        input_file = os.path.join(data_dir, f"{split}.jsonl")
        
        if not os.path.exists(input_file):
            print(f"Skipping {input_file} - not found")
            continue
        
        # Create output files for each compression ratio
        output_files = {}
        for ratio in compression_ratios:
            output_file = os.path.join(output_dir, f"bug2fix_{split}_compress_{ratio:.1f}.jsonl")
            output_files[ratio] = open(output_file, 'w', encoding='utf-8')
        
        print(f"\nProcessing {split} set...")
        stats = defaultdict(int)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    buggy = data.get("buggy", "")
                    fixed = data.get("fixed", "")
                    
                    # Skip empty or invalid code
                    if not buggy or not fixed:
                        stats["skipped_empty"] += 1
                        continue
                    
                    # Extract spans ONCE for both buggy and fixed versions
                    buggy_visitor = JavaCodeVisitor(buggy)
                    fixed_visitor = JavaCodeVisitor(fixed)
                    
                    buggy_spans = buggy_visitor.get_map()
                    fixed_spans = fixed_visitor.get_map()
                    
                    buggy_splits = buggy.split()
                    fixed_splits = fixed.split()
                    
                    # Process each compression ratio with pre-computed spans
                    for ratio in compression_ratios:
                        buggy_target = max(1, int(len(buggy_splits) * (1 - ratio)))
                        fixed_target = max(1, int(len(fixed_splits) * (1 - ratio)))
                        
                        compressed_buggy = compressor.compress(buggy, buggy_target, buggy_spans)
                        compressed_fixed = compressor.compress(fixed, fixed_target, fixed_spans)
                        
                        output_data = {
                            "buggy": compressed_buggy,
                            "fixed": compressed_fixed,
                            "compression_ratio": ratio,
                            "original_buggy_length": len(buggy_splits),
                            "original_fixed_length": len(fixed_splits),
                            "compressed_buggy_length": len(compressed_buggy.split()),
                            "compressed_fixed_length": len(compressed_fixed.split())
                        }
                        
                        output_files[ratio].write(json.dumps(output_data) + '\n')
                    
                    stats["processed"] += 1
                    
                    if line_num % 500 == 0:
                        print(f"  Processed {line_num} samples...")
                
                except Exception as e:
                    stats["error"] += 1
                    if line_num % 500 == 0:
                        print(f"  Error at line {line_num}: {str(e)[:50]}")
        
        # Close output files
        for ratio, f in output_files.items():
            f.close()
        
        print(f"Completed {split}:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped: {stats['skipped_empty']}")
        print(f"  Errors: {stats['error']}")


if __name__ == "__main__":
    # Input and output paths
    data_dir = "BugFixData"
    output_dir = "BugFixData_Compressed"
    
    # Compression ratios to try
    compression_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Starting BUG2FIX data processing...")
    process_bugfix_data(data_dir, output_dir, compression_ratios)
    print("\nCompleted!")
