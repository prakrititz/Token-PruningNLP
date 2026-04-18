#!/usr/bin/env python3
"""
Simple script to convert buggy/fixed file pairs to JSONL format
"""

import json
import os
from pathlib import Path


def convert_to_jsonl(buggy_file, fixed_file, output_file):
    """
    Convert buggy and fixed code files to JSONL format.
    
    Args:
        buggy_file: Path to .buggy file
        fixed_file: Path to .fixed file
        output_file: Path to output .jsonl file
    """
    buggy_file = Path(buggy_file)
    fixed_file = Path(fixed_file)
    output_file = Path(output_file)
    
    # Read buggy codes
    with open(buggy_file, 'r', encoding='utf-8') as f:
        buggy_lines = f.readlines()
    
    # Read fixed codes
    with open(fixed_file, 'r', encoding='utf-8') as f:
        fixed_lines = f.readlines()
    
    # Ensure same number of lines
    if len(buggy_lines) != len(fixed_lines):
        print(f"Warning: Different number of lines in {buggy_file} ({len(buggy_lines)}) and {fixed_file} ({len(fixed_lines)})")
    
    # Create JSONL
    with open(output_file, 'w', encoding='utf-8') as out:
        for buggy, fixed in zip(buggy_lines, fixed_lines):
            entry = {
                "buggy": buggy.strip(),
                "fixed": fixed.strip()
            }
            out.write(json.dumps(entry) + '\n')
    
    print(f"✓ Created {output_file} with {min(len(buggy_lines), len(fixed_lines))} entries")


def main():
    """Convert all train/val/test pairs to JSONL"""
    
    # Paths
    base_dir = Path("BugFix/data")
    output_dir = Path("BugFix/data")
    
    # Define file pairs
    pairs = [
        ("train.buggy-fixed.buggy", "train.buggy-fixed.fixed", "train.jsonl"),
        ("valid.buggy-fixed.buggy", "valid.buggy-fixed.fixed", "valid.jsonl"),
        ("test.buggy-fixed.buggy", "test.buggy-fixed.fixed", "test.jsonl"),
    ]
    
    print("Converting buggy/fixed pairs to JSONL format...\n")
    
    for buggy, fixed, output in pairs:
        buggy_path = base_dir / buggy
        fixed_path = base_dir / fixed
        output_path = output_dir / output
        
        if buggy_path.exists() and fixed_path.exists():
            convert_to_jsonl(buggy_path, fixed_path, output_path)
        else:
            print(f"✗ Missing files: {buggy_path} or {fixed_path}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
