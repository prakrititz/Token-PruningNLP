"""
Process BugFixData with code-aware compression
"""

import os
import json
from bug2fix_parser import process_bugfix_data

if __name__ == "__main__":
    data_dir = "BugFixData"
    output_dir = "BugFixData_Compressed"
    compression_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Processing BugFixData with compression algorithm...")
    process_bugfix_data(data_dir, output_dir, compression_ratios)
    
    print("\n✓ Complete! Output in BugFixData_Compressed/")
