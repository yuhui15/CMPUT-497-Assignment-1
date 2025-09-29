"""
Token Extraction and Organization Script (Fixed)
Extracts tokens from se13_tokens.xlsx and organizes them by sentence.
Also tokenizes Chinese sentences and creates SimAlign-compatible token files.
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Dict


def extract_english_tokens(excel_path: str, output_path: str) -> Dict[str, List[str]]:
    """
    Extract tokens from se13_tokens.xlsx and organize by sentence.
    
    Args:
        excel_path: Path to se13_tokens.xlsx
        output_path: Path for output token file
        
    Returns:
        Dictionary mapping sentence_id to list of tokens
    """
    print(f"Reading {excel_path}...")
    df = pd.read_excel(excel_path)
    
    # Group tokens by sentence_id
    sentence_tokens = {}
    for sentence_id, group in df.groupby('sentence_id', sort=False):
        # Extract raw_text tokens in order and convert to strings
        tokens = [str(token) for token in group['raw_text'].tolist()]
        sentence_tokens[sentence_id] = tokens
    
    print(f"Extracted tokens from {len(sentence_tokens)} sentences")
    
    # Write to output file (one sentence per line, tokens separated by spaces)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence_id in sorted(sentence_tokens.keys()):
            tokens = sentence_tokens[sentence_id]
            f.write(' '.join(tokens) + '\n')
    
    print(f"Saved English tokens to {output_path}")
    return sentence_tokens


def tokenize_chinese_sentences(input_path: str, output_path: str) -> Dict[int, List[str]]:
    """
    Tokenize Chinese sentences from segmented_translated_sentences_in_chinese.txt.
    Assumes the file contains pre-segmented Chinese text (space-separated words).
    
    Args:
        input_path: Path to input Chinese sentence file
        output_path: Path for output token file
        
    Returns:
        Dictionary mapping line number to list of tokens
    """
    print(f"Reading {input_path}...")
    
    sentence_tokens = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Split by spaces (assumes pre-segmented text)
            tokens = line.split()
            
            # Remove empty tokens and convert to strings
            tokens = [str(t) for t in tokens if t]
            
            sentence_tokens[idx] = tokens
    
    print(f"Tokenized {len(sentence_tokens)} Chinese sentences")
    
    # Write to output file (one sentence per line, tokens separated by spaces)
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx in sorted(sentence_tokens.keys()):
            tokens = sentence_tokens[idx]
            f.write(' '.join(tokens) + '\n')
    
    print(f"Saved Chinese tokens to {output_path}")
    return sentence_tokens


def verify_token_counts(english_tokens: Dict, chinese_tokens: Dict):
    """Verify that English and Chinese have the same number of sentences."""
    en_count = len(english_tokens)
    zh_count = len(chinese_tokens)
    
    print(f"\n=== Verification ===")
    print(f"English sentences: {en_count}")
    print(f"Chinese sentences: {zh_count}")
    
    if en_count == zh_count:
        print("✓ Token counts match!")
    else:
        print(f"⚠ Warning: Token counts don't match ({en_count} vs {zh_count})")


def main():
    """Main execution function."""
    # File paths
    english_excel = 'se13_tokens.xlsx'
    chinese_input = 'segmented_translated_sentences_in_chinese.txt'
    
    # Output paths for SimAlign-compatible token files
    english_output = 'english_tokens.txt'
    chinese_output = 'chinese_tokens.txt'
    
    print("=" * 60)
    print("Token Extraction and Organization")
    print("=" * 60)
    
    # Extract English tokens
    print("\n[1/2] Processing English tokens...")
    try:
        english_tokens = extract_english_tokens(english_excel, english_output)
    except FileNotFoundError:
        print(f"ERROR: Could not find {english_excel}")
        print("Please make sure the file is in the current directory.")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Tokenize Chinese sentences
    print("\n[2/2] Processing Chinese tokens...")
    try:
        chinese_tokens = tokenize_chinese_sentences(chinese_input, chinese_output)
    except FileNotFoundError:
        print(f"ERROR: Could not find {chinese_input}")
        print("Please make sure the file is in the current directory.")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Verify
    verify_token_counts(english_tokens, chinese_tokens)
    
    # Show sample output
    print("\n=== Sample Output ===")
    print("English (first sentence):")
    first_key = sorted(english_tokens.keys())[0]
    sample_tokens = english_tokens[first_key][:10]
    print(f"  {' '.join(sample_tokens)}{'...' if len(english_tokens[first_key]) > 10 else ''}")
    
    print("\nChinese (first sentence):")
    if chinese_tokens:
        sample_tokens = chinese_tokens[0][:10]
        print(f"  {' '.join(sample_tokens)}{'...' if len(chinese_tokens[0]) > 10 else ''}")
    
    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print(f"✓ English tokens saved to: {english_output}")
    print(f"✓ Chinese tokens saved to: {chinese_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()