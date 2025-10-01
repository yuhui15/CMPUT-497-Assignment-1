import pandas as pd
from simalign import SentenceAligner
import jieba
import warnings
import numpy as np

# Suppress specific runtime warnings from sklearn
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='divide by zero encountered')
warnings.filterwarnings('ignore', message='overflow encountered')
warnings.filterwarnings('ignore', message='invalid value encountered')

# Also configure numpy to not show warnings
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# Mapping of matching method shortcuts to their full names in the output
MATCHING_METHOD_MAP = {
    'm': 'mwmf',      # Match
    'a': 'inter',     # ArgMax
    'i': 'itermax',   # IterMax
}

def read_english_tokens_from_excel(excel_file):
    """
    Read English tokens from se13_tokens.xlsx and organize by sentence_id.
    Returns a list of tokenized sentences and writes to file.
    """
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Get unique sentence IDs in order of appearance
    sentence_ids = df['sentence_id'].unique()
    
    sentences = []
    
    for sent_id in sentence_ids:
        # Get all tokens for this sentence in order
        sentence_df = df[df['sentence_id'] == sent_id]
        # Use raw_text column and convert all to strings
        tokens = [str(token) for token in sentence_df['raw_text'].tolist()]
        sentences.append(tokens)
    
    return sentences

def read_chinese_translations(translation_file):
    """
    Read Chinese translations from translations.txt and tokenize them.
    Returns a list of tokenized Chinese sentences.
    """
    chinese_sentences = []
    
    with open(translation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Tokenize Chinese sentence using jieba
                tokens = list(jieba.cut(line))
                chinese_sentences.append(tokens)
    
    return chinese_sentences

def write_tokens_to_file(tokens_list, output_file):
    """
    Write tokenized sentences to a file, one sentence per line.
    Tokens are separated by spaces.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for tokens in tokens_list:
            # Ensure all tokens are strings
            tokens_str = [str(token) for token in tokens]
            f.write(' '.join(tokens_str) + '\n')

def align_sentences(english_tokens, chinese_tokens, matching_method='i'):
    """
    Align English and Chinese tokens using SimAlign.
    
    Args:
        english_tokens: List of English tokenized sentences
        chinese_tokens: List of Chinese tokenized sentences
        matching_method: 'i' (itermax), 'm' (match), or 'a' (argmax)
    
    Returns:
        List of alignments for all sentence pairs
    """
    # Initialize SentenceAligner with specified matching method
    myaligner = SentenceAligner(matching_methods=matching_method)
    
    # Get the full method name for extracting results
    method_key = MATCHING_METHOD_MAP.get(matching_method, 'itermax')
    
    print(f"  Using matching method: '{matching_method}' -> '{method_key}'")
    
    all_alignments = []
    
    # Process each sentence pair
    for idx, (eng_sent, chi_sent) in enumerate(zip(english_tokens, chinese_tokens)):
        try:
            # Get word alignments
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alignments = myaligner.get_word_aligns(eng_sent, chi_sent)
            
            # Extract alignments using the correct method key
            method_alignments = alignments.get(method_key, [])
            all_alignments.append(method_alignments)
            
        except Exception as e:
            # If alignment fails for a sentence, add empty alignment
            print(f"  Warning: Failed to align sentence {idx}: {str(e)}")
            all_alignments.append([])
        
        # Print progress every 50 sentences
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(english_tokens)} sentence pairs...")
    
    return all_alignments

def write_alignments_to_file(alignments, output_file):
    """
    Write alignments to file in the required format.
    Each line contains alignment pairs for one sentence.
    Format: list of (src_idx, tgt_idx) pairs per sentence
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for alignment in alignments:
            # Write alignment as string representation of list
            f.write(str(alignment) + '\n')

def main():
    # File paths
    excel_file = 'se13_tokens.xlsx'
    translation_file = 'translations.txt'
    english_tokens_file = 'english_tokens.txt'
    chinese_tokens_file = 'chinese_tokens.txt'
    alignments_file = 'alignments.txt'
    
    # Choose matching method: 'i' (itermax), 'm' (match), or 'a' (argmax)
    MATCHING_METHOD = 'i'  # Change this to 'm' or 'a' to experiment
    
    print("="*60)
    print("Word Alignment Script - Task 3")
    print("="*60)
    
    print("\nStep 1: Reading English tokens from Excel file...")
    english_sentences = read_english_tokens_from_excel(excel_file)
    print(f"✓ Read {len(english_sentences)} English sentences.")
    
    print("\nStep 2: Writing English tokens to file...")
    write_tokens_to_file(english_sentences, english_tokens_file)
    print(f"✓ English tokens written to '{english_tokens_file}'")
    
    print("\nStep 3: Reading and tokenizing Chinese translations...")
    chinese_sentences = read_chinese_translations(translation_file)
    print(f"✓ Read {len(chinese_sentences)} Chinese sentences.")
    
    print("\nStep 4: Writing Chinese tokens to file...")
    write_tokens_to_file(chinese_sentences, chinese_tokens_file)
    print(f"✓ Chinese tokens written to '{chinese_tokens_file}'")
    
    # Verify both have same number of sentences
    if len(english_sentences) != len(chinese_sentences):
        print(f"\n⚠ Warning: Number of English sentences ({len(english_sentences)}) "
              f"does not match Chinese sentences ({len(chinese_sentences)})")
        print("  Alignment will only process the minimum number of sentences.")
        min_len = min(len(english_sentences), len(chinese_sentences))
        english_sentences = english_sentences[:min_len]
        chinese_sentences = chinese_sentences[:min_len]
    
    print(f"\nStep 5: Performing word alignment using SimAlign (method: {MATCHING_METHOD})...")
    print("  (This may take a few minutes...)")
    alignments = align_sentences(english_sentences, chinese_sentences, MATCHING_METHOD)
    print(f"✓ Aligned {len(alignments)} sentence pairs.")
    
    print("\nStep 6: Writing alignments to file...")
    write_alignments_to_file(alignments, alignments_file)
    print(f"✓ Alignments written to '{alignments_file}'")
    
    print("\n" + "="*60)
    print("Alignment process completed successfully!")
    print("="*60)
    
    # Print some statistics
    total_alignments = sum(len(a) for a in alignments)
    avg_alignments = total_alignments / len(alignments) if alignments else 0
    empty_alignments = sum(1 for a in alignments if len(a) == 0)
    
    print(f"\nStatistics:")
    print(f"  • Matching method: {MATCHING_METHOD} ({MATCHING_METHOD_MAP[MATCHING_METHOD]})")
    print(f"  • Total sentences: {len(alignments)}")
    print(f"  • Total alignment pairs: {total_alignments}")
    print(f"  • Average alignments per sentence: {avg_alignments:.2f}")
    if empty_alignments > 0:
        print(f"  • Sentences with no alignments: {empty_alignments}")
    
    # Show a sample alignment
    if alignments and len(alignments) > 0:
        print(f"\n" + "-"*60)
        print("Sample alignment (first sentence):")
        print(f"  English tokens: {english_sentences[0]}")
        print(f"  Chinese tokens: {chinese_sentences[0]}")
        print(f"  Alignments: {alignments[0]}")
        print("-"*60)

if __name__ == "__main__":
    main()