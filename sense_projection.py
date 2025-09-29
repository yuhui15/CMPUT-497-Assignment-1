"""
Sense Projection Script
Projects BabelNet sense annotations from English to target language via word alignments.

Required input files:
1. se13_tokens.xlsx - Maps tokens to instance IDs
2. se13.key.txt - Gold sense annotations (instance_id bn:########x)
3. alignment output file - Word alignments (e.g., align_out_inter.txt)
4. english_tokens.txt - English tokens (one sentence per line)
5. chinese_tokens.txt - Target language tokens (one sentence per line)

Output:
senses.tsv - BabelNet ID and target lemma pairs (tab-separated)
"""

import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import argparse


class SenseProjector:
    """Projects sense annotations via word alignments."""
    
    def __init__(self):
        self.instance_to_bn = {}  # instance_id -> BabelNet ID
        self.sentence_to_instances = defaultdict(list)  # sentence_id -> [(token_index, instance_id)]
        self.alignments = []  # List of alignments per sentence
        self.english_tokens = []  # English tokens per sentence
        self.target_tokens = []  # Target language tokens per sentence
        
    def load_sense_annotations(self, key_file: str):
        """
        Load gold sense annotations from se13.key.txt
        
        Format: instance_id bn:########x
        Example: semeval2013.d000.s000.t000 bn:00034587n
        """
        print(f"Loading sense annotations from {key_file}...")
        
        with open(key_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    instance_id = parts[0]
                    bn_ids = parts[1:]  # Can have multiple BabelNet IDs
                    
                    # Only store if there's exactly ONE sense
                    # (Assignment says: "If an English token is tagged with multiple synsets, do not project")
                    if len(bn_ids) == 1:
                        self.instance_to_bn[instance_id] = bn_ids[0]
        
        print(f"  Loaded {len(self.instance_to_bn)} unique sense annotations")
    
    def load_token_mappings(self, tokens_file: str):
        """
        Load se13_tokens.xlsx to map sentence positions to instance IDs.
        
        Columns: sentence_id, type, lemma, pos, raw_text, instance_id
        """
        print(f"Loading token mappings from {tokens_file}...")
        
        df = pd.read_excel(tokens_file)
        
        # Group by sentence
        for sentence_id, group in df.groupby('sentence_id', sort=False):
            token_index = 0
            for _, row in group.iterrows():
                instance_id = row['instance_id']
                
                # Only record tokens with instance IDs (content words)
                if pd.notna(instance_id) and instance_id != 'None':
                    self.sentence_to_instances[sentence_id].append((token_index, instance_id))
                
                token_index += 1
        
        print(f"  Loaded mappings for {len(self.sentence_to_instances)} sentences")
    
    def load_alignments(self, alignment_file: str):
        """
        Load word alignments from SimAlign output.
        
        Format: sent_id TAB alignment_pairs
        Example: 0\t0-5 1-3 2-7
        """
        print(f"Loading alignments from {alignment_file}...")
        
        with open(alignment_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    self.alignments.append([])
                    continue
                
                parts = line.split('\t')
                if len(parts) == 2:
                    # Parse alignment pairs
                    align_pairs = []
                    for pair in parts[1].split():
                        if '-' in pair:
                            src, tgt = pair.split('-')
                            align_pairs.append((int(src), int(tgt)))
                    self.alignments.append(align_pairs)
                else:
                    self.alignments.append([])
        
        print(f"  Loaded alignments for {len(self.alignments)} sentences")
    
    def load_token_files(self, english_file: str, target_file: str):
        """Load tokenized files (one sentence per line, space-separated)."""
        print(f"Loading token files...")
        
        with open(english_file, 'r', encoding='utf-8') as f:
            self.english_tokens = [line.strip().split() for line in f if line.strip()]
        
        with open(target_file, 'r', encoding='utf-8') as f:
            self.target_tokens = [line.strip().split() for line in f if line.strip()]
        
        print(f"  English: {len(self.english_tokens)} sentences")
        print(f"  Target: {len(self.target_tokens)} sentences")
    
    def project_senses(self) -> List[Tuple[str, str]]:
        """
        Project senses from English to target language via alignments.
        
        Returns:
            List of (babelnet_id, target_lemma) tuples
        """
        print("\nProjecting senses...")
        
        projected_senses = []  # List of (bn_id, target_lemma)
        
        # Get sentence IDs in order
        sentence_ids = sorted(self.sentence_to_instances.keys())
        
        for sent_idx, sentence_id in enumerate(sentence_ids):
            if sent_idx >= len(self.alignments):
                continue
            
            # Get instance mappings for this sentence
            token_to_instance = dict(self.sentence_to_instances[sentence_id])
            
            # Get alignments for this sentence
            sent_alignments = self.alignments[sent_idx]
            
            # Get tokens for this sentence
            if sent_idx >= len(self.english_tokens) or sent_idx >= len(self.target_tokens):
                continue
            
            en_tokens = self.english_tokens[sent_idx]
            tgt_tokens = self.target_tokens[sent_idx]
            
            # For each alignment pair
            for en_idx, tgt_idx in sent_alignments:
                # Check if English token has an instance ID
                if en_idx in token_to_instance:
                    instance_id = token_to_instance[en_idx]
                    
                    # Check if this instance has a sense annotation
                    if instance_id in self.instance_to_bn:
                        bn_id = self.instance_to_bn[instance_id]
                        
                        # Get target token (validate indices)
                        if tgt_idx < len(tgt_tokens):
                            target_lemma = tgt_tokens[tgt_idx].lower()
                            projected_senses.append((bn_id, target_lemma))
        
        print(f"  Projected {len(projected_senses)} senses to target language")
        return projected_senses
    
    def save_output(self, senses: List[Tuple[str, str]], output_file: str):
        """
        Save projected senses to TSV file.
        
        Format: bn:########x TAB target_lemma
        """
        print(f"\nSaving output to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for bn_id, lemma in senses:
                f.write(f"{bn_id}\t{lemma}\n")
        
        print(f"  Saved {len(senses)} sense projections")
    
    def show_statistics(self, senses: List[Tuple[str, str]]):
        """Display projection statistics."""
        print("\n" + "="*60)
        print("PROJECTION STATISTICS")
        print("="*60)
        
        # Count unique BabelNet IDs and lemmas
        unique_bn_ids = set(bn for bn, _ in senses)
        unique_lemmas = set(lemma for _, lemma in senses)
        
        print(f"Total projections: {len(senses)}")
        print(f"Unique BabelNet IDs: {len(unique_bn_ids)}")
        print(f"Unique target lemmas: {len(unique_lemmas)}")
        
        # Show sample projections
        print("\nSample projections:")
        for i, (bn_id, lemma) in enumerate(senses[:10]):
            print(f"  {bn_id} -> {lemma}")
        
        if len(senses) > 10:
            print(f"  ... and {len(senses) - 10} more")
    
    def show_example_sentence(self, sent_idx: int = 0):
        """Show a detailed example of sense projection for one sentence."""
        print("\n" + "="*60)
        print(f"EXAMPLE SENTENCE (index {sent_idx})")
        print("="*60)
        
        sentence_ids = sorted(self.sentence_to_instances.keys())
        if sent_idx >= len(sentence_ids):
            print("Sentence index out of range")
            return
        
        sentence_id = sentence_ids[sent_idx]
        
        # Get data for this sentence
        en_tokens = self.english_tokens[sent_idx]
        tgt_tokens = self.target_tokens[sent_idx]
        alignments = self.alignments[sent_idx]
        token_to_instance = dict(self.sentence_to_instances[sentence_id])
        
        print(f"\nSentence ID: {sentence_id}")
        print(f"\nEnglish tokens ({len(en_tokens)}):")
        print("  " + " ".join(f"[{i}]{tok}" for i, tok in enumerate(en_tokens)))
        
        print(f"\nTarget tokens ({len(tgt_tokens)}):")
        print("  " + " ".join(f"[{i}]{tok}" for i, tok in enumerate(tgt_tokens)))
        
        print(f"\nAlignments ({len(alignments)} pairs):")
        for en_idx, tgt_idx in alignments:
            en_tok = en_tokens[en_idx] if en_idx < len(en_tokens) else "???"
            tgt_tok = tgt_tokens[tgt_idx] if tgt_idx < len(tgt_tokens) else "???"
            print(f"  [{en_idx}]{en_tok} <-> [{tgt_idx}]{tgt_tok}")
        
        print(f"\nSense projections:")
        projection_count = 0
        for en_idx, tgt_idx in alignments:
            if en_idx in token_to_instance:
                instance_id = token_to_instance[en_idx]
                if instance_id in self.instance_to_bn:
                    bn_id = self.instance_to_bn[instance_id]
                    en_tok = en_tokens[en_idx]
                    tgt_tok = tgt_tokens[tgt_idx] if tgt_idx < len(tgt_tokens) else "???"
                    print(f"  {bn_id}: [{en_idx}]{en_tok} -> [{tgt_idx}]{tgt_tok}")
                    projection_count += 1
        
        if projection_count == 0:
            print("  (No sense projections in this sentence)")


def main():
    parser = argparse.ArgumentParser(description='Project BabelNet senses via word alignments')
    parser.add_argument('--tokens', type=str, default='se13_tokens.xlsx',
                       help='Path to se13_tokens.xlsx')
    parser.add_argument('--key', type=str, default='se13.key.txt',
                       help='Path to se13.key.txt (gold senses)')
    parser.add_argument('--alignments', type=str, required=True,
                       help='Path to alignment file (e.g., align_out_inter.txt)')
    parser.add_argument('--english', type=str, default='english_tokens.txt',
                       help='Path to English token file')
    parser.add_argument('--target', type=str, default='chinese_tokens.txt',
                       help='Path to target language token file')
    parser.add_argument('--output', type=str, default='senses.tsv',
                       help='Output file for projected senses')
    parser.add_argument('--show-example', type=int, default=None,
                       help='Show detailed example for specific sentence index')
    
    args = parser.parse_args()
    
    # Initialize projector
    projector = SenseProjector()
    
    # Load all required data
    projector.load_sense_annotations(args.key)
    projector.load_token_mappings(args.tokens)
    projector.load_alignments(args.alignments)
    projector.load_token_files(args.english, args.target)
    
    # Project senses
    senses = projector.project_senses()
    
    # Save output
    projector.save_output(senses, args.output)
    
    # Show statistics
    projector.show_statistics(senses)
    
    # Show example if requested
    if args.show_example is not None:
        projector.show_example_sentence(args.show_example)
    else:
        # Show first sentence by default
        projector.show_example_sentence(0)
    
    print("\n" + "="*60)
    print("✓ Sense projection complete!")
    print("="*60)


if __name__ == "__main__":
    main()