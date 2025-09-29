"""
Alignment Quality Evaluation (Without Gold Standard)
Provides various methods to assess alignment quality for your report.
"""

import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import argparse
import random


class AlignmentEvaluator:
    """Evaluate alignment quality without gold standard."""
    
    def __init__(self, english_file: str, target_file: str, alignment_file: str):
        """Load alignment data."""
        print("Loading files...")
        
        # Load tokens
        with open(english_file, 'r', encoding='utf-8') as f:
            self.english_sents = [line.strip().split() for line in f if line.strip()]
        
        with open(target_file, 'r', encoding='utf-8') as f:
            self.target_sents = [line.strip().split() for line in f if line.strip()]
        
        # Load alignments
        self.alignments = []
        with open(alignment_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    self.alignments.append([])
                    continue
                
                parts = line.split('\t')
                if len(parts) == 2:
                    align_pairs = []
                    for pair in parts[1].split():
                        if '-' in pair:
                            src, tgt = pair.split('-')
                            align_pairs.append((int(src), int(tgt)))
                    self.alignments.append(align_pairs)
                else:
                    self.alignments.append([])
        
        print(f"  Loaded {len(self.english_sents)} sentence pairs")
        print(f"  Loaded {len(self.alignments)} alignments\n")
    
    def compute_basic_statistics(self):
        """Compute basic alignment statistics."""
        print("="*70)
        print("BASIC STATISTICS")
        print("="*70)
        
        total_alignments = sum(len(a) for a in self.alignments)
        avg_alignments = total_alignments / len(self.alignments) if self.alignments else 0
        
        # Count sentences with no alignments
        no_align = sum(1 for a in self.alignments if len(a) == 0)
        
        # Alignment distribution
        align_counts = [len(a) for a in self.alignments]
        min_align = min(align_counts) if align_counts else 0
        max_align = max(align_counts) if align_counts else 0
        
        print(f"Total sentences: {len(self.alignments)}")
        print(f"Total alignments: {total_alignments}")
        print(f"Average alignments per sentence: {avg_alignments:.2f}")
        print(f"Sentences with no alignments: {no_align} ({100*no_align/len(self.alignments):.1f}%)")
        print(f"Min alignments in a sentence: {min_align}")
        print(f"Max alignments in a sentence: {max_align}")
        
        # Coverage statistics
        english_coverage = []
        target_coverage = []
        
        for sent_idx, aligns in enumerate(self.alignments):
            if sent_idx >= len(self.english_sents) or sent_idx >= len(self.target_sents):
                continue
            
            en_len = len(self.english_sents[sent_idx])
            tgt_len = len(self.target_sents[sent_idx])
            
            if en_len > 0 and tgt_len > 0:
                en_aligned = len(set(i for i, j in aligns))
                tgt_aligned = len(set(j for i, j in aligns))
                
                english_coverage.append(en_aligned / en_len)
                target_coverage.append(tgt_aligned / tgt_len)
        
        avg_en_coverage = sum(english_coverage) / len(english_coverage) if english_coverage else 0
        avg_tgt_coverage = sum(target_coverage) / len(target_coverage) if target_coverage else 0
        
        print(f"\nCoverage:")
        print(f"  Average English token coverage: {100*avg_en_coverage:.1f}%")
        print(f"  Average Target token coverage: {100*avg_tgt_coverage:.1f}%")
    
    def analyze_alignment_patterns(self):
        """Analyze one-to-one, one-to-many patterns."""
        print("\n" + "="*70)
        print("ALIGNMENT PATTERNS")
        print("="*70)
        
        one_to_one = 0
        one_to_many_src = 0
        many_to_one_tgt = 0
        many_to_many = 0
        
        for aligns in self.alignments:
            if not aligns:
                continue
            
            src_counts = Counter(i for i, j in aligns)
            tgt_counts = Counter(j for i, j in aligns)
            
            for i, j in aligns:
                src_count = src_counts[i]
                tgt_count = tgt_counts[j]
                
                if src_count == 1 and tgt_count == 1:
                    one_to_one += 1
                elif src_count == 1 and tgt_count > 1:
                    many_to_one_tgt += 1
                elif src_count > 1 and tgt_count == 1:
                    one_to_many_src += 1
                else:
                    many_to_many += 1
        
        total = one_to_one + one_to_many_src + many_to_one_tgt + many_to_many
        
        if total > 0:
            print(f"One-to-one: {one_to_one} ({100*one_to_one/total:.1f}%)")
            print(f"One-to-many (source): {one_to_many_src} ({100*one_to_many_src/total:.1f}%)")
            print(f"Many-to-one (target): {many_to_one_tgt} ({100*many_to_one_tgt/total:.1f}%)")
            print(f"Many-to-many: {many_to_many} ({100*many_to_many/total:.1f}%)")
        
        print("\nInterpretation:")
        print("  - High one-to-one: Good for simple translations")
        print("  - Many one-to-many/many-to-one: Expected for morphologically different languages")
        print("  - High many-to-many: May indicate alignment issues")
    
    def find_unaligned_tokens(self, num_examples=5):
        """Find examples of unaligned tokens."""
        print("\n" + "="*70)
        print("UNALIGNED TOKENS (Potential Issues)")
        print("="*70)
        
        unaligned_examples = []
        
        for sent_idx, aligns in enumerate(self.alignments):
            if sent_idx >= len(self.english_sents) or sent_idx >= len(self.target_sents):
                continue
            
            en_tokens = self.english_sents[sent_idx]
            tgt_tokens = self.target_sents[sent_idx]
            
            en_aligned = set(i for i, j in aligns)
            tgt_aligned = set(j for i, j in aligns)
            
            en_unaligned = [i for i in range(len(en_tokens)) if i not in en_aligned]
            tgt_unaligned = [j for j in range(len(tgt_tokens)) if j not in tgt_aligned]
            
            if en_unaligned or tgt_unaligned:
                unaligned_examples.append({
                    'sent_idx': sent_idx,
                    'en_tokens': en_tokens,
                    'tgt_tokens': tgt_tokens,
                    'en_unaligned': en_unaligned,
                    'tgt_unaligned': tgt_unaligned,
                    'en_unaligned_words': [en_tokens[i] for i in en_unaligned],
                    'tgt_unaligned_words': [tgt_tokens[j] for j in tgt_unaligned]
                })
        
        print(f"Sentences with unaligned tokens: {len(unaligned_examples)}")
        print(f"\nShowing {min(num_examples, len(unaligned_examples))} examples:\n")
        
        for i, ex in enumerate(random.sample(unaligned_examples, min(num_examples, len(unaligned_examples)))):
            print(f"Example {i+1} (Sentence {ex['sent_idx']}):")
            print(f"  English unaligned: {ex['en_unaligned_words']}")
            print(f"  Target unaligned: {ex['tgt_unaligned_words']}")
            print()
    
    def show_manual_inspection_samples(self, num_samples=10, random_seed=42):
        """Show samples for manual quality inspection."""
        print("\n" + "="*70)
        print("MANUAL INSPECTION SAMPLES")
        print("="*70)
        print("Review these alignments manually to assess quality\n")
        
        random.seed(random_seed)
        sample_indices = random.sample(range(len(self.alignments)), 
                                      min(num_samples, len(self.alignments)))
        
        for idx in sample_indices:
            if idx >= len(self.english_sents) or idx >= len(self.target_sents):
                continue
            
            en_tokens = self.english_sents[idx]
            tgt_tokens = self.target_sents[idx]
            aligns = self.alignments[idx]
            
            print(f"--- Sentence {idx} ---")
            print(f"EN:  {' '.join(f'[{i}]{tok}' for i, tok in enumerate(en_tokens))}")
            print(f"TGT: {' '.join(f'[{j}]{tok}' for j, tok in enumerate(tgt_tokens))}")
            print(f"Alignments ({len(aligns)} pairs):")
            
            for i, j in aligns:
                en_word = en_tokens[i] if i < len(en_tokens) else "???"
                tgt_word = tgt_tokens[j] if j < len(tgt_tokens) else "???"
                print(f"  [{i}]{en_word} <-> [{j}]{tgt_word}")
            
            print()
    
    def identify_typical_errors(self):
        """Identify common alignment error patterns."""
        print("\n" + "="*70)
        print("TYPICAL ERROR PATTERNS")
        print("="*70)
        
        # 1. Function words analysis
        print("\n1. Function Words (often problematic):")
        function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                         'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by'}
        
        function_word_aligns = 0
        function_word_total = 0
        
        for sent_idx, aligns in enumerate(self.alignments):
            if sent_idx >= len(self.english_sents):
                continue
            
            en_tokens = [t.lower() for t in self.english_sents[sent_idx]]
            aligned_indices = set(i for i, j in aligns)
            
            for i, token in enumerate(en_tokens):
                if token in function_words:
                    function_word_total += 1
                    if i in aligned_indices:
                        function_word_aligns += 1
        
        if function_word_total > 0:
            print(f"   Function words aligned: {function_word_aligns}/{function_word_total} "
                  f"({100*function_word_aligns/function_word_total:.1f}%)")
            print(f"   Expected: Lower alignment rate (function words often have no direct translation)")
        
        # 2. Very long alignment chains
        print("\n2. Suspicious Alignment Chains:")
        max_chain_src = 0
        max_chain_tgt = 0
        
        for aligns in self.alignments:
            src_counts = Counter(i for i, j in aligns)
            tgt_counts = Counter(j for i, j in aligns)
            
            if src_counts:
                max_chain_src = max(max_chain_src, max(src_counts.values()))
            if tgt_counts:
                max_chain_tgt = max(max_chain_tgt, max(tgt_counts.values()))
        
        print(f"   Max alignments from single source word: {max_chain_src}")
        print(f"   Max alignments to single target word: {max_chain_tgt}")
        print(f"   Note: Values > 5 may indicate errors")
        
        # 3. Crossing alignments (potential issues)
        print("\n3. Crossing Alignments:")
        crossing_count = 0
        total_sentences = 0
        
        for aligns in self.alignments:
            if len(aligns) < 2:
                continue
            
            total_sentences += 1
            has_crossing = False
            
            for (i1, j1) in aligns:
                for (i2, j2) in aligns:
                    if i1 < i2 and j1 > j2:  # Crossing
                        has_crossing = True
                        break
                if has_crossing:
                    break
            
            if has_crossing:
                crossing_count += 1
        
        if total_sentences > 0:
            print(f"   Sentences with crossing alignments: {crossing_count}/{total_sentences} "
                  f"({100*crossing_count/total_sentences:.1f}%)")
            print(f"   Expected: Some crossings normal for different word orders")
    
    def compare_methods(self, other_alignment_file: str, method_name: str):
        """Compare with alignments from different method."""
        print("\n" + "="*70)
        print(f"COMPARISON WITH {method_name}")
        print("="*70)
        
        # Load other alignments
        other_aligns = []
        with open(other_alignment_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    other_aligns.append(set())
                    continue
                
                parts = line.split('\t')
                if len(parts) == 2:
                    align_set = set()
                    for pair in parts[1].split():
                        if '-' in pair:
                            src, tgt = pair.split('-')
                            align_set.add((int(src), int(tgt)))
                    other_aligns.append(align_set)
                else:
                    other_aligns.append(set())
        
        # Compare
        agreement = []
        
        for idx in range(min(len(self.alignments), len(other_aligns))):
            set1 = set(self.alignments[idx])
            set2 = other_aligns[idx]
            
            if len(set1) == 0 and len(set2) == 0:
                continue
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            if union > 0:
                agreement.append(intersection / union)
        
        if agreement:
            avg_agreement = sum(agreement) / len(agreement)
            print(f"Average agreement (Jaccard): {avg_agreement:.3f}")
            print(f"Interpretation:")
            print(f"  > 0.7: High agreement (methods similar)")
            print(f"  0.4-0.7: Moderate agreement (expected)")
            print(f"  < 0.4: Low agreement (methods very different)")
    
    def generate_report_for_assignment(self, output_file: str = "alignment_quality_report.txt"):
        """Generate a comprehensive report for the assignment."""
        import sys
        
        # Redirect output to file
        original_stdout = sys.stdout
        with open(output_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            print("ALIGNMENT QUALITY EVALUATION REPORT")
            print("="*70)
            print()
            
            self.compute_basic_statistics()
            self.analyze_alignment_patterns()
            self.identify_typical_errors()
            self.find_unaligned_tokens(num_examples=5)
            self.show_manual_inspection_samples(num_samples=10)
            
        sys.stdout = original_stdout
        print(f"\n✓ Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate alignment quality without gold standard')
    parser.add_argument('--english', type=str, default='english_tokens.txt',
                       help='English token file')
    parser.add_argument('--target', type=str, default='chinese_tokens.txt',
                       help='Target token file')
    parser.add_argument('--alignments', type=str, required=True,
                       help='Alignment file to evaluate')
    parser.add_argument('--compare', type=str, default=None,
                       help='Compare with another alignment file')
    parser.add_argument('--compare-name', type=str, default='Other Method',
                       help='Name of comparison method')
    parser.add_argument('--report', type=str, default='alignment_quality_report.txt',
                       help='Output report file')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of manual inspection samples')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AlignmentEvaluator(args.english, args.target, args.alignments)
    
    # Run evaluations
    evaluator.compute_basic_statistics()
    evaluator.analyze_alignment_patterns()
    evaluator.identify_typical_errors()
    evaluator.find_unaligned_tokens(num_examples=5)
    evaluator.show_manual_inspection_samples(num_samples=args.samples)
    
    # Compare with other method if provided
    if args.compare:
        evaluator.compare_methods(args.compare, args.compare_name)
    
    # Generate report
    print("\n" + "="*70)
    print("Generating comprehensive report...")
    evaluator.generate_report_for_assignment(args.report)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nUse these analyses in your assignment report:")
    print(f"  1. Basic statistics (coverage, alignment density)")
    print(f"  2. Alignment patterns (one-to-one vs many-to-many)")
    print(f"  3. Typical errors (function words, crossings)")
    print(f"  4. Manual inspection examples (good and bad alignments)")


if __name__ == "__main__":
    main()
