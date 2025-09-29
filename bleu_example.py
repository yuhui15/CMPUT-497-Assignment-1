# Before running this, install Python's "sacrebleu" package: pip install sacrebleu

# Example: Using sacrebleu to evaluate a translation
import sacrebleu

# Reference translation (what the correct output should be)
reference = "The cat is sitting on the mat."

# Hypothesis (translation to evaluate)
hypothesis = "The cat is on the mat."

# Compute BLEU score
bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
print(bleu)
# Output: BLEU = 51.54 100.0/83.3/60.0/25.0 (BP = 0.867 ratio = 0.875 hyp_len = 7 ref_len = 8)
