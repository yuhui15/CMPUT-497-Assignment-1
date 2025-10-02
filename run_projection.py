import pandas as pd
import ast


def load_gold_sense_annotations(key_file):
    """
    Load gold sense annotations from se13.key.txt.
    
    Args:
        key_file: Path to se13.key.txt
        
    Returns:
        Dictionary mapping instance_id to synset_id
    """
    sense_dict = {}
    with open(key_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    instance_id, synset_id = parts
                    sense_dict[instance_id] = synset_id
    return sense_dict


def load_alignments(alignment_file):
    """
    Load word alignments from alignments.txt.
    
    Args:
        alignment_file: Path to alignments.txt
        
    Returns:
        List of alignment lists, where each alignment is [(eng_idx, chi_idx), ...]
    """
    alignments = []
    with open(alignment_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                alignment = ast.literal_eval(line)
                alignments.append(alignment)
    return alignments


def load_tokens(token_file):
    """
    Load tokenized sentences from file.
    
    Args:
        token_file: Path to token file
        
    Returns:
        List of sentences, where each sentence is a list of tokens
    """
    sentences = []
    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split()
                sentences.append(tokens)
    return sentences


def project_senses(se13_tokens_file, key_file, alignment_file, 
                   english_tokens_file, chinese_tokens_file, output_file):
    """
    Project English sense annotations onto aligned Chinese tokens.
    
    Args:
        se13_tokens_file: Path to se13_tokens.xlsx
        key_file: Path to se13.key.txt
        alignment_file: Path to alignments.txt
        english_tokens_file: Path to english_tokens.txt
        chinese_tokens_file: Path to chinese_tokens.txt
        output_file: Path to output senses.tsv file
    """
    # Load gold sense annotations
    print("Loading gold sense annotations...")
    sense_dict = load_gold_sense_annotations(key_file)
    print(f"Loaded {len(sense_dict)} sense annotations")
    
    # Load se13_tokens.xlsx
    print("Loading se13_tokens.xlsx...")
    df = pd.read_excel(se13_tokens_file)
    
    # Load alignments
    print("Loading alignments...")
    alignments = load_alignments(alignment_file)
    print(f"Loaded {len(alignments)} sentence alignments")
    
    # Load English and Chinese tokens
    print("Loading English and Chinese tokens...")
    english_sentences = load_tokens(english_tokens_file)
    chinese_sentences = load_tokens(chinese_tokens_file)
    print(f"Loaded {len(english_sentences)} English sentences")
    print(f"Loaded {len(chinese_sentences)} Chinese sentences")
    
    # Group se13 tokens by sentence_id
    print("Grouping tokens by sentence...")
    sentence_groups = df.groupby('sentence_id')
    
    # Map sentence_id to sentence index
    sentence_ids = sorted(df['sentence_id'].unique())
    sentence_id_to_idx = {sid: idx for idx, sid in enumerate(sentence_ids)}
    
    # Create index mapping for each sentence
    print("Creating token mappings...")
    sentence_token_map = {}
    for sentence_id, group in sentence_groups:
        tokens_data = []
        for _, row in group.iterrows():
            # Convert token to string for comparison
            token = str(row['lemma']) if pd.notna(row['lemma']) else str(row['raw_text'])
            tokens_data.append({
                'token': token,
                'raw_text': str(row['raw_text']) if pd.notna(row['raw_text']) else '',
                'instance_id': row['instance_id'] if pd.notna(row['instance_id']) and row['instance_id'] != 'None' else None,
                'type': row['type']
            })
        sentence_token_map[sentence_id] = tokens_data
    
    # Project senses
    print("Projecting senses...")
    projections = []
    
    for sentence_id in sentence_ids:
        if sentence_id not in sentence_id_to_idx:
            continue
            
        sent_idx = sentence_id_to_idx[sentence_id]
        
        if sent_idx >= len(alignments) or sent_idx >= len(english_sentences) or sent_idx >= len(chinese_sentences):
            continue
        
        alignment = alignments[sent_idx]
        english_tokens = english_sentences[sent_idx]
        chinese_tokens = chinese_sentences[sent_idx]
        se13_tokens = sentence_token_map.get(sentence_id, [])
        
        # Create mapping from English token position in english_tokens.txt to se13 token index
        eng_pos_to_se13_idx = {}
        se13_idx = 0
        
        for eng_pos, eng_token in enumerate(english_tokens):
            # Find matching token in se13_tokens
            while se13_idx < len(se13_tokens):
                se13_token_data = se13_tokens[se13_idx]
                se13_token = se13_token_data['raw_text']
                
                # Convert both to strings for comparison and handle case sensitivity
                # This handles the case where se13 tokens might be integers (e.g., "40")
                se13_token_str = str(se13_token).lower()
                eng_token_str = str(eng_token).lower()
                
                if se13_token_str == eng_token_str:
                    eng_pos_to_se13_idx[eng_pos] = se13_idx
                    se13_idx += 1
                    break
                else:
                    se13_idx += 1
        
        # Process alignments
        for eng_pos, chi_pos in alignment:
            if eng_pos >= len(english_tokens) or chi_pos >= len(chinese_tokens):
                continue
            
            # Find corresponding se13 token
            if eng_pos not in eng_pos_to_se13_idx:
                continue
            
            se13_idx = eng_pos_to_se13_idx[eng_pos]
            se13_token_data = se13_tokens[se13_idx]
            
            # Check if this token has an instance_id and sense annotation
            instance_id = se13_token_data['instance_id']
            if instance_id and instance_id in sense_dict:
                synset_id = sense_dict[instance_id]
                
                # Check if English token has multiple synsets (should skip)
                # Count how many times this instance_id appears in this sentence
                instance_count = sum(1 for t in se13_tokens if t['instance_id'] == instance_id)
                
                # Only project if unique (appears once)
                if instance_count == 1:
                    chinese_lemma = chinese_tokens[chi_pos]
                    projections.append((synset_id, chinese_lemma))
    
    # Write output
    print(f"Writing {len(projections)} projections to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for synset_id, chinese_lemma in projections:
            f.write(f"{synset_id}\t{chinese_lemma}\n")
    
    print("Done!")
    print(f"Total projections: {len(projections)}")


if __name__ == "__main__":
    # File paths
    se13_tokens_file = "se13_tokens.xlsx"
    key_file = "se13.key.txt"
    alignment_file = "alignments.txt"
    english_tokens_file = "english_tokens.txt"
    chinese_tokens_file = "chinese_tokens.txt"
    output_file = "senses.tsv"
    
    # Run projection
    project_senses(
        se13_tokens_file,
        key_file,
        alignment_file,
        english_tokens_file,
        chinese_tokens_file,
        output_file
    )