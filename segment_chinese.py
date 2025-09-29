import jieba
import codecs
import os

# --- Configuration ---
INPUT_FILE_NAME = "translated_sentences.txt"
OUTPUT_FILE_NAME = "segmented_translated_sentences_in_chinese.txt"
ENCODING = 'utf-8' # Crucial for handling Chinese characters

def segment_chinese_file(input_path: str, output_path: str):
    """
    Reads a file of Chinese sentences, segments each line using Jieba,
    and writes the segmented sentences to a new output file.
    """
    print(f"Starting segmentation of: {input_path}")
    
    try:
        # 1. Read the input file (using codecs for robust UTF-8 handling)
        with codecs.open(input_path, 'r', encoding=ENCODING) as infile:
            chinese_corpus = [line.strip() for line in infile.readlines()]
            # Filter out any completely empty lines that might result from stripping
            chinese_corpus = [line for line in chinese_corpus if line]

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # 2. Segment each sentence and store the results
    segmented_corpus = []
    print(f"Processing {len(chinese_corpus)} sentences...")
    
    # Use jieba.cut for segmentation
    for sentence in chinese_corpus:
        # jieba.cut returns a generator of words, which we join with a space
        segmented_line = " ".join(jieba.cut(sentence))
        segmented_corpus.append(segmented_line)

    # 3. Write the segmented corpus to the new output file
    try:
        # Open output file in write mode ('w') with UTF-8 encoding
        with codecs.open(output_path, 'w', encoding=ENCODING) as outfile:
            for segmented_sentence in segmented_corpus:
                # Write each segmented sentence followed by a newline
                outfile.write(segmented_sentence + '\n')
        
        print(f"Segmentation complete! Output saved to: {output_path}")

    except Exception as e:
        print(f"Error writing output file: {e}")


# --- Execution ---
if __name__ == "__main__":
    # Ensure the input file exists (You can place the full content of 
    # translated_sentences.txt into a file with this name)
    
    # Simple check to confirm the input file is in the same directory
    if os.path.exists(INPUT_FILE_NAME):
        segment_chinese_file(INPUT_FILE_NAME, OUTPUT_FILE_NAME)
    else:
        print(f"Please ensure '{INPUT_FILE_NAME}' is in the same directory as this script.")