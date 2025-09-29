import json

def separate_translations(input_file_path, source_output_path, translation_output_path):
    """
    Reads a JSON file with source and translated sentences, and writes them
    into two separate text files.

    Args:
        input_file_path (str): The path to the input JSON file.
        source_output_path (str): The path for the output source text file.
        translation_output_path (str): The path for the output translated text file.
    """
    try:
        # Open the original file and load the JSON data
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Open the two new files for writing the separated sentences
        with open(source_output_path, 'w', encoding='utf-8') as source_file, \
             open(translation_output_path, 'w', encoding='utf-8') as mt_file:

            # Loop through each dictionary in the list
            for item in data:
                # Get the source and translation, defaulting to an empty string if a key is missing
                source_sentence = item.get('src', '')
                translated_sentence = item.get('mt', '')

                # Write each sentence to its respective file, followed by a newline character
                source_file.write(source_sentence + '\n')
                mt_file.write(translated_sentence + '\n')
        
        print(f"Successfully created '{source_output_path}' and '{translation_output_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_file_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- --- ---
# How to use the script:
# 1. Save the code above as a Python file (e.g., `process_translations.py`).
# 2. Place the file in the same directory as your `translation_output.txt` file.
# 3. Run the script from your terminal.

# Define the file paths
input_file = 'translation_output.txt'
source_output_file = 'source_sentences.txt'
translation_output_file = 'translated_sentences.txt'

# Execute the function
separate_translations(input_file, source_output_file, translation_output_file)