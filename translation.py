import pandas as pd
from deep_translator import GoogleTranslator
from comet import download_model, load_from_checkpoint

def preprocess_text(text):
    """Preprocess text by cleaning whitespace and handling null values"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    text = ' '.join(text.split())
    return text

def main():
    # Load data
    print("Loading data...")
    table_sentences = pd.read_excel('se13_sentences.xlsx')
    raw_text = table_sentences['raw_text'].tolist()
    print(f"Loaded {len(raw_text)} sentences\n")
    
    # Initialize translator
    translator = GoogleTranslator(source='auto', target='zh-CN')
    
    # Translate sentences
    print("Translating sentences...")
    translations = []
    translation_pairs = []
    
    for i, text in enumerate(raw_text):
        cleaned_text = preprocess_text(text)
        original_text = str(text) if text is not None else ""
        
        if cleaned_text:
            try:
                translation_result = translator.translate(cleaned_text)
                translations.append(translation_result)
                translation_pairs.append({'src': original_text, 'mt': translation_result})
            except Exception as e:
                print(f"Translation failed for sentence {i}: {str(e)}")
                translations.append("")
                translation_pairs.append({'src': original_text, 'mt': ""})
        else:
            translations.append("")
            translation_pairs.append({'src': original_text, 'mt': ""})
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(raw_text)}")
    
    # Save translations to file
    print("\nSaving translations...")
    with open('translations.txt', 'w', encoding='utf-8') as f:
        for trans in translations:
            f.write(trans + '\n')
    
    # Evaluate with CometKiwi
    print("\nEvaluating with CometKiwi...")
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    model_output = model.predict(translation_pairs, batch_size=8, gpus=1)
    
    # Save scores to file
    with open('translation_scores.txt', 'w', encoding='utf-8') as f:
        for score in model_output.scores:
            f.write(f"{score}\n")
    
    print(f"\nSystem Score: {model_output.system_score}")
    print("Files created:")
    print("  - translations.txt")
    print("  - translation_scores.txt")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Program execution error: {str(e)}")