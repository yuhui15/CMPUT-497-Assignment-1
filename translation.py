import pandas as pd
import json
import time
from deep_translator import GoogleTranslator

def preprocess_text(text):
    """Preprocess text by cleaning whitespace and handling null values"""
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

def translate_in_batches(texts, batch_size=20):
    """Translate texts in batches with error handling"""
    translator = GoogleTranslator(source='auto', target='zh-CN')
    all_results = []
    failed_translations = []
    
    print("Starting batch translation...")
    
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        for i, text in enumerate(batch_texts):
            global_index = batch_start + i
            
            try:
                # Preprocess text
                cleaned_text = preprocess_text(text)
                original_text = str(text) if text is not None else ""
                
                if not cleaned_text:
                    print(f"Warning: Empty text at index {global_index}")
                    translation_dict = {
                        'index': global_index,
                        'src': original_text,
                        'mt': "",
                        'src_lang': 'en',
                        'dest_lang': 'zh-cn',
                        'status': 'empty'
                    }
                else:
                    # Perform translation
                    translation_result = translator.translate(cleaned_text)
                    
                    if not translation_result or translation_result.strip() == "":
                        raise Exception("Empty translation result")
                    
                    translation_dict = {
                        'index': global_index,
                        'src': original_text,
                        'mt': translation_result,
                        'src_lang': 'en',
                        'dest_lang': 'zh-cn',
                        'status': 'success'
                    }
                
                all_results.append(translation_dict)
                
                # Progress display
                if (global_index + 1) % 10 == 0 or global_index == 0:
                    print(f"Completed {global_index+1}/{len(texts)} ({((global_index+1)/len(texts)*100):.1f}%)")
                    
            except Exception as e:
                print(f"Translation failed for sentence {global_index}: {str(e)}")
                failed_translations.append(global_index)
                
                # Create error record
                error_dict = {
                    'index': global_index,
                    'src': original_text if 'original_text' in locals() else str(text),
                    'mt': f"[Translation failed: {str(e)}]",
                    'src_lang': 'en',
                    'dest_lang': 'zh-cn',
                    'status': 'failed',
                    'error': str(e)
                }
                
                all_results.append(error_dict)
        
        # Delay between batches to avoid rate limiting
        if batch_start + batch_size < len(texts):
            time.sleep(1)
    
    return all_results, failed_translations

def main():
    print("Starting data loading...")
    table_sentences = pd.read_excel('se13_sentences.xlsx')
    raw_text = table_sentences['raw_text'].tolist()
    print(f"Loaded {len(raw_text)} sentences")
    
    # Translate in batches
    translations, failed_translations = translate_in_batches(raw_text, batch_size=25)
    
    # Save results to JSON file
    print("Saving results to file...")
    with open('translation_output.txt', 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i, translation_dict in enumerate(translations):
            if i < len(translations) - 1:
                f.write(json.dumps(translation_dict, ensure_ascii=False, indent=2) + ',\n')
            else:
                f.write(json.dumps(translation_dict, ensure_ascii=False, indent=2) + '\n')
        f.write(']\n')
    
    # Output summary statistics
    successful = len([t for t in translations if t.get('status') == 'success'])
    empty = len([t for t in translations if t.get('status') == 'empty'])
    failed = len([t for t in translations if t.get('status') == 'failed'])
    
    print(f"\n=== Translation Complete ===")
    print(f"Total sentences: {len(raw_text)}")
    print(f"Successful translations: {successful}")
    print(f"Empty text entries: {empty}")
    print(f"Failed translations: {failed}")
    print(f"Success rate: {(successful/len(raw_text)*100):.1f}%")
    
    if failed_translations:
        print(f"Failed sentence indices (first 10): {failed_translations[:10]}")
    
    print(f"Results saved to translation_output.txt")
    
    # Save statistics report
    with open('translation_stats.txt', 'w', encoding='utf-8') as f:
        f.write(f"Translation Statistics\n")
        f.write(f"===================\n")
        f.write(f"Total sentences: {len(raw_text)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Empty text: {empty}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success rate: {(successful/len(raw_text)*100):.1f}%\n")
        if failed_translations:
            f.write(f"Failed indices: {failed_translations}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTranslation interrupted by user")
    except Exception as e:
        print(f"Program execution error: {str(e)}")