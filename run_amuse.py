import pandas as pd
import csv
import numpy as np
from collections import Counter
import json
import requests

def main():
    # Load data from Excel files
    table_sentences = pd.read_excel('se13_sentences.xlsx')
    table_tokens = pd.read_excel('se13_tokens.xlsx')

    # Extract raw text from both tables
    raw_text = table_sentences['raw_text'].tolist()
    raws = table_tokens['raw_text'].tolist()

    # Prepare request body for AMUSE API
    body = []
    for text in raw_text:
        dict = {}
        dict['text'] = text
        dict['lang'] = 'EN'
        body.append(dict)
    
    # Configure API endpoint and headers
    url = 'http://nlp.uniroma1.it/amuse-wsd/api/model'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    # Send POST request to AMUSE WSD API
    responses = requests.post(url, headers=headers, json=body)

    # Check response status and save results
    if responses.status_code == 200:
        result_data = responses.json()
        with open('amuse_results.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
    else:
        print(f"Failed status code: {responses.status_code}")
        print(f"Error information: {responses.text}")
    
    # Parse JSON response
    responses = responses.json()

    # Flatten token list from all sentences
    temp = []
    for response in responses:
        temp.extend(response['tokens'])

    responses = temp
    instances = table_tokens['instance_id'].tolist()

    # Align tokens and write output
    index = 0
    with open('amuse_output.key', 'w', encoding='utf-8') as f:
        for instance, raw in zip(instances, raws):
            raw = str(raw)
            
            # Write instance ID if present
            if instance is np.nan:
                pass
            else:
                f.write(f"{instance} ")

            # Apply hardcoded corrections for tokenization mismatches
            # Handle contractions
            if responses[index]["text"] == "'s":
                responses[index]["text"] = "s"
            
            if responses[index]["text"] == "''":
                responses[index]["text"] = "'"
            
            # Handle number formatting
            if responses[index]["text"] == "13,000":
                responses[index]["text"] = "13000"
            
            if responses[index]["text"] == "'":
                responses[index]["text"] = ""

            if responses[index]["text"] == "'re":
                responses[index]["text"] = "re"
            
            if responses[index]["text"] == "471.50":
                responses[index]["text"] = "471.5"

            if responses[index]["text"] == "3.540":
                responses[index]["text"] = "3.54"

            if responses[index]["text"] == "140,000":
                responses[index]["text"] = "140000"

            # Handle special characters in names
            if responses[index]["text"] == "Latinobarómetro":
                responses[index]["text"] = "Latinobar��metro"

            if responses[index]["text"] == "Inácio":
                responses[index]["text"] = "In��cio"

            if responses[index]["text"] == "Chávez":
                responses[index]["text"] = "Ch��vez"

            if responses[index]["text"] == "Piqué":
                responses[index]["text"] = "Piqu��"

            if responses[index]["text"] == "Libération":
                responses[index]["text"] = "Lib��ration"
            
            if responses[index]["text"] == "900,000":
                responses[index]["text"] = "900000"
            
            if responses[index]["text"] == "François":
                responses[index]["text"] = "Fran��ois"

            if responses[index]["text"] == "150,000":
                responses[index]["text"] = "150000"

            # Handle boolean values
            if responses[index]["text"] == "false":
                responses[index]["text"] = "False"
            
            if responses[index]["text"] == "true":
                responses[index]["text"] = "1"
            
            # Match tokens using character-level comparison and write synset IDs
            while responses[index]["text"] in raw:
                if instance is np.nan:
                    pass
                else:
                    f.write(f"{responses[index]['bnSynsetId']} ")
                index += 1
                if index >= len(responses):
                    break
            
            # Write newline after each instance
            if instance is np.nan:
                pass
            else:
                f.write("\n")

main()