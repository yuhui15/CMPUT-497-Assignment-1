import pandas as pd
import csv
import numpy as np
from collections import Counter
import json
import requests

def main():
    table_sentences = pd.read_excel('se13_sentences.xlsx')
    table_tokens = pd.read_excel('se13_tokens.xlsx')

    raw_text = table_sentences['raw_text'].tolist()
    raws = table_tokens['raw_text'].tolist()

    body = []
    for text in raw_text:
        dict = {}
        dict['text'] = text
        dict['lang'] = 'EN'
        body.append(dict)
    
    url = 'http://nlp.uniroma1.it/amuse-wsd/api/model'
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }

    responses = requests.post(url, headers=headers, json=body)

    if responses.status_code == 200:
        result_data = responses.json()
        with open('amuse_results.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
    else:
        print(f"Failed status code: {responses.status_code}")
        print(f"Error information: {responses.text}")
    
    responses = responses.json()

    temp= []

    for response in responses:
        temp.extend(response['tokens'])


    responses = temp
    instances = table_tokens['instance_id'].tolist()

    index = 0
    with open('amuse_output.key', 'w', encoding='utf-8') as f:
        for instance, raw in zip(instances, raws):
            raw = str(raw)
            if instance is np.nan:
                pass
            else:
                f.write(f"{instance} ")

            if responses[index]["text"] == "'s":
                responses[index]["text"] = "s"

            if responses[index]["text"] == "''":
                responses[index]["text"] = "'"

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

            if responses[index]["text"] == "140,000":
                responses[index]["text"] = "140000"

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
            
            if responses[index]["text"] == "Libération":
                responses[index]["text"] = "Lib��ration"
            
            if responses[index]["text"] == "900,000":
                responses[index]["text"] = "900000"

            if responses[index]["text"] == "900,000":
                responses[index]["text"] = "900000"
            
            if responses[index]["text"] == "François":
                responses[index]["text"] = "Fran��ois"

            if responses[index]["text"] == "150,000":
                responses[index]["text"] = "150000"

            if responses[index]["text"] == "false":
                responses[index]["text"] = "False"
            
            if responses[index]["text"] == "true":
                responses[index]["text"] = "1"
            
            while responses[index]["text"] in raw:
                #print(responses[index]["text"] , end=" ")
                if instance is np.nan:
                    pass
                else:
                    f.write(f"{responses[index]['bnSynsetId']} ")
                index += 1
                if index >= len(responses):
                    break
            if instance is np.nan:
                pass
            else:
                f.write("\n")
            #print(raw)
main()