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

    annotation={}

    annotation['``'] = ""
    annotation['&amp;'] = ""
    annotation['Mr.'] = ""

    for raw in raws:
        raw = str(raw)
        if len(raw.split('-')) > 1 or len(raw.split(' '))>1:
            annotation[raw] = ""

    del annotation["Dow Jones"]
    del annotation["solicitor general"]

    sorted_keys = sorted(annotation.keys(), key=len, reverse=True)
    annotation = {key: annotation[key] for key in sorted_keys}

    body = []
    cnt = 0
    for text in raw_text:
        cnt += 1
        for key in annotation.keys():
            if cnt !=252 or key != "Costa Rica":
                text = text.replace(key, "?")
        if cnt == 47:
            text = text.replace("Dow Jones" , "?")

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
    
    cnt = 0

    with open('wsd_output.txt', 'w', encoding='utf-8') as f:
        for response, instance in zip(responses, instances):
            if instance is np.nan:
                pass
            else:
                f.write(f"{instance} {response['bnSynsetId']}\n")

main()

""""
    for response, raw in zip(responses, raws):
        cnt+=1
        print(response['text'],raw,cnt)
"""