import pandas as pd
import csv
import numpy as np
from collections import Counter
import json
import requests
from deep_translator import GoogleTranslator


def main():
    table_sentences = pd.read_excel('se13_sentences.xlsx')
    raw_text = table_sentences['raw_text'].tolist()
    translator = GoogleTranslator(source='auto', target='zh-CN')
    list = []
    with open('translation_output.txt', 'w', encoding='utf-8') as f:
        f.write('[\n')
        for index in range(len(raw_text)):
            translation = translator.translate(raw_text[index])
            dict = {}
            dict['src'] = raw_text[index]
            dict['mt'] = translation
            if index < len(raw_text)-1:
                f.write(json.dumps(dict, ensure_ascii=False) + ',\n')
            else:
                f.write(json.dumps(dict, ensure_ascii=False) + '\n')
        f.write(']\n')

main()