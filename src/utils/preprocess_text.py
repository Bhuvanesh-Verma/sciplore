import json
from collections import defaultdict
from datetime import datetime
from os import listdir
from os.path import join, isfile
from pathlib import Path

import argparse
import yaml
from tqdm import tqdm

from src.utils.data_preprocessor import Preprocessor


def break_sentence(long_sentence, max_len, min_len):
    """
    This method breaks a long sentence into small chunks based on the max_len and min_len.
    :param long_sentence: string representing a sentence
    :param max_len: integer value representing maximum length a chunk can have
    :param min_len: integer value representing minimum length a chunk should have
    :return: a string if original sentence is within given range otherwise list of string representing small chunks of
    long sentence
    """
    text = []
    tokens = long_sentence.split()
    if len(tokens) <= max_len and len(tokens) >= min_len:
        return long_sentence
    for l in range(0,len(tokens), max_len):
        start = l
        end = l + max_len
        if end > len(tokens):
            end = len(tokens)
        if end-start < min_len:
            continue
        text.append(' '.join(tokens[start:end]))
    return text
def process_scientific_article(data, preprocessor, chunk=False):
    text = []

    if not chunk:
        text = data['abstract']
        for head, content in data['text'].items():
            if content and content != '':
                text = text + ' ' + content.strip()
        text = preprocessor.preprocess_text(text)
    else:
        max_len = 510
        min_len = 10
        abstract = preprocessor.preprocess_text(data['abstract'])
        abstract = break_sentence(abstract, max_len = max_len-2, min_len=min_len)
        if isinstance(abstract, list):
            text.extend(abstract)
        elif isinstance(abstract, str):
            text.append(abstract)
        for head, content in data['text'].items():
            if content and content != '':
                content = preprocessor.preprocess_text(content)
                b = break_sentence(content, max_len = max_len-2, min_len=min_len)
                if isinstance(b,list):
                    text.extend(b)
                elif isinstance(b, str):
                    text.append(b)
                else:
                    raise TypeError(f'Invalid type:{type(b)} returned.')


    return text


def preprocess(json_data, processor_args, chunk=False):
    docs = defaultdict(list)
    preprocessor = Preprocessor(processor_args)
    print('Preprocessing docs')
    for i, data in json_data.items():
        text = process_scientific_article(data, preprocessor)
        if chunk:
            docs['title'].extend([data['title']] * len(text))
            docs['abstract'].extend([data['abstract']] * len(text))
            docs['text'].extend(text)
            docs['id'].extend([i] * len(text))
            docs['label'].extend([data['label']] * len(text))
            docs['sections'].extend([list(data['text'].keys())]*len(text))
        else:
            docs['text'].append(text)
            docs['abstract'].append(data['abstract'])
            docs['id'].append(i)  # Used for supervised learning
            docs['label'].append(data['label'])
            docs['title'].append(data['title'])
            docs['sections'].append(list(data['text'].keys()))


    # POS preprocessing
    if 'remove_pos' in processor_args:
        docs['text'] = preprocessor.pos_preprocessing(docs=docs['text'])
    return docs


if __name__ == '__main__':
    data_path = 'data/balance_corpus.json'
    save_path = 'data/base_dataset.json'
    with open(data_path) as f:
        data = json.load(f)
    args = {'remove_paran_content': True,
            'remove_pos':["ADV","PRON","CCONJ","PUNCT","PART","DET","ADP","SPACE","NUM","SYM"]}
    docs = preprocess(json_data=data,processor_args=args)

    with open(save_path, "w+") as outfile:
        json.dump(docs, outfile, indent=4, sort_keys=False)



