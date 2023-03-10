import re
from os import listdir

from frame_semantic_transformer import FrameSemanticTransformer
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import json

sent_bound = 'STB'
sec_bound = 'SBA'

def get_frames(model, sents):
    frames = []
    for sent in sents:
        if len(sent) == 0:
            continue
        try:
            results = model.detect_frames(sent.strip())
        except IndexError:
            print(f'Sentence: {sent}\nIndexError')
            frames.append(sent_bound)
            continue
        for frame in results.frames:
            frames.append(frame.name.strip())
        frames.append(sent_bound)
    return frames
def create_sentence_frames(model, data):
    frame_data = defaultdict(lambda: defaultdict(list))
    for doc_id, doc_data in tqdm(data.items()):
        abstract = doc_data['abstract']
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])\s*(?![^()]*\))', abstract)
        frame_data[doc_id]['abstract'] = get_frames(model, sentences)
        for sec_name, sec_data in doc_data['text'].items():
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])\s*(?![^()]*\))', sec_data)
            frames = get_frames(model, sentences)
            frames.append(sec_bound)
            frame_data[doc_id]['text'].extend(frames)
        frame_data[doc_id]['label'] = doc_data['label']

    return frame_data


if __name__ == '__main__':
    data_path = 'data/balance_corpus.json'
    save_path = 'data/frames_dataset.json'
    with open(data_path) as f:
        data = json.load(f)
    model = FrameSemanticTransformer()
    frame_data = create_sentence_frames(model, data)

    with open(save_path, "w+") as outfile:
        json.dump(frame_data, outfile, indent=4, sort_keys=False)