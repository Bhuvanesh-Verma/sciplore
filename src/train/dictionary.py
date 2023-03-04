import argparse
from collections import defaultdict, Counter

import nltk
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.data.sci_dataset import SciDataset
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic

def get_vocab(texts):
    vocab = texts.split()

    unigm = dict(Counter(vocab).most_common(100))
    unigm = {uni:c/sum(list(unigm.values())) for uni, c in unigm.items()}

    bigram_vocab = list(nltk.bigrams(texts.split()))
    bigm = dict(Counter(bigram_vocab).most_common(30))
    bigm = {bi:c/sum(list(bigm.values())) for bi, c in bigm.items()}

    trigram_vocab = list(nltk.trigrams(texts.split()))
    trigm = dict(Counter(trigram_vocab).most_common(30))
    trigm = {tri:c/sum(list(trigm.values())) for tri, c in trigm.items()}

    return {'uni':unigm, 'bi':bigm, 'tri':trigm}


def pipeline(dataset, data_type, config):
    quan = dataset.label_data['Quantitative']
    qual = dataset.label_data['Qualitative']
    mix = dataset.label_data['Qualitative and Quantitative']
    quan_vocab = get_vocab(quan)
    qual_vocab = get_vocab(qual)
    mix_vocab = get_vocab(mix)
    texts, labels = dataset.get_text_label(data_type)
    results = np.zeros((len(texts),3))
    for i, text in enumerate(texts):
        text_vocab = get_vocab(text)
        for v, vocab in text_vocab.items():
            if v not in config['type']:
                continue
            for word, f in vocab.items():
                results[i][dataset.label2idx['Quantitative']] += quan_vocab[v].get(word,0)*f
                results[i][dataset.label2idx['Qualitative']] += qual_vocab[v].get(word, 0)*f
                results[i][dataset.label2idx['Qualitative and Quantitative']] += mix_vocab[v].get(word, 0)*f

    df = pd.DataFrame(results)
    preds = [dataset.idx2label[id] for id in list(df.apply(lambda x: x.idxmax(), axis=1))]
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    print(classification_report(labels, preds))
    return {'f1_score': f1, 'accuracy': acc}




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for similarity based models')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/dict_train.yaml',
    )

    parser.add_argument(
        '-data_type',
        help='type of data to use for modelling',
        type=str, default='abstract',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    dataset = SciDataset()

    data_type = args.data_type

    stats = pipeline(dataset, data_type, config_data)
    print(stats)
