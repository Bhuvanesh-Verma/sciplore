import argparse
import json
from collections import defaultdict

import torch
import yaml
from transformers import pipeline
from sklearn.metrics import f1_score, accuracy_score

from src.train.base_models import get_section_text, str2sections
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def zs_full_text(texts, labels, model_name):
    classifier = pipeline("zero-shot-classification",
                          model=model_name, device='cuda:0')
    preds = []
    for text in texts:
        text = text.split()
        chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
        chunk_results = []
        for chunk in chunks:
            result = classifier(' '.join(chunk), list(set(labels)))
            chunk_results.append(result)
        # Aggregate the predicted labels and confidence scores for each chunk
        label_scores = {}
        for chunk_result in chunk_results:
            for i, label in enumerate(chunk_result['labels']):
                if label not in label_scores:
                    label_scores[label] = 0
                label_scores[label] += chunk_result['scores'][i]
        # Choose the label with the highest score as the final predicted label for the article
        predicted_label = max(label_scores, key=label_scores.get)
        preds.append(predicted_label)
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'f1_score':f1, 'accuracy': acc}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for zero shot')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/zs_train.yaml',
    )

    parser.add_argument(
        '-data_type',
        help='type of data to use for modelling',
        type=str, default='sec-text',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    data_path = config_data['data_path']
    with open(data_path) as f:
        data = json.load(f)

    labels = data['label']
    for i, lab in enumerate(labels):
        if lab == 'Mixed':
            labels[i] = 'Qualitative and Quantitative'
    new_data = str2sections(data)

    data_type = args.data_type
    texts = None
    if data_type == 'abstract':
        texts = [sec2text['Abstract'] for sec2text in new_data]
    elif data_type == 'full_text':
        texts = data['full_text']
    elif data_type == 'sec-text':
        texts = get_section_text(new_data)
    else:
        ValueError(f'Incorrect data type {data_type}')

    zs_full_text(texts, labels, config_data["model_name"])