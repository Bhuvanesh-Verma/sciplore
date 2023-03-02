import argparse
import json
from collections import defaultdict

import torch
import yaml
from transformers import pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.train.base_models import get_section_text, str2sections
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(nli_model, tokenizer, sequence, labels):
    # pose sequence as a NLI premise and label as a hypothesis
    nli_model = nli_model.to(device)
    premise = sequence
    #hypothesis = {label: f'The given text follows {label} approach.' for label in labels}
    hypothesis = { 'Quantitative': """The approach in the given text is primarily numerical using statistical methods and relies on measurements and calculations.""",
                   'Qualitative': """The approach in the given text is primarily focused on understanding subjective experiences and social phenomena using interviews, 
                   observations, or case studies.""",
                   }
    results = defaultdict(list)
    for label, h in hypothesis.items():

        # run through model pre-trained on MNLI
        x = tokenizer.encode(premise, h, return_tensors='pt',truncation='longest_first')
        logits = nli_model(x.to(device))[0]

        # we throw away "neutral" (dim 1) and take the probability of
        # "entailment" (2) as the probability of the label being true
        if torch.argmax(logits[0]) == 1:
            continue
        entail_contradiction_logits = logits[:, [0, 2]]
        results['labels'].append(label)
        probs = entail_contradiction_logits.softmax(dim=1)

        results['scores'].append(float(probs[0][1]))

    return results
def zs_manual(texts, labels, model_name, max_token):
    nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preds = []
    indices_to_remove = []
    for k, text in enumerate(texts):
        text = text.split()
        chunks = [text[i:i + max_token] for i in range(0, len(text), max_token)]
        chunk_results = []
        for chunk in chunks:
            result = predict(nli_model, tokenizer, ' '.join(chunk), list(set(labels)))
            chunk_results.append(result)

        label_scores = {}
        for chunk_result in chunk_results:
            for i, label in enumerate(chunk_result['labels']):
                if label not in label_scores:
                    label_scores[label] = 0
                label_scores[label] += chunk_result['scores'][i]

        if len(label_scores) == 0:
            indices_to_remove.append(k)
            print(f'Document {k} skipped')
            continue
        label_scores = {k:label/sum(list(label_scores.values())) for k, label in label_scores.items()}

        if any(0.33 < value < 0.67 for value in label_scores.values()):
            predicted_label = 'Qualitative and Quantitative'
        else:
            predicted_label = max(label_scores, key=label_scores.get)
        preds.append(predicted_label)
    labels = [label for i, label in enumerate(labels) if i not in indices_to_remove]
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    print(classification_report(labels, preds))
    return {'f1_score':f1, 'accuracy': acc}

def zs_auto(texts, labels, model_name, max_token):
    classifier = pipeline("zero-shot-classification",
                          model=model_name, device='cuda:0')
    preds = []
    for text in texts:
        text = text.split()
        chunks = [text[i:i + max_token] for i in range(0, len(text), max_token)]
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
    print(classification_report(labels, preds))
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
        type=str, default='abstract',
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

    #zs_full_text(texts, labels, config_data["model_name"], config_data['max_token'])
    print(zs_manual(texts, labels, config_data["model_name"], config_data['max_token']))