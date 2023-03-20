import argparse
import json
from collections import defaultdict

import torch
import yaml
from transformers import pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.sci_dataset import SciDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(nli_model, tokenizer, sequence, labels):
    # pose sequence as a NLI premise and label as a hypothesis
    nli_model = nli_model.to(device)
    premise = sequence
    #hypothesis = {label: f'The given text follows {label} approach.' for label in labels}
    hypothesis = {
        'Quantitative': """Given text is primarily numerical using statistical methods and relies on measurements and calculations.""",
        'Qualitative': """Given text is primarily focused on understanding subjective experiences and social phenomena using interviews, 
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
        label_scores = {u:label/sum(list(label_scores.values())) for u, label in label_scores.items()}

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

def zs_pipeline(dataset, data_type, config_data):
    text, labels = dataset.get_text_label(data_type)
    if config_data["exp_type"] == 'auto':
        return zs_auto(text, labels, config_data["model_name"], config_data['max_token'])
    elif config_data["exp_type"] == 'manual':
        return zs_manual(text, labels, config_data["model_name"], config_data['max_token'])
    else:
        ValueError(f'Incorrect experiment type {config_data["exp_type"]}')

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
        type=str, default='full_text',
    )


    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    dataset = SciDataset()

    print(zs_pipeline(dataset, args.data_type, config_data))


