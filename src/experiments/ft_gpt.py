import argparse
import json
import re
import sys
from collections import defaultdict, Counter

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import  GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.sci_dataset import SciDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classifier(model, tokenizer, text):
    # create prompt (in compliance with the one used during training)
    prompt = f'<startoftext>Sentence: {text}\nResearch Design:'
    # generate tokens
    generated = tokenizer(f"{prompt}", return_tensors="pt")
    input_id = torch.tensor(generated['input_ids']).cuda()
    attn_mask = torch.tensor(generated['attention_mask']).cuda()
    # perform prediction
    sample_outputs = model.generate(input_id, do_sample=False, top_k=50, max_length=512,
                                    top_p=0.90, temperature=0, num_return_sequences=0,
                                    pad_token_id=tokenizer.eos_token_id)
    # decode the predicted tokens into texts

    pred_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    # extract the predicted sentiment
    try:
        pred_sentiment = re.findall("Research Design: (.*)", pred_text)[-1]
        pred_sentiment = pred_sentiment.split("<endoftext>")[0]
    except:
        pred_sentiment = "None"

    return pred_sentiment


def test_pipeline(dataset, data_type, model_name):
    texts, labels = dataset.get_text_label(data_type)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<startoftext>', eos_token='<endoftext>',
                                              pad_token='<pad>')
    model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    model.resize_token_embeddings(len(tokenizer))
    preds = []
    for text in tqdm(texts):
        text = text.split()
        chunks = [text[i:i + 32] for i in range(0, len(text), 32)]
        chunk_results = []
        for chunk in tqdm(chunks):

            result = classifier(model, tokenizer,' '.join(chunk))
            chunk_results.append(result)
        # Aggregate the predicted labels and confidence scores for each chunk
        label_scores = dict(Counter(chunk_results))
        label_scores = {u: label / sum(list(label_scores.values())) for u, label in label_scores.items()}
        predicted_label = max(label_scores, key=label_scores.get)
        if model_name == 'models/3_labels_32':
            if predicted_label == 'Mixed':
                predicted_label = 'Qualitative and Quantitative'
        else:
            if any(0.33 < value < 0.67 for value in label_scores.values()):
                predicted_label = 'Qualitative and Quantitative'
            else:
                predicted_label = max(label_scores, key=label_scores.get)
        preds.append(predicted_label)
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    print(classification_report(labels, preds))
    return {'f1_score': f1, 'accuracy': acc}

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

    dataset = SciDataset()

    #test_pipeline(dataset, args.data_type, 'models/gpt_2labels_32')

    exp_results = defaultdict(lambda: defaultdict())

    sys.stdout = open(f'logs/ft_gpt_models_exp.txt', 'w')

    for model_name in [ 'models/2_labels_32', 'models/3_labels_32']:
        config_data['model_name'] = model_name
        print(model_name)
        for data_type in ['abstract', 'full_text', 'sec-text', 'sec-name']:
            print(f'\nExperiment for Data Type: {data_type}')
            exp_results[model_name][data_type] = test_pipeline(dataset, data_type, model_name)

    # pd.DataFrame(exp_results).T.to_csv(f'experiments/few_shot_report.csv')
    rows = []
    for model_type, exp_result in exp_results.items():
        for data_type, results in exp_result.items():
            row = (model_type, data_type, results['accuracy'], results['f1_score'])
            rows.append(row)

    pd.DataFrame(rows, columns=['model_type', 'data_type', 'best_accuracy','best_f1']).to_csv(f'experiments/ft_gpt_models_report.csv')


