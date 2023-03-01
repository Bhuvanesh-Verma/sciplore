import argparse
import json
import sys
from collections import defaultdict

import pandas as pd
import yaml

from src.train.base_models import pipeline, get_section_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training base models')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/base_train.yaml',
    )

    args, remaining_args = parser.parse_known_args()


    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    model_type = config_data['model_type']
    data_path = config_data['data_path']

    with open(data_path) as f:
        data = json.load(f)

    labels = data['label']
    text = None
    exp_results = defaultdict(lambda: defaultdict())

    sys.stdout = open(f'logs/{model_type}_exp.txt', 'w')

    for feat_type in ['tfidf', 'bow']:
        config_data['feat_type'] = feat_type
        for data_type in ['abstract', 'full_text', 'sec-name', 'sec-text']:
            if data_type == 'abstract':
                text = data['abstract']
            elif data_type == 'full_text':
                text = data['text']
            elif data_type == 'sec-name':
                text = [' '.join(secs) for secs in data['sections']]
            elif data_type == 'sec-text':
                text, labels = get_section_text()
            else:
                ValueError(f'Incorrect data type {data_type}')
            print(f'\nExperiment for Data Type: {data_type} and Feature type: {feat_type}')
            exp_results[feat_type][data_type] = pipeline(text, labels, config_data)

    rows = []
    for feat_type, class_data in exp_results.items():
        for data_type, results in class_data.items():
            row = (feat_type, data_type, results['best_f1'], results['avg_f1'],
                   results['best_accuracy'],results['avg_accuracy'],results['best_params'])
            rows.append(row)
    pd.DataFrame(rows, columns=['feature_type', 'data_type', 'best_f1', 'avg_f1',
                                'best_accuracy', 'avg_accuracy',
                                'best_params']).to_csv(f'experiments/{model_type}_report.csv')