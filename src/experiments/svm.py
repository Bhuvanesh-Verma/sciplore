import argparse
import json
import logging
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

    parser.add_argument(
        '-data_type',
        help='type of data to use for modelling',
        type=str, default='abstract',
    )

    args, remaining_args = parser.parse_known_args()
    sys.stdout = open('logs/svm_exp.txt', 'w')
    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    data_path = config_data['data_path']
    with open(data_path) as f:
        data = json.load(f)
    data_type = args.data_type
    labels = data['label']
    text = None
    exp_results = defaultdict(lambda: defaultdict())
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
                                'best_params']).to_csv('experiments/svm_report.csv')
    #pd.DataFrame(exp_results).T.to_csv('experiments/svm_report.csv')