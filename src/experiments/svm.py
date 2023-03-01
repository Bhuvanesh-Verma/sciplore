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
    exp_results = defaultdict()
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
        print(f'\nExperiment for {data_type}')
        exp_results[data_type] = pipeline(text, labels, config_data)

    pd.DataFrame(exp_results).T.to_csv('experiments/svm_report.csv')