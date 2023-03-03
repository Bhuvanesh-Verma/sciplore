import argparse
import json
import sys
from collections import defaultdict

import pandas as pd
import torch
import yaml

from src.train.base_models import get_section_text, str2sections
from src.train.few_shot import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for zero shot experiments')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/fs_train.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    data_path = config_data['data_path']
    with open(data_path) as f:
        data = json.load(f)

    new_data = str2sections(data)

    labels = data['label']
    text = None

    exp_results = defaultdict(lambda: defaultdict())

    sys.stdout = open(f'logs/few_shot_exp.txt', 'w')

    for data_type in ['abstract', 'sec-text']:
        if data_type == 'abstract':
            text = [sec2text['Abstract'] for sec2text in new_data]
        elif data_type == 'sec-text':
            text = get_section_text(new_data)
        else:
            ValueError(f'Incorrect data type {data_type}')
        print(f'\nExperiment for Data Type: {data_type}')

        exp_results[data_type] = pipeline(text, labels, config_data)

        pd.DataFrame(exp_results).T.to_csv(f'experiments/few_shot_report.csv')
