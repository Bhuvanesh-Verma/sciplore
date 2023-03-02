import argparse
import json
import sys
from collections import defaultdict

import pandas as pd
import torch
import yaml

from src.train.base_models import get_section_text, str2sections
from src.train.zero_shot import zs_auto, zs_manual

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for zero shot experiments')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/zs_train.yaml',
    )

    args, remaining_args = parser.parse_known_args()


    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    model_name = config_data['model_name'].split('/')[1]
    data_path = config_data['data_path']

    with open(data_path) as f:
        data = json.load(f)

    new_data = str2sections(data)
    labels = data['label']
    for i, lab in enumerate(labels):
        if lab == 'Mixed':
            labels[i] = 'Qualitative and Quantitative'
    text = None
    exp_results = defaultdict(lambda: defaultdict())

    sys.stdout = open(f'logs/{model_name}_exp.txt', 'w')

    for exp_type in ['auto', 'manual']:
        for data_type in ['abstract', 'full_text', 'sec-text']:
            if data_type == 'abstract':
                text = [sec2text['Abstract'] for sec2text in new_data]
            elif data_type == 'full_text':
                text = data['full_text']
            elif data_type == 'sec-text':
                text = get_section_text(new_data)
            else:
                ValueError(f'Incorrect data type {data_type}')
            print(f'\nExperiment for Data Type: {data_type}')
            if exp_type == 'auto':
                exp_results[exp_type][data_type] = zs_auto(text, labels, config_data["model_name"], config_data['max_token'])
            elif exp_type == 'manual':
                exp_results[exp_type][data_type] = zs_manual(text, labels, config_data["model_name"],
                                                           config_data['max_token'])
            else:
                ValueError(f'Incorrect data type {exp_type}')

        #pd.DataFrame(exp_results).T.to_csv(f'experiments/{model_name}_report.csv')
        rows = []
        for exp_type, class_data in exp_results.items():
            for data_type, results in class_data.items():
                row = (exp_type, data_type, results['f1_score'],results['accuracy'])
                rows.append(row)
        pd.DataFrame(rows, columns=['experiment_type', 'data_type', 'f1_score', 'accuracy']).to_csv(f'experiments/{model_name}_report.csv')
