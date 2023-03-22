import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.sci_dataset import SciDataset
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

    dataset = SciDataset()

    exp_results = defaultdict(lambda: defaultdict())

    sys.stdout = open(f'logs/few_shot_models_exp.txt', 'w')

    for model_name in ['sentence-transformers/paraphrase-mpnet-base-v2', 'sentence-transformers/all-mpnet-base-v2', ]:
        config_data['model_name'] = model_name
        print(model_name)
        for data_type in ['abstract','full_text', 'sec-text', 'sec-name']:
            print(f'\nExperiment for Data Type: {data_type}')
            exp_results[model_name][data_type] = pipeline(dataset, data_type, config_data)

    #pd.DataFrame(exp_results).T.to_csv(f'experiments/few_shot_report.csv')
    rows = []
    for model_type, exp_result in exp_results.items():
        for data_type, results in exp_result.items():
            row = (model_type, data_type, results['best_accuracy'], results['avg_accuracy'], results['best_f1'],
                   results['avg_f1'])
            rows.append(row)

    pd.DataFrame(rows, columns=['model_type', 'data_type', 'best_accuracy', 'avg_accuracy',
                                'best_f1', 'avg_f1']).to_csv(f'experiments/few_shot_models_report.csv')
