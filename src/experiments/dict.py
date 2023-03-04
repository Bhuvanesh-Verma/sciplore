import argparse
import sys
from collections import defaultdict

import pandas as pd
import torch
import yaml

from src.data.sci_dataset import SciDataset
from src.train.dictionary import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for zero shot experiments')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/dict_train.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    dataset = SciDataset()

    exp_results = defaultdict(lambda: defaultdict())

    sys.stdout = open(f'logs/dict_models_exp.txt', 'w')

    for type in [['uni'], ['uni', 'bi']]:
        config_data['type'] = type
        type = '-'.join(type)
        for data_type in ['abstract', 'full_text', 'sec-text', 'sec-name']:
            print(f'\nExperiment for Data Type: {data_type} with {type}')
            exp_results[type][data_type] = pipeline(dataset, data_type, config_data)

    #pd.DataFrame(exp_results).T.to_csv(f'experiments/few_shot_report.csv')
    rows = []
    for type, exp_result in exp_results.items():
        for data_type, results in exp_result.items():
            row = (type, data_type, results['f1_score'], results['accuracy'])
            rows.append(row)

    pd.DataFrame(rows, columns=['model_type', 'data_type', 'f1_score', 'accuracy']).to_csv(f'experiments/dict_models_report.csv')
