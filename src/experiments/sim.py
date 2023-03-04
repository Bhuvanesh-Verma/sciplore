import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.sci_dataset import SciDataset
from src.train.similarity import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for similarity experiments')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/sim_train.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    dataset = SciDataset()

    exp_results = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    sys.stdout = open(f'logs/sim_models_exp.txt', 'w')

    for model_name in ['sentence-transformers/paraphrase-mpnet-base-v2', 'sentence-transformers/all-mpnet-base-v2', ]:
        config_data['model_name'] = model_name
        print(f'{model_name}')
        for exp_type in [True, False]:
            config_data['chunk'] = exp_type
            if exp_type:
                exp_type = 'chunked'
            else:
                exp_type = 'avg'
            for data_type in ['abstract', 'full_text', 'sec-name', 'sec-text']:
                print(f'\nExperiment with {exp_type} embedding and Data Type: {data_type} ')
                exp_results[model_name][exp_type][data_type] = pipeline(dataset, data_type, config_data)

    # pd.DataFrame(exp_results).T.to_csv(f'experiments/{model_name}_report.csv')
    rows = []
    for model_name, exp_result in exp_results.items():
        for exp_type, class_data in exp_result.items():
            for data_type, results in class_data.items():
                row = (
                model_name, exp_type, data_type, np.around(results['f1_score'], 2), np.around(results['accuracy'], 2))
                rows.append(row)
    pd.DataFrame(rows, columns=['model_name', 'embed_type', 'data_type', 'f1_score', 'accuracy']).to_csv(
        f'experiments/sim_models_report.csv')
