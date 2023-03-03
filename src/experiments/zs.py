import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.sci_dataset import SciDataset
from src.train.zero_shot import zs_pipeline

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


    dataset = SciDataset()
    exp_results = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    sys.stdout = open(f'logs/zero_shot_models_exp.txt', 'w')

    for model_name in ['joeddav/xlm-roberta-large-xnli', 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli', 'facebook/bart-large-mnli']:
        config_data['model_name'] = model_name
        print(f'{model_name}')
        for exp_type in ['auto', 'manual']:
            config_data['exp_name'] = exp_type
            for data_type in ['abstract', 'full_text','sec-name', 'sec-text']:
                print(f'\nExperiment Type: {exp_type} for Data Type: {data_type} ')
                exp_results[model_name][exp_type][data_type] = zs_pipeline(dataset, data_type, config_data)

    # pd.DataFrame(exp_results).T.to_csv(f'experiments/{model_name}_report.csv')
    rows = []
    for model_name, exp_result in exp_results.items():
        for exp_type, class_data in exp_result.items():
            for data_type, results in class_data.items():
                row = (model_name, exp_type, data_type, np.around(results['f1_score'],2), np.around(results['accuracy'],2))
                rows.append(row)
    pd.DataFrame(rows, columns=['model_name','experiment_type', 'data_type', 'f1_score', 'accuracy']).to_csv(
        f'experiments/zero_shot_models_report.csv')
