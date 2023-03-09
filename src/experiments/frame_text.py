import argparse
import json
import sys
from collections import defaultdict

import pandas as pd
import yaml

from src.train.frame_text import pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training frame and text comparison models')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/frame_text_train.yaml',
    )

    parser.add_argument(
        '-data_type',
        help='type of data to use for modelling',
        type=str, default='full_text',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)


    with open(config_data['data_path']) as f:
        dataset = json.load(f)

    exp_results = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    sys.stdout = open(f'logs/frame_text_models_exp.txt', 'w')

    for model_name in ['sentence-transformers/paraphrase-mpnet-base-v2', 'sentence-transformers/all-mpnet-base-v2', ]:
        config_data['model_name'] = model_name
        print(model_name)
        for exp_type in ['text', 'frame']:
            config_data['exp_type'] = exp_type
            for data_type in ['abstract', 'full_text', 'sec_text']:
                print(f'\nExperiment for Data Type: {data_type} and Feature type: {exp_type}')
                exp_results[model_name][exp_type][data_type] = pipeline(dataset, data_type, config_data)

    rows = []
    for model_name, exp_result in exp_results.items():
        for exp_type, class_data in exp_result.items():
            for data_type, results in class_data.items():
                row = (model_name, exp_type, data_type, results['best_f1'], results['avg_f1'],
                       results['best_accuracy'], results['avg_accuracy'])
                rows.append(row)
    pd.DataFrame(rows, columns=['model_name', 'exp_type', 'data_type', 'best_f1', 'avg_f1',
                                'best_accuracy', 'avg_accuracy']).to_csv(f'experiments/frame_text_models_report.csv')
