import argparse
import json
import sys
from collections import defaultdict

import pandas as pd
import yaml

from src.train.frames import pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for frame experiments')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/frame_train.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)
    with open(config_data['data_path']) as f:
        dataset = json.load(f)
    exp_results = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    sys.stdout = open(f'logs/frame_models_exp.txt', 'w')
    for model_type in ['svm', 'knn', 'bayes']:
        config_data['model_type'] = model_type
        config_data['params'] = config_data[model_type]
        print(f'{model_type}')
        for feat_type in ['tfidf', 'bow']:
            config_data['feat_type'] = feat_type
            for data_type in ['abstract', 'full_text', 'sec-text']:
                print(f'\nExperiment for Data Type: {data_type} and Feature type: {feat_type}')
                exp_results[model_type][feat_type][data_type] = pipeline(dataset, data_type, config_data)

    rows = []
    for model_type, exp_result in exp_results.items():
        for feat_type, class_data in exp_result.items():
            for data_type, results in class_data.items():
                row = (model_type, feat_type, data_type, results['best_f1'], results['avg_f1'],
                       results['best_accuracy'],results['avg_accuracy'],results['best_params'])
                rows.append(row)
    pd.DataFrame(rows, columns=['model_type','feature_type', 'data_type', 'best_f1', 'avg_f1',
                                'best_accuracy', 'avg_accuracy',
                                'best_params']).to_csv(f'experiments/frame_models_report.csv')