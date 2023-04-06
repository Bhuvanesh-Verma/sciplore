import argparse
import json
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from src.train.gnn import gnn_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for multi label classification training')

    parser.add_argument(
        '-config',
        help='Path to data config file',
        type=str, default='configs/gnn_train.yaml',
    )


    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)



    with open("data/frame_text_dataset.json", "r") as fd:
        chunk_data = json.load(fd)

    exp_results = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    sys.stdout = open(f'logs/gnn_models_exp.txt', 'w')

    for model_name in ['sentence-transformers/paraphrase-mpnet-base-v2', 'sentence-transformers/all-mpnet-base-v2', ]:
        for data_type in ['abstract', 'sec_name', 'sec_text', 'full_text']:
            if model_name=='sentence-transformers/all-mpnet-base-v2':
                with open(f'data/trans_feat_matrix_{data_type}.pkl', 'rb') as f:
                    data = pickle.load(f)
            if model_name=='sentence-transformers/paraphrase-mpnet-base-v2':
                with open(f'data/trans_feat_matrix_{data_type}_v2.pkl', 'rb') as f:
                    data = pickle.load(f)
            print(model_name)
            for feat_type in ['struc','cat','emb', 'struc+cat', 'struc+emb','cat+emb', 'all']:
                print(f'\nExperiment for Feature Type: {feat_type} and data type: {data_type}')
                config['data']['feat_type'] = feat_type
                exp_results[model_name][data_type][feat_type] = gnn_pipeline(data, chunk_data, config)

    # pd.DataFrame(exp_results).T.to_csv(f'experiments/few_shot_report.csv')
    rows = []
    for model_type, exp_result in exp_results.items():
        for data_type, results in exp_result.items():
            for feat_type, res in results.items():
                row = (model_type, data_type, feat_type, np.around(res['best_val_f1'],2), np.around(res['best_val_acc'],2), np.around(res['avg_val_f1'],2),
                       np.around(res['avg_val_acc'],2))
                rows.append(row)

    pd.DataFrame(rows, columns=['model_type', 'data_type', 'feat_type' ,'best_f1', 'best_accuracy','avg_f1', 'avg_accuracy'
                                 ]).to_csv(f'experiments/gnn_models_report.csv')

