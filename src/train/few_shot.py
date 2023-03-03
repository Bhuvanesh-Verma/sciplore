import argparse
import json
from collections import defaultdict

import datasets
import evaluate
import numpy as np
import torch
import yaml
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, classification_report
from src.train.base_models import str2sections, get_section_text

def compute_metrics(y_pred, y_test):
    return {
        "f1": f1_score(y_test,y_pred.cpu().data.numpy(), average='macro'),
        "accuracy": accuracy_score(y_test, y_pred.cpu().data.numpy()),
        "y_pred": y_pred.cpu().data.numpy()
    }
def pipeline(text, labels, config):
    label2idx = {label:i for i, label in enumerate(set(labels))}
    idx2label = {i:label for label, i in label2idx.items()}

    n_splits = 5  # number of splits to create
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    f1s = []
    accs = []
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2",
                                        use_differentiable_head=True,
                                        head_params={"out_features": 3},
                                        )
    # Split the data into training and testing sets
    for i, (train_index, test_index) in enumerate(sss.split(text, labels)):
        print(f'\n-------Split {i+1}-------')
        X_train, X_test = [text[j] for j in train_index], [text[j] for j in test_index]
        y_train, y_test = torch.tensor([label2idx[labels[j]] for j in train_index]), torch.tensor([label2idx[labels[j]] for j in test_index])
        train_dataset = {'text': X_train, 'label': y_train}

        train_dataset = Dataset.from_dict(train_dataset)

        test_dataset = {'text': X_test, 'label': y_test}
        test_dataset = Dataset.from_dict(test_dataset)


        # Create trainer
        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            loss_class=CosineSimilarityLoss,
            metric=compute_metrics,
            batch_size=8,
            num_iterations=15,  # Number of text pairs to generate for contrastive learning
            num_epochs=3  # Number of epochs to use for contrastive learning
        )
        # Train and evaluate!
        trainer.train()
        metrics = trainer.evaluate()
        gold = [idx2label[int(l)] for l in y_test]
        preds = [idx2label[int(l)] for l in metrics['y_pred']]
        print(classification_report(gold, preds))
        accs.append(metrics['accuracy'])
        f1s.append(metrics['f1'])


    stats = {'best_accuracy': np.around(accs[np.argmax(accs)],2),
             'avg_accuracy': np.around(np.mean(accs),2),
             'best_f1': np.around(accs[np.argmax(f1s)], 2),
             'avg_f1': np.around(np.mean(f1s), 2),
             }
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training base models')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/fs_train.yaml',
    )

    parser.add_argument(
        '-data_type',
        help='type of data to use for modelling',
        type=str, default='sec-name',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    data_path = config_data['data_path']
    with open(data_path) as f:
        data = json.load(f)

    new_data = str2sections(data)
    data_type = args.data_type
    labels = data['label']
    text = None
    if data_type == 'abstract':
        text = [sec2text['Abstract'] for sec2text in new_data]
    elif data_type == 'full_text':
        text = data['full_text']
    elif data_type == 'sec-name':
        text = [' '.join(secs) for secs in data['sections']]
    elif data_type == 'sec-text':
        text = get_section_text(new_data)
    else:
        ValueError(f'Incorrect data type {data_type}')

    stats = pipeline(text, labels, config_data)
    print(stats)