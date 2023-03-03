import argparse

import numpy as np
import yaml
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.data.sci_dataset import SciDataset


def compute_metrics(y_pred, y_test):
    return {
        "f1": f1_score(y_test, y_pred.cpu().data.numpy(), average='macro'),
        "accuracy": accuracy_score(y_test, y_pred.cpu().data.numpy()),
        "y_pred": y_pred.cpu().data.numpy()
    }


def pipeline(dataset, data_type, config):
    idx2label = dataset.idx2label
    f1s = []
    accs = []

    # Split the data into training and testing sets
    for i, (X_train, y_train, X_test, y_test) in enumerate(dataset.get_train_test_split(data_type,label_type='numeric', n_splits=5)):
        print(f'\n-------Split {i + 1}-------')

        train_dataset = {'text': X_train, 'label': y_train}
        train_dataset = Dataset.from_dict(train_dataset)

        test_dataset = {'text': X_test, 'label': y_test}
        test_dataset = Dataset.from_dict(test_dataset)

        model = SetFitModel.from_pretrained(config['model_name'])

        # Create trainer
        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            loss_class=CosineSimilarityLoss,
            metric=compute_metrics,
            batch_size=8,
            num_iterations=15,  # Number of text pairs to generate for contrastive learning
            num_epochs=2  # Number of epochs to use for contrastive learning
        )
        # Train and evaluate!
        trainer.train()
        metrics = trainer.evaluate()
        gold = [idx2label[int(l)] for l in y_test]
        preds = [idx2label[int(l)] for l in metrics['y_pred']]
        print(classification_report(gold, preds))
        accs.append(metrics['accuracy'])
        f1s.append(metrics['f1'])

    stats = {'best_accuracy': np.around(accs[np.argmax(accs)], 2),
             'avg_accuracy': np.around(np.mean(accs), 2),
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

    dataset = SciDataset()
    data_type = args.data_type

    stats = pipeline(dataset, data_type, config_data)
    print(stats)
