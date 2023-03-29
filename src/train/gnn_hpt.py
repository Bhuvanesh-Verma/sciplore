import argparse
import json
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import wandb
import yaml
from torch import nn, optim
from torch_geometric.nn import GAT
from tqdm import tqdm

from src.train.gnn import train, evaluate
from src.utils.graph_dataset import get_transductive_data

with open('configs/gnn_sweep.yaml') as file:
    sweep_config = yaml.safe_load(file)

with open('configs/gnn_train.yaml') as file:
    config = yaml.safe_load(file)


def main():

    wandb.init(entity=config['wandb']['entity'])

    with open('data/trans_feat_matrix.pkl', 'rb') as f:
        data = pickle.load(f)

    with open("data/frame_text_dataset.json", "r") as fd:
        chunk_data = json.load(fd)

    #Transductive
    feature_matrix = torch.tensor(data['feat'])
    G = data['graph']
    node_labels = defaultdict()

    for doc_id, doc_data in chunk_data.items():
        node_labels[doc_id] = doc_data['label']

    node_labels = {k: v.lower() for k, v in node_labels.items()}
    labels = sorted([lab.lower() for lab in list(set(list(node_labels.values())))])
    label2id = {label: i for i, label in enumerate(labels)}

    model_data = get_transductive_data(G, feature_matrix, node_labels, label2id)

    # Set the seed for the random number generator
    SEED = config['train']['seed']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = config['train']['device']

    overall_summary = defaultdict()
    best_train_f1s = []
    best_val_f1s = []
    best_train_accs = []
    best_val_accs = []
    for split, split_data in model_data.items():
        metadata = split_data['metadata']
        train_indices = metadata['train']
        val_indices = metadata['val']
        test_indices = metadata['test']
        data = split_data['data']
        label2id = metadata['label2id']

        num_features = data.x.shape[1]
        num_classes = len(label2id)

        if wandb.config.num_hidden < wandb.config.num_heads != 0:
            num_head = wandb.config.num_hidden/2
        else:
            num_head = wandb.config.num_heads

        # Create the GAT model
        model = GAT(in_channels=num_features, hidden_channels=wandb.config.num_hidden,
                    out_channels=num_classes,
                    heads=num_head, num_layers=wandb.config.num_layers, dropout=wandb.config.dropout, dtype=torch.float32)
        model.to(device)
        # Define a loss function and an optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

        # Early Stopping
        best_val_loss = float('inf')
        best_val_f1 = 0
        patience = wandb.config.patience
        no_improvement_epochs = 0

        # Iterate over the training data
        train_f1s = []
        train_accs = []
        val_f1s = []
        val_accs = []

        # Model info for each epoch
        model_info = []
        summary = defaultdict()
        for epoch in tqdm(range(wandb.config.epochs)):
            train_loss, train_mirco_f1, train_accuracy = train(model, loss_fn, optimizer, data,train_indices, device)
            train_f1s.append(train_mirco_f1)
            train_accs.append(train_accuracy)
            val_loss, val_mirco_f1, val_accuracy = evaluate(model, loss_fn, data, val_indices, device)
            val_f1s.append(val_mirco_f1)
            val_accs.append(val_accuracy)
            if val_mirco_f1 > best_val_f1:
                best_val_f1 = val_mirco_f1
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_macro_f1": train_mirco_f1,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_macro_f1": val_mirco_f1,
                    "val_accuracy": val_accuracy,
                    "best_val_f1": best_val_f1
                }
            )

            # If the validation loss has improved, update the best validation loss and reset the number of epochs with no improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
            # If the validation loss has not improved for the specified number of epochs, log a message and stop the training process
            elif no_improvement_epochs >= patience:
                print(f'No improvement for {patience} epochs, stopping training...')
                break
            # If the validation loss has not improved but the patience threshold has not been reached, increment the number of epochs with no improvement
            else:
                no_improvement_epochs += 1
        best_epoch = torch.argmax(torch.tensor(val_f1s))
        """summary["best_train_f1"] = max(train_f1s)
        summary["best_val_f1"] = max(val_f1s)
        summary["best_train_accuracy"] = max(train_accs)
        summary["best_val_accuracy"] = max(val_accs)
        summary["best_epoch"] = best_epoch + 1"""

        best_train_f1s.append(max(train_f1s))
        best_val_f1s.append(max(val_f1s))
        best_train_accs.append(max(train_accs))
        best_val_accs.append(max(val_accs))

        """wandb.run.summary["best_train_f1"] = max(train_f1s)
        wandb.run.summary["best_val_f1"] = max(val_f1s)
        wandb.run.summary["best_train_accuracy"] = max(train_accs)
        wandb.run.summary["best_val_accuracy"] = max(val_accs)
        wandb.run.summary["best_epoch"] = torch.argmax(torch.tensor(val_f1s)) + 1"""

    overall_summary['best_train_f1'] = max(best_train_f1s)
    overall_summary['best_val_f1'] = max(best_val_f1s)
    overall_summary['best_train_acc'] = max(best_train_accs)
    overall_summary['best_val_acc'] = max(best_val_accs)

    overall_summary['avg_train_f1'] = np.around(np.mean(best_train_f1s), 2)
    overall_summary['avg_val_f1'] = np.around(np.mean(best_val_f1s), 2)
    overall_summary['avg_train_acc'] = np.around(np.mean(best_train_accs), 2)
    overall_summary['avg_val_acc'] = np.around(np.mean(best_val_accs), 2)


    wandb.run.summary.update(overall_summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for sweep')

    parser.add_argument(
        '-count',
        help='Number of sweep run',
        type=int, default=1,
    )
    args, remaining_args = parser.parse_known_args()
    sweep_id = wandb.sweep(sweep=sweep_config, project=config['wandb']['project'])
    wandb.agent(sweep_id, function=main, count=args.count)
    api = wandb.Api()
    sweep = api.sweep(f"{config['wandb']['entity']}/{config['wandb']['project']}/{sweep_id}")

    best_config = sweep.best_run().config

    with open('data/trans_feat_matrix.pkl', 'rb') as f:
        data = pickle.load(f)

    with open("data/frame_text_dataset.json", "r") as fd:
        chunk_data = json.load(fd)

    #Transductive
    feature_matrix = torch.tensor(data['feat'])
    feature_matrix = feature_matrix[:, 5:]
    G = data['graph']
    node_labels = defaultdict()

    for doc_id, doc_data in chunk_data.items():
        node_labels[doc_id] = doc_data['label']

    node_labels = {k: v.lower() for k, v in node_labels.items()}
    labels = sorted([lab.lower() for lab in list(set(list(node_labels.values())))])
    label2id = {label: i for i, label in enumerate(labels)}

    model_data = get_transductive_data(G, feature_matrix, node_labels, label2id)

    #Inductive

    """model_data = get_train_data(G, feature_matrix, node_labels, label2id)
    metadata = {'label2id': label2id, 'graph': G}"""

    # Set the seed for the random number generator
    SEED = config['train']['seed']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = config['train']['device']

    name = config['data']['name']

    overall_summary = defaultdict()
    best_train_f1s = []
    best_val_f1s = []
    best_train_accs = []
    best_val_accs = []
    for split, split_data in model_data.items():

        metadata = split_data['metadata']
        train_indices = metadata['train']
        val_indices = metadata['val']
        test_indices = metadata['test']
        data = split_data['data']
        label2id = metadata['label2id']

        num_features = data.x.shape[1]
        num_classes = len(label2id)

        #pred_data = get_pred_data(G, node_features, labels)

        # Create the GAT model
        model = GAT(in_channels=num_features, hidden_channels=best_config['num_hidden'], out_channels=num_classes,
                    heads=best_config['num_heads'],num_layers=best_config['num_layers'],dropout=best_config['dropout'],  dtype=torch.float32)
        model.to(device)

        # Define a loss function and an optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=float(best_config['lr']),
                               weight_decay=float(best_config['weight_decay']))

        # Early Stopping
        best_val_loss = float('inf')
        patience = config['train']['patience']
        no_improvement_epochs = 0

        # Iterate over the training data
        train_f1s = []
        train_accs = []
        val_f1s = []
        val_accs = []

        # Model info for each epoch
        model_info = []
        summary = defaultdict()
        for epoch in tqdm(range(config['train']['epochs'])):
            train_loss, train_mirco_f1, train_accuracy = train(model, loss_fn, optimizer, data,train_indices, device)
            train_f1s.append(train_mirco_f1)
            train_accs.append(train_accuracy)
            val_loss, val_mirco_f1, val_accuracy = evaluate(model, loss_fn, data, val_indices, device)
            val_f1s.append(val_mirco_f1)
            val_accs.append(val_accuracy)

            model_info.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_micro_f1": train_mirco_f1,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_mirco_f1": val_mirco_f1,
                    "val_accuracy": val_accuracy,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            )

            # If the validation loss has improved, update the best validation loss and reset the number of epochs with no improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
            # If the validation loss has not improved for the specified number of epochs, log a message and stop the training process
            elif no_improvement_epochs >= patience:
                print(f'No improvement for {patience} epochs, stopping training...')
                break
            # If the validation loss has not improved but the patience threshold has not been reached, increment the number of epochs with no improvement
            else:
                no_improvement_epochs += 1

        best_epoch = torch.argmax(torch.tensor(val_f1s))
        summary["best_train_f1"] = max(train_f1s)
        summary["best_val_f1"] = max(val_f1s)
        summary["best_train_accuracy"] = max(train_accs)
        summary["best_val_accuracy"] = max(val_accs)
        summary["best_epoch"] = best_epoch + 1


        torch.save(model_info[best_epoch], f"models/gnn/cat_emb/split-{split}_epoch-{best_epoch}.pt")

        best_train_f1s.append(max(train_f1s))
        best_val_f1s.append(max(val_f1s))
        best_train_accs.append(max(train_accs))
        best_val_accs.append(max(val_accs))

    overall_summary['best_train_f1'] = max(best_train_f1s)
    overall_summary['best_val_f1'] = max(best_val_f1s)
    overall_summary['best_train_acc'] = max(best_train_accs)
    overall_summary['best_val_acc'] = max(best_val_accs)

    overall_summary['avg_train_f1'] = np.around(np.mean(best_train_f1s),2)
    overall_summary['avg_val_f1'] = np.around(np.mean(best_val_f1s),2)
    overall_summary['avg_train_acc'] = np.around(np.mean(best_train_accs),2)
    overall_summary['avg_val_acc'] = np.around(np.mean(best_val_accs),2)

    print(overall_summary)
