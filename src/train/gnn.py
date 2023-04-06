import argparse
import json
import pickle
import random
from collections import defaultdict
from os import listdir
from os.path import join, isfile

import networkx as nx
import numpy as np
import torch
import wandb
import yaml
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.nn import GAT
from tqdm import tqdm

from src.utils.graph_dataset import get_transductive_data


def train(model, loss_fn, optimizer, train_data, train_indices, device):
    model.train()

    # Pass the input data through the model to calculate the output
    x = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)
    y = train_data.y.to(device).index_select(0,torch.tensor(train_indices).to(device))

    output = model(x, edge_index).index_select(0,torch.tensor(train_indices).to(device))

    # Calculate the loss between the output and the target labels
    train_loss = loss_fn(output, y)

    preds = torch.argmax(output, dim=1)
    train_mirco_f1 = f1_score(y.cpu(), preds.cpu(), average='macro')
    train_accuracy = accuracy_score(y.cpu(), preds.cpu())


    # Backpropagate the loss to update the model weights
    train_loss.backward()

    # Update the model weights using the optimizer
    optimizer.step()

    return train_loss, train_mirco_f1, train_accuracy

def evaluate(model, loss_fn, val_data, val_indices, device):
    # Evaluate the GAT model on the test data
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        x = val_data.x.to(device)
        edge_index = val_data.edge_index.to(device)
        y = val_data.y.to(device).index_select(0,torch.tensor(val_indices).to(device))
        output = model(x, edge_index).index_select(0,torch.tensor(val_indices).to(device))
        val_loss = loss_fn(output, y)
        preds = torch.argmax(output, dim=1)
        val_mirco_f1 = f1_score(y.cpu(), preds.cpu(), average='macro')
        val_accuracy = accuracy_score(y.cpu(), preds.cpu())
        #print(classification_report(y.cpu(), preds.cpu()))

    return val_loss, val_mirco_f1, val_accuracy


def gnn_pipeline(data, chunk_data, config):
    # Transductive
    feature_matrix = torch.tensor(data['feat'])
    feat_type = config['data']['feat_type']
    if feat_type == 'struc':
        feature_matrix = feature_matrix[:, :5]
    elif feat_type == 'struc+cat':
        feature_matrix = feature_matrix[:, :505]
    elif feat_type == 'cat':
        feature_matrix = feature_matrix[:, 5:505]
    elif feat_type == 'struc+emb':
        feature_matrix = torch.cat((feature_matrix[:, :5], feature_matrix[:, 505:]), dim=1)
    elif feat_type == 'cat+emb':
        feature_matrix = feature_matrix[:, 5:]
    elif feat_type == 'emb':
        feature_matrix = feature_matrix[:, 505:]
    else:
        feature_matrix = feature_matrix

    G = data['graph']
    node_labels = defaultdict()

    for doc_id, doc_data in chunk_data.items():
        node_labels[doc_id] = doc_data['label']

    node_labels = {k: v.lower() for k, v in node_labels.items()}
    labels = sorted([lab.lower() for lab in list(set(list(node_labels.values())))])
    label2id = {label: i for i, label in enumerate(labels)}

    model_data = get_transductive_data(G, feature_matrix, node_labels, label2id)

    # Inductive

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

        # pred_data = get_pred_data(G, node_features, labels)

        # Create the GAT model
        model = GAT(in_channels=num_features, hidden_channels=config['model']['num_hidden'], out_channels=num_classes,
                    heads=config['model']['num_heads'], num_layers=config['model']['num_layers'],
                    dropout=config['model']['dropout'], dtype=torch.float32)
        model.to(device)

        # Define a loss function and an optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=float(config['optimizer']['lr']),
                               weight_decay=float(config['optimizer']['weight_decay']))

        if config['train']['load']:
            checkpoint = torch.load(config['train']['ckpt'], map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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
            train_loss, train_mirco_f1, train_accuracy = train(model, loss_fn, optimizer, data, train_indices, device)
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

        if config['train']['save']:
            torch.save(model_info[best_epoch], f"{config['train']['save_path']}/{name}.pt")

        best_train_f1s.append(max(train_f1s))
        best_val_f1s.append(max(val_f1s))
        best_train_accs.append(max(train_accs))
        best_val_accs.append(max(val_accs))

    overall_summary['best_train_f1'] = max(best_train_f1s)
    overall_summary['best_val_f1'] = max(best_val_f1s)
    overall_summary['best_train_acc'] = max(best_train_accs)
    overall_summary['best_val_acc'] = max(best_val_accs)

    overall_summary['avg_train_f1'] = np.around(np.mean(best_train_f1s), 2)
    overall_summary['avg_val_f1'] = np.around(np.mean(best_val_f1s), 2)
    overall_summary['avg_train_acc'] = np.around(np.mean(best_train_accs), 2)
    overall_summary['avg_val_acc'] = np.around(np.mean(best_val_accs), 2)

    print(overall_summary)
    return overall_summary




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for multi label classification training')

    parser.add_argument(
        '-config',
        help='Path to data config file',
        type=str, default='configs/gnn_train.yaml',
    )
    parser.add_argument(
        '-data_type',
        help='type of data to use for modelling',
        type=str, default='full_text',
    )

    args, remaining_args = parser.parse_known_args()
    data_type = args.data_type

    with open(args.config) as file:
        config = yaml.safe_load(file)

    with open(f'data/trans_feat_matrix_{data_type}.pkl', 'rb') as f:
        data = pickle.load(f)

    with open("data/frame_text_dataset.json", "r") as fd:
        chunk_data = json.load(fd)

    gnn_pipeline(data, chunk_data, config)
