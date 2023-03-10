import argparse
import json

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from src.train.frames import get_section_frames
from src.utils.SimpleClassifier import SimpleNeuralNetwork


def neural_pipeline(train_data, train_labels, val_data, val_labels):
    # Convert the training data and labels to tensors
    train_data = torch.Tensor(train_data)
    train_labels = torch.Tensor(train_labels)

    # Create a TensorDataset object from the training data and labels
    train_dataset = TensorDataset(train_data, train_labels)

    # Create a DataLoader object from the training dataset
    batch_size = 8
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Convert the validation data and labels to tensors
    val_data = torch.Tensor(val_data)
    val_labels = torch.Tensor(val_labels)

    # Create a TensorDataset object from the validation data and labels
    val_dataset = TensorDataset(val_data, val_labels)

    # Create a DataLoader object from the validation dataset
    batch_size = 4
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleNeuralNetwork(in_size=384, out_size=3)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Define the number of epochs and move the model to the GPU (if available)
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Start the training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            labels = torch.tensor(labels, dtype=torch.long)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')
    # Run the validation loop at the end of each epoch
    correct = 0
    total = 0
    outputs_ = []
    trues = []
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            trues.extend(labels.tolist())
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, dim=1)
            outputs_.extend(predicted.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (
            100 * correct / total))

    return trues, outputs_


def get_features(X_train, X_test):

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    train_data = np.zeros((len(X_train), 384))
    for i, sentence in enumerate(X_train):
        train_data[i] = model.encode(sentence)

    test_data = np.zeros((len(X_test), 384))
    for i, sentence in enumerate(X_test):
        test_data[i] = model.encode(sentence)

    return train_data, test_data
def pipeline(dataset, data_type, config):
    exp_type = config['exp_type']
    labels = []
    data = []
    for doc_id, doc_data in dataset.items():
        chunks = doc_data[exp_type][data_type]
        data.extend(chunks)
        labels.extend([doc_data['label']]*len(chunks))

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    best_f1 = -1
    f1s = []
    accs = []
    best_acc = -1
    label2idx = {label:i for i, label in enumerate(list(sorted(set(labels))))}
    idx2label = {i:label for label, i in label2idx.items()}
    for k, (train_index, test_index) in enumerate(sss.split(data, labels)):
        X_train, X_test = [data[i] for i in train_index], [data[i] for i in test_index]
        y_train, y_test = [label2idx[labels[i]] for i in train_index], [label2idx[labels[i]] for i in test_index]
        X_train, X_test = get_features(X_train, X_test)
        true, pred = neural_pipeline(X_train, y_train, X_test, y_test)
        true = [idx2label[t] for t in true]
        pred = [idx2label[p] for p in pred]
        f_score = f1_score(true, pred, average='macro')
        accuracy = accuracy_score(true, pred)
        print(f'Test macro f1 score: {f_score}')
        print(f'Test accuracy: {accuracy}')
        print(classification_report(true, pred))
        if f_score > best_f1:
            best_f1 = f_score
        if accuracy > best_acc:
            best_acc = accuracy
        f1s.append(f_score)
        accs.append(accuracy)
    stats = {'best_f1': np.around(best_f1, 2), 'avg_f1': np.around(np.mean(f1s), 2),
             'best_accuracy': np.around(best_acc, 2),
             'avg_accuracy': np.around(np.mean(accs), 2),}
    return stats


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

    stats = pipeline(dataset, args.data_type, config_data)
    print(stats)