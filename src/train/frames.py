import argparse
import json

import numpy as np
import yaml
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.sci_dataset import SciDataset
from src.train.base_models import get_features, train_model

sent_bound = 'STB'
sec_bound = 'SBA'

def get_section_frames(dataset):
    imp_sections = ['Introduction', 'Methodology', 'Conclusion', 'Conclusions', 'Discussion',
                    'Results', 'Concluding remarks', 'Method', 'Data']
    sci_dataset = SciDataset()
    sec_names = sci_dataset.get_section_name()
    data = []
    for secs, (doc_id, doc_data) in zip(sec_names, dataset.items()):
        sec_text = [' '.join(doc_data['abstract'])]
        sec_text.extend(' '.join(doc_data['text']).split(sec_bound)[:-1])
        full_text = []

        for sec, text in zip(secs, sec_text):

            for s in imp_sections:
                if s in sec or sec in s:
                    text = text.replace(sent_bound, '')
                    full_text.append(text)
                    break
        data.append(' '.join(full_text))

    return data

def pipeline(dataset, data_type, config):
    labels = []
    data = []
    if data_type == 'abstract':
        for doc_id, doc_data in dataset.items():
            labels.append(doc_data['label'])
            abstract = ' '.join(doc_data['abstract']).replace(sent_bound, '')
            data.append(abstract)
    elif data_type == 'full_text':
        for doc_id, doc_data in dataset.items():
            labels.append(doc_data['label'])
            text = ' '.join(doc_data['text']).replace(sec_bound, '').replace(sent_bound, '')
            data.append(text)
    elif data_type == 'sec-text':
        for doc_id, doc_data in dataset.items():
            labels.append(doc_data['label'])
        data = get_section_frames(dataset)
    else:
        ValueError(f'Incorrect data type {data_type}')

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    best_f1 = -1
    f1s = []
    accs = []
    best_acc = -1
    for k, (train_index, test_index) in enumerate(sss.split(data, labels)):
        X_train, X_test = [data[i] for i in train_index], [data[i] for i in test_index]
        y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]
        print(f'\n-------Split {k + 1}-------')
        X_train, X_test = get_features(X_train, X_test, config['feat_type'])
        clf = train_model(X_train, y_train, params=config['params'], model_type=config['model_type'])
        best_clf = clf.best_estimator_
        y_pred = best_clf.predict(X_test)
        # print(classification_report(y_test, y_pred))
        f_score = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Test macro f1 score: {f_score}')
        print(f'Test accuracy: {accuracy}')

        if f_score > best_f1:
            best_f1 = f_score
            best_params = clf.best_params_
        if accuracy > best_acc:
            best_acc = accuracy
        f1s.append(f_score)
        accs.append(accuracy)
    stats = {'best_f1': np.around(best_f1, 2), 'avg_f1': np.around(np.mean(f1s), 2),
             'best_accuracy': np.around(best_acc, 2),
             'avg_accuracy': np.around(np.mean(accs), 2), 'best_params': best_params}
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training frame based models')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/frame_train.yaml',
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