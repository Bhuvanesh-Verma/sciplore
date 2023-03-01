import argparse
import json
from collections import defaultdict

import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from src.utils.data_preprocessor import Preprocessor

import fasttext
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def get_bow_rep(X_train, X_test):
    # todo: Option to add parameters like ngram_range
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

def get_tfidf_rep(X_train, X_test):
    # todo: Option to add parameters like ngram_range
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec

def get_fasttext_rep(data):
    ft_model = fasttext.load_model('models/cc.en.300.bin')
    X = []
    for doc in data:
        doc_vector = []
        for word in doc.split():
            if word in ft_model:
                doc_vector.append(ft_model[word])
        if len(doc_vector) > 0:
            doc_embedding = np.mean(doc_vector, axis=0)
        else:
            doc_embedding = np.zeros((300,))
        X.append(doc_embedding)

    return X

def get_doc2vec_rep(data):
    tagged_docs = [TaggedDocument(words.split(), [str(i)]) for i, words in enumerate(data)]
    d2v_model = Doc2Vec(tagged_docs, vector_size=300, min_count=2, epochs=40)
    doc_embeddings = [d2v_model.infer_vector(doc.words) for doc in tagged_docs]
    return doc_embeddings

def train_model(X_train, y_train, params, model_type='svm'):
    model = None
    if model_type == 'knn':
        model = KNeighborsClassifier()
    elif model_type == 'svm':
        model = SVC()
    elif model_type == 'bayes':
        model = MultinomialNB()
    else:
        ValueError(f'Incorrect model type {model_type}')
    scorer = make_scorer(f1_score, average='macro')
    clf = GridSearchCV(model, params, cv=5, scoring=scorer)
    clf.fit(X_train, y_train)
    print('Best hyperparameters:', clf.best_params_)
    print(f'Macro F1 score:{clf.best_score_}\n')
    return clf


def pipeline(text, labels, config):
    stats = defaultdict()
    n_splits = 5  # number of splits to create
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    best_f1 = -1
    f1s = []
    accs = []
    best_acc = -1
    # Split the data into training and testing sets
    for i, (train_index, test_index) in enumerate(sss.split(text, labels)):
        print(f'\n-------Split {i+1}-------')
        X_train, X_test = [text[i] for i in train_index], [text[i] for i in test_index]
        y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]
        X_train, X_test = get_features(X_train, X_test, config['feat_type'])
        clf = train_model(X_train, y_train, params=config['params'], model_type=config['model_type'])
        best_clf = clf.best_estimator_
        y_pred = best_clf.predict(X_test)
        #print(classification_report(y_test, y_pred))
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
    stats = {'best_f1': np.around(best_f1,2), 'avg_f1': np.around(np.mean(f1s), 2), 'best_accuracy': np.around(best_acc,2),
             'avg_accuracy': np.around(np.mean(accs),2), 'best_params': best_params}
    return stats


def get_features(X_train, X_test, feat_type='bow'):
    if feat_type == 'bow':
        X_train, X_test = get_bow_rep(X_train, X_test)
    elif feat_type == 'tfidf':
        X_train, X_test = get_tfidf_rep(X_train, X_test)
    else:
        ValueError(f'Incorrect feature type {feat_type}')
    return X_train, X_test

def get_section_text():
    corpus_path = 'data/balance_corpus.json'
    with open(corpus_path) as f:
        corpus = json.load(f)
    imp_sections = ['Introduction', 'Methodology', 'Conclusion', 'Conclusions', 'Discussion',
                    'Results', 'Concluding remarks', 'Method', 'Data']
    sectioned_data = defaultdict(list)
    for i, v in corpus.items():
        text = []
        for k, txt in v['text'].items():
            text.extend([txt for s in imp_sections if k in s])
        sectioned_data['text'].append(' '.join(text))
        sectioned_data['label'].append(v['label'])

    args = {'remove_paran_content': True,
            'remove_pos': ["ADV", "PRON", "CCONJ", "PUNCT", "PART", "DET", "ADP", "SPACE", "NUM", "SYM"]}

    preprocessor = Preprocessor(args)

    for i, txt in enumerate(sectioned_data['text']):
        sectioned_data['text'][i] = preprocessor.preprocess_text(txt)
    sectioned_data['text'] = preprocessor.pos_preprocessing(docs=sectioned_data['text'])

    return sectioned_data['text'], sectioned_data['label']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training base models')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/base_train.yaml',
    )

    parser.add_argument(
        '-data_type',
        help='type of data to use for modelling',
        type=str, default='abstract',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    data_path = config_data['data_path']
    with open(data_path) as f:
        data = json.load(f)
    data_type = args.data_type
    labels = data['label']
    text = None
    if data_type == 'abstract':
        text = data['abstract']
    elif data_type == 'full_text':
        text = data['text']
    elif data_type == 'sec-name':
        text = [' '.join(secs) for secs in data['sections']]
    elif data_type == 'sec-text':
        text, labels = get_section_text()
    else:
        ValueError(f'Incorrect data type {data_type}')

    stats = pipeline(text, labels, config_data)
    print(stats)