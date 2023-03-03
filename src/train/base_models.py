import argparse

import fasttext
import numpy as np
import yaml
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.data.sci_dataset import SciDataset


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


def get_features(X_train, X_test, feat_type='bow'):
    if feat_type == 'bow':
        X_train, X_test = get_bow_rep(X_train, X_test)
    elif feat_type == 'tfidf':
        X_train, X_test = get_tfidf_rep(X_train, X_test)
    else:
        ValueError(f'Incorrect feature type {feat_type}')
    return X_train, X_test


def pipeline(dataset, data_type, config):
    best_f1 = -1
    f1s = []
    accs = []
    best_acc = -1
    # Split the data into training and testing sets
    for i, (X_train, y_train, X_test, y_test) in enumerate(dataset.get_train_test_split(data_type, n_splits=5)):
        print(f'\n-------Split {i + 1}-------')
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
    parser = argparse.ArgumentParser(description='Arguments for training base models')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/base_train.yaml',
    )

    parser.add_argument(
        '-data_type',
        help='type of data to use for modelling',
        type=str, default='sec-text',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    dataset = SciDataset()

    data_type = args.data_type

    stats = pipeline(dataset, data_type, config_data)
    print(stats)
