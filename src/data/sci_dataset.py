import argparse
import json
from collections import defaultdict

import yaml
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.data_preprocessor import Preprocessor


class SciDataset():

    def __init__(self):
        self.data_path = 'data/dataset.json'
        with open(self.data_path) as f:
            self.data = json.load(f)
        f.close()
        self.fix_mixed_label()
        self.preprocess_labels()
        self.label2idx = {label:i for i, label in enumerate(sorted(set(self.data['label'])))}
        self.idx2label = {i:label for label, i in self.label2idx.items()}

        self.sep = 'SBA'

    def preprocess_labels(self):
        ## Source : https://core.ac.uk/download/pdf/234627217.pdf https://www.ucg.ac.me/skladiste/blog_609332/objava_105202/fajlovi/Creswell.pdf
        with open('data/labels.json') as f:
            self.label_data = json.load(f)
        args = {'remove_paran_content': True,
                'remove_pos': ["ADV", "PRON", "CCONJ", "PUNCT", "PART", "DET", "ADP", "SPACE", "NUM", "SYM"]}
        preprocessor = Preprocessor(args)
        quan_p = preprocessor.preprocess_text(self.label_data['Quantitative'])
        qual_p = preprocessor.preprocess_text(self.label_data['Qualitative'])
        mix_p = preprocessor.preprocess_text(self.label_data['Qualitative and Quantitative'])
        qual, quan, mix = preprocessor.pos_preprocessing(docs=[qual_p, quan_p, mix_p])
        self.label_data['Quantitative'] = quan
        self.label_data['Qualitative'] = qual
        self.label_data['Qualitative and Quantitative'] = mix
    def get_sectioned_text(self):
        new_data = []

        for secs, text in zip(self.data['sections'], self.data['full_text']):
            sec2text = defaultdict(str)
            for sec, txt in zip(secs, text.split(self.sep)):
                sec2text[sec] = txt
            new_data.append(sec2text)

        return new_data

    def fix_mixed_label(self):
        for i, label in enumerate(self.data['label']):
            if label == 'Mixed':
                self.data['label'][i] = 'Qualitative and Quantitative'

    def get_imp_sec_text(self):
        imp_sections = ['Introduction', 'Methodology', 'Conclusion', 'Conclusions', 'Discussion',
                        'Results', 'Concluding remarks', 'Method', 'Data']
        sectioned_data = defaultdict(list)
        new_data = self.get_sectioned_text()
        for sec2text in new_data:
            text = []
            for sec, txt in sec2text.items():
                text.extend([txt for s in imp_sections if sec in s or s in sec])
            sectioned_data['text'].append(' '.join(text))

        return sectioned_data['text']

    def get_abstract(self, type='raw'):
        if type == 'raw':
            return self.data['abstract']
        elif type == 'clean':
            abstracts = []
            for text in self.data['full_text']:
                abstracts.append(text.split(self.sep)[0])
            return abstracts
        else:
            ValueError(f'Incorrect type {type} mentioned. Use "raw" or "clean".')

    def get_full_text(self):
        full_text = []
        for text in self.data['full_text']:
            full_text.append(' '.join(text.split(self.sep)))
        return full_text

    def get_section_name(self, type='unit'):
        if type == 'unit':
            return self.data['sections']
        elif type == 'seq':
            return [' '.join(secs) for secs in self.data['sections']]
        else:
            ValueError(f'Incorrect type {type} mentioned. Use "unit" or "seq".')

    def get_title(self):
        return self.data['title']

    def get_label(self, type='alpha'):
        if type == 'alpha':
            return self.data['label']
        elif type == 'numeric':
            return [self.label2idx[l] for l in self.data['label']]
        else:
            ValueError(f'Incorrect type {type} mentioned. Use "alpha" or "numeric".')

    def get_text_label(self, data_type, label_type='alpha'):
        labels = self.get_label(label_type)
        text = None
        if data_type == 'abstract':
            text = self.get_abstract(type='clean')
        elif data_type == 'full_text':
            text = self.get_full_text()
        elif data_type == 'sec-name':
            text = self.get_section_name(type='seq')
        elif data_type == 'sec-text':
            text = self.get_imp_sec_text()
        else:
            ValueError(f'Incorrect data type {data_type}')

        return text, labels

    def get_train_test_split(self, data_type, label_type='alpha', n_splits = 5):
        text, label = self.get_text_label(data_type, label_type)
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(text, label):
            X_train, X_test = [text[i] for i in train_index], [text[i] for i in test_index]
            y_train, y_test = [label[i] for i in train_index], [label[i] for i in test_index]
            yield X_train, y_train, X_test, y_test

