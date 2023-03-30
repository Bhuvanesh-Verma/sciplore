import argparse
from collections import defaultdict

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.data.sci_dataset import SciDataset
from sklearn.metrics.pairwise import cosine_similarity


def classifier(embedding, qual_emb, quan_emb, mix_emb):

    sim_qual = cosine_similarity(embedding.reshape(1, -1), qual_emb.reshape(1, -1))[0][0]
    sim_quan = cosine_similarity(embedding.reshape(1, -1), quan_emb.reshape(1, -1))[0][0]
    sim_mix = cosine_similarity(embedding.reshape(1, -1), mix_emb.reshape(1, -1))[0][0]

    return np.array([sim_qual, sim_mix, sim_quan])

def pipeline(dataset, data_type, config):
    texts, labels = dataset.get_text_label(data_type)
    model = SentenceTransformer(config['model_name'])
    model.max_seq_length = 510
    quan_emb = model.encode(dataset.label_data['Quantitative'])
    qual_emb = model.encode(dataset.label_data['Qualitative'])
    mix_emb = model.encode(dataset.label_data['Qualitative and Quantitative'])
    max_token = config['max_token']
    preds = []

    for k, text in enumerate(texts):
        text = text.split()
        chunks = [text[i:i + max_token] for i in range(0, len(text), max_token)]
        embeddings = []
        for chunk in chunks:
            embeddings.append(model.encode(' '.join(chunk)))

        if config['chunk']:
            label_scores = defaultdict(list)
            for emb in embeddings:
                result = classifier(emb, qual_emb, quan_emb, mix_emb)

                for i, lab in dataset.idx2label.items():
                    label_scores[lab].append(result[i])
            predicted_label = max(label_scores, key=label_scores.get)
        else:
            embedding = np.mean(embeddings, axis=0)
            result = classifier(embedding, qual_emb, quan_emb, mix_emb)
            predicted_label = dataset.idx2label[np.argmax(result)]
        preds.append(predicted_label)
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    print(classification_report(labels, preds))
    return {'f1_score': f1, 'accuracy': acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for similarity based models')

    parser.add_argument(
        '-config',
        help='Path to train config file',
        type=str, default='./configs/sim_train.yaml',
    )

    parser.add_argument(
        '-data_type',
        help='type of data to use for modelling',
        type=str, default='full_text',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    dataset = SciDataset()

    data_type = args.data_type

    stats = pipeline(dataset, data_type, config_data)
    print(stats)
