import json
import pickle
import random
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from torch_geometric.data import Data

node_features = ['publisher', 'type', 'subject']
embedding_features = ['articletitle', 'journaltitle', 'abstract']


def add_feature_to_nodes(G, paper_id, metadata):
    for feature in node_features + embedding_features:
        if feature in metadata[paper_id]:
            if metadata[paper_id][feature]:
                G.nodes[paper_id][feature] = metadata[paper_id][feature]

    return G

def get_cat_features(features):
    with open('data/doi2metadata_abs.json', "r") as fd:
        data = json.load(fd)
    ohf = []
    for feature in features:
        if feature == 'subject':
            unique_subjects = set(
                [i for doi, val in data.items() if feature in val and len(val[feature]) != 0 for i in val[feature]])
            ohf.extend(list(unique_subjects))
        else:
            unique_v = set([val[feature] for doi, val in data.items() if feature in val])
            ohf.extend(list(unique_v))
    ohf = sorted([f.lower() for f in ohf if f is not None])
    node_feature2i = {feat:idx for idx, feat in enumerate(ohf)}
    return ohf, node_feature2i

def one_hot_encode(feat2i, features):
    encoding = [0] * len(feat2i)
    if not features:
        return encoding
    for feature in features:
        encoding[feat2i[feature]] = 1
    return  encoding

def get_ptg_data(G, node_features, nodes, label2id, labels):
    node2idx = {node:idx for idx, node in enumerate(G.nodes()) if node in nodes}
    indices = list(node2idx.values())
    random.shuffle(indices)
    x = node_features[indices]
    idx2new_i = {idx: i for i, idx in enumerate(indices)}
    source_target_idx = [(node2idx[s], node2idx[t]) for s, t in G.edges if t in nodes and s in nodes]
    edge_index = torch.tensor([[idx2new_i[s] for s, t in source_target_idx], [idx2new_i[t] for s, t in source_target_idx]],
                              dtype=torch.long)
    idx2node = {idx:node for node, idx in node2idx.items()}
    y_labeled = torch.tensor([label2id[labels[idx2node[idx]]] for idx in indices])
    data = Data(x=x.to(torch.float32), edge_index=edge_index, y=y_labeled)
    return data
def get_train_data(G, node_features, labels, label2id):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    doc_data = list(labels.keys())
    labs = list(labels.values())
    model_data = defaultdict(lambda: defaultdict())
    for k, (train_index, test_index) in enumerate(sss.split(doc_data, labs)):
        train_nodes, test_nodes = [doc_data[i] for i in train_index], [doc_data[i] for i in test_index]
        #y_train, y_test = [label2id[labs[i]] for i in train_index], [label2id[labs[i]] for i in test_index]

        train_data = get_ptg_data(G, node_features, train_nodes, label2id, labels)
        test_data = get_ptg_data(G, node_features, test_nodes, label2id, labels)
        model_data[k]['train'] = train_data
        model_data[k]['test'] = test_data

    return model_data


def get_transductive_data(G,feat,labels, label2id):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    doc_data = list(labels.keys())
    labs = list(labels.values())
    model_data = defaultdict(lambda: defaultdict())
    for k, (train_index, test_index) in enumerate(sss.split(doc_data, labs)):
        train_nodes, test_nodes = [doc_data[i] for i in train_index], [doc_data[i] for i in test_index]
        train_indices, val_indices = [i for i, node in enumerate(G.nodes()) if node in train_nodes],  [i for i, node in enumerate(G.nodes()) if node in test_nodes]
        pred_indices = [i for i, node in enumerate(G.nodes()) if node not in labels]

        node2idx = {node: idx for idx, node in enumerate(G.nodes())}
        source_target_idx = [(node2idx[s], node2idx[t]) for s, t in G.edges]
        edge_index = torch.tensor([[s for s, t in source_target_idx], [t for s, t in source_target_idx]], dtype=torch.long)
        y_labeled = []
        for node in G.nodes():
            if node in labels:
                y_labeled.append(label2id[labels[node]])
            else:
                y_labeled.append(-1)
        y_labeled = torch.tensor(y_labeled)
        data = Data(x=feat.to(torch.float32), edge_index=edge_index, y=y_labeled)
        metadata = {'train': train_indices, 'val': val_indices, 'test': pred_indices, 'label2id':label2id, 'graph':G}
        model_data[k]['data'] = data
        model_data[k]['metadata'] = metadata
    return model_data
def calculate_struct_features(graph, node,):
    features = {}

    # Node degree
    total_nodes = graph.number_of_nodes()
    in_degree = graph.in_degree(node)
    out_degree = graph.out_degree(node)
    features["in_degree"] = in_degree/total_nodes
    features["out_degree"] = out_degree/total_nodes

    # Node neighbors
    """neighbor_attributes = []
    for neighbor in graph.neighbors(node):
        neighbor_attributes.append(graph.node[neighbor]["attribute"])
    features["avg_neighbor_attribute"] = sum(neighbor_attributes) / len(neighbor_attributes)
    features["min_neighbor_attribute"] = min(neighbor_attributes)
    features["max_neighbor_attribute"] = max(neighbor_attributes)"""

    # Graph structure
    shortest_paths = nx.shortest_path_length(graph, node)
    features["mean_shortest_path_length"] = sum(shortest_paths.values()) / len(shortest_paths)
    features["clustering_coefficient"] = nx.clustering(graph, node)

    return list(features.values())


def generate_features():
    with open("data/new_citation_net.json", "r") as fd:
        citation_net = json.load(fd)

    with open("data/doi2metadata_abs.json", "r") as fd:
        metadata = json.load(fd)

    with open("data/frame_text_dataset.json", "r") as fd:
        chunk_data = json.load(fd)

    filter_data = defaultdict()

    for doc_id, citations in citation_net.items():
        if doc_id in chunk_data:
            filter_data[doc_id] = citations

    G = nx.DiGraph()

    device = 'cuda'
    only_embed = False

    for paper_id, cites in filter_data.items():
        if paper_id not in metadata:
            continue
        for cite_id in cites:
            if cite_id not in metadata:
                continue
            G.add_edge(paper_id, cite_id)
            G = add_feature_to_nodes(G, cite_id, metadata)
        G = add_feature_to_nodes(G, paper_id, metadata)

    cat_features, node_feature2i = get_cat_features(node_features)

    # Compute PageRank
    pagerank = nx.pagerank(G)

    model = SentenceTransformer(
        'sentence-transformers/all-mpnet-base-v2')  # paraphrase-multilingual-mpnet-base-v2 all-mpnet-base-v2

    # model = transformers.BertModel.from_pretrained('bert-base-uncased')
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
    # model = AutoModel.from_pretrained("facebook/opt-30b")
    model.to(device)
    # Create the feature matrix
    feature_matrix = []
    for node in tqdm(G.nodes()):
        feature_emebedding = []
        if only_embed:
            features = []
        else:
            features = calculate_struct_features(G, node)
            features.extend([pagerank[node]])
            avail_node_features = []
            for feature in node_features:
                if feature in G.nodes[node]:
                    if feature == 'subject':
                        feat_val = [val.lower() for val in G.nodes[node][feature] if val.lower() in cat_features]
                        avail_node_features.extend(feat_val)
                    else:
                        feat_val = G.nodes[node][feature].lower()
                        if feat_val in cat_features:
                            avail_node_features.append(feat_val)

            # cat_feat = [x / sum(cat_feat) for x in cat_feat]
            features.extend(one_hot_encode(node_feature2i, avail_node_features))
        for feature in embedding_features:
            if feature in G.nodes[node]:
                feat_val = G.nodes[node][feature].lower()
                embeds = model.encode(feat_val)
                feature_emebedding.append(embeds.tolist())
        if feature_emebedding:
            averaged_embeddings = np.mean(np.array(feature_emebedding), axis=0)
            features.extend(list(averaged_embeddings))
        else:
            features.extend([0] * model.config.hidden_size)
        feature_matrix.append(features)
    feature_matrix = torch.tensor(feature_matrix)
    with open('data/trans_feat_matrix.pkl', 'wb') as f:
        pickle.dump({'feat':feature_matrix, 'graph':G}, f)
