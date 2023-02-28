import json
from collections import defaultdict

# This script is used to select 19 articles from each class and create new dataset file

data_path = 'data/full_corpus.json'
save_path = 'data/balance_corpus.json'
with open(data_path) as f:
    data = json.load(f)

label2doc = defaultdict(list)
for k,v in data.items():
    label2doc[v['rd']].append(k)

new_dataset = defaultdict(dict)
for label, doc_ids in label2doc.items():
    for id in doc_ids[:19]:
        doc = data[id]
        title = doc['title']
        abstract = doc['abstract']
        text = doc['text']
        doi = doc['doi']
        new_dataset[doi] = {'file_name':doc['file_name'],'title': title, 'abstract': abstract, 'label': label, 'text': text}

with open(save_path, "w+") as outfile:
    json.dump(new_dataset, outfile, indent=4, sort_keys=False)
