import json
from collections import defaultdict
from src.utils.data_preprocessor import Preprocessor


def process_scientific_article(data, preprocessor, chunk=False):
    sections = []
    text = preprocessor.preprocess_text(data['abstract'])
    sections.append('Abstract')
    for head, content in data['text'].items():
        if content and content != '':
            text = text + " SBA " + preprocessor.preprocess_text(content)
            sections.append(head)

    return text, sections


def preprocess(json_data, processor_args, chunk=False):
    docs = defaultdict(list)
    preprocessor = Preprocessor(processor_args)
    print('Preprocessing docs')
    for i, data in json_data.items():
        text, sections = process_scientific_article(data, preprocessor, chunk=chunk)
        docs['full_text'].append(text)
        docs['abstract'].append(data['abstract'])
        docs['id'].append(i)  # Used for supervised learning
        docs['label'].append(data['label'])
        docs['title'].append(data['title'])
        docs['sections'].append(sections)


    # POS preprocessing
    if 'remove_pos' in processor_args:
        docs['full_text'] = preprocessor.pos_preprocessing(docs=docs['full_text'])
    return docs


if __name__ == '__main__':
    data_path = 'data/balance_corpus.json'
    save_path = 'data/dataset.json'
    with open(data_path) as f:
        data = json.load(f)
    args = {'remove_paran_content': True,
            'remove_pos':["ADV","PRON","CCONJ","PUNCT","PART","DET","ADP","SPACE","NUM","SYM"]}
    docs = preprocess(json_data=data,processor_args=args, chunk=True)

    with open(save_path, "w+") as outfile:
        json.dump(docs, outfile, indent=4, sort_keys=False)



