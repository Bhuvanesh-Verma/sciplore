import json
import re
from collections import defaultdict

from transformers import BertTokenizer


sent_bound = 'STB'
sec_bound = 'SBA'
exp = r'(?<=[.!?])\s+(?=[A-Z])\s*(?![^()]*\))'
imp_sections = ['Introduction', 'Methodology', 'Conclusion', 'Conclusions', 'Discussion',
                    'Results', 'Concluding remarks', 'Method', 'Data']
def chunk_text(text, max_tokens=32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Tokenize text into sentences
    sentences = re.split(exp, text)

    # Initialize chunks list and current chunk
    text_chunks = []
    current_chunk = ""


    # Loop through sentences and add to chunks while preserving sentence boundaries
    for i, sentence in enumerate(sentences):

        sentence_tokens = tokenizer.tokenize(sentence)
        if len(current_chunk.split()) + len(sentence_tokens) <= max_tokens:
            # If adding the sentence to the current chunk doesn't exceed the token limit, add it to the current chunk
            current_chunk += " " + sentence
        else:
            # If adding the sentence to the current chunk would exceed the token limit, start a new chunk
            if current_chunk:
                text_chunks.append(current_chunk.strip())

            current_chunk = sentence

    # Add final chunk to chunks list
    if current_chunk:
        text_chunks.append(current_chunk.strip())

    return text_chunks

def chunk_sent(text):
    return re.split(exp, text)

if __name__ == '__main__':
    data_path = 'data/labels.json'
    save_path = 'data/3_labels_dataset_32.json'
    with open(data_path) as f:
        text_data = json.load(f)

    label_dataset = defaultdict(list)

    for label, text in text_data.items():
        chunks = chunk_text(text)
        label_dataset['text'].extend(chunks)
        label_dataset['label'].extend([label]*len(chunks))


    with open(save_path, "w+") as outfile:
        json.dump(label_dataset, outfile, indent=4, sort_keys=False)








