import json
import re
from collections import defaultdict

from transformers import BertTokenizer


sent_bound = 'STB'
sec_bound = 'SBA'
exp = r'(?<=[.!?])\s+(?=[A-Z])\s*(?![^()]*\))'
imp_sections = ['Introduction', 'Methodology', 'Conclusion', 'Conclusions', 'Discussion',
                    'Results', 'Concluding remarks', 'Method', 'Data']
def chunk_text(text, frames, max_tokens=512):
    # Tokenize text into sentences
    sentences = re.split(exp, text)
    frames = frames.split(sent_bound)[:-1]

    # Initialize chunks list and current chunk
    text_chunks = []
    frame_chunk = []
    current_chunk = ""
    current_idx = []


    # Loop through sentences and add to chunks while preserving sentence boundaries
    for i, sentence in enumerate(sentences):
        if frames[i] == ' ':
            continue
        sentence_tokens = tokenizer.tokenize(sentence)
        if len(current_chunk.split()) + len(sentence_tokens) <= max_tokens:
            # If adding the sentence to the current chunk doesn't exceed the token limit, add it to the current chunk
            current_chunk += " " + sentence
            current_idx.append(i)
        else:
            # If adding the sentence to the current chunk would exceed the token limit, start a new chunk
            if current_chunk:
                text_chunks.append(current_chunk.strip())
            if len(current_idx) > 0:
                curr_frames = [' '.join(f.split('_')).strip() for k, frame in enumerate(frames) for f in frame.split() if k in current_idx]
                frame_chunk.append(' '.join(curr_frames).strip())

            current_chunk = sentence
            current_idx = [i]

    # Add final chunk to chunks list
    if current_chunk:
        text_chunks.append(current_chunk.strip())
    if len(current_idx) > 0:
        curr_frames = [' '.join(f.split('_')).strip() for k, frame in enumerate(frames) for f in frame.split() if k in current_idx]
        frame_chunk.append(' '.join(curr_frames).strip())

    return text_chunks, frame_chunk


if __name__ == '__main__':
    data_path = 'data/balance_corpus.json'
    frame_path = 'data/frames_dataset.json'
    with open(data_path) as f:
        text_data = json.load(f)

    with open(frame_path) as f:
        frame_data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    frame_text_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (doc_id, doc_data),(f_id, f_data) in zip(text_data.items(), frame_data.items()):
        if f_id != doc_id:
            continue
        abstract_text = doc_data['abstract']
        abstract_frames = ' '.join(f_data['abstract'])
        frame_text_data[f_id]['text']['abstract'], frame_text_data[f_id]['frame']['abstract'] = chunk_text(abstract_text, abstract_frames)

        full_text = doc_data['text']
        all_frames = ' '.join(f_data['text']).split(sec_bound)[:-1]
        txts = []
        frs = []
        s_txts = []
        s_frs = []
        for (sec_name, sec_text), frames in zip(full_text.items(), all_frames):
            if sec_text == '' or frames == '':
                continue
            sec_chunks, sec_frame_chunks = chunk_text(sec_text, frames)
            for s in imp_sections:
                if s in sec_name or sec_name in s:
                    s_txts.extend(sec_chunks)
                    s_frs.extend(sec_frame_chunks)
                    break
            txts.extend(sec_chunks)
            frs.extend(sec_frame_chunks)
        frame_text_data[f_id]['text']['full_text'] = txts
        frame_text_data[f_id]['frame']['full_text'] = frs
        frame_text_data[f_id]['text']['sec_text'] = s_txts
        frame_text_data[f_id]['frame']['sec_text'] = s_frs
        frame_text_data[f_id]['label'] = f_data['label']

    with open('data/frame_text_dataset.json', "w+") as outfile:
        json.dump(frame_text_data, outfile, indent=4, sort_keys=False)








