# Dataset class
import json
import re

import pandas as pd
import torch

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer


class RDDataset (Dataset):
    def __init__(self, txt_list, label_list, tokenizer, max_length):
        # define variables
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        # iterate through the dataset
        for txt, label in zip(txt_list, label_list):
            # prepare the text
            prep_txt = f'<startoftext>Sentence: {txt}\nResearch Design: {label}<endoftext>'
            # tokenize
            encodings_dict = tokenizer(prep_txt, truncation=True,max_length=max_length, padding= "max_length")
            # append to list
            self.input_ids.append(torch. tensor(encodings_dict['input_ids']))
            self.attn_masks.append (torch. tensor(encodings_dict['attention_mask']))
            self.labels.append(label)
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self. input_ids[idx], self.attn_masks[idx], self. labels[idx]

# Data load function
def load_sentiment_dataset (tokenizer):
    data_path = 'data/3_labels_dataset_32.json'
    with open(data_path) as f:
        text_data = json.load(f)
    texts = text_data['text']
    labels = text_data['label']
    label2idx = {label: i for i, label in enumerate(list(sorted(set(labels))))}
    idx2label = {i: label for label, i in label2idx.items()}
    X_train, X_test, y_train, y_test = train_test_split(texts, labels,shuffle=True, test_size=0.1, random_state=42, stratify=labels)
    X_eval, X_test, y_eval, y_test = train_test_split(X_test, y_test,shuffle=True, test_size=0.5, random_state=42, stratify=y_test)

    train_dataset = RDDataset(X_train, y_train, tokenizer, max_length=512)
    eval_dataset = RDDataset(X_eval, y_eval, tokenizer, max_length=512)
    # return
    return train_dataset, eval_dataset, (X_test, y_test)


## Load model and data
# set model name
model_name = "gpt2"
# seed
torch.manual_seed (42)
# load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token= '<startoftext>',eos_token= '<endoftext>', pad_token= '<pad>')
model = GPT2LMHeadModel .from_pretrained(model_name). cuda()
model.resize_token_embeddings(len(tokenizer))
train_dataset, eval_dataset, test_dataset = load_sentiment_dataset(tokenizer)

training_args = TrainingArguments(output_dir='results', num_train_epochs=4, logging_steps=10,load_best_model_at_end=True,
                                  save_strategy="epoch",per_device_train_batch_size=2, per_device_eval_batch_size=2,warmup_steps=100,
                                  weight_decay=0.01, logging_dir='logs', evaluation_strategy='epoch')
# start training
Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                   'attention_mask': torch.stack([f[1] for f in data]),
                                   'labels': torch.stack([f[0] for f in data])}).train()


# set the model to eval mode
model. eval ()
# run model inference on all test data
original_label, predicted_label, original_text, predicted_text = [],[],[],[]

# iter over all test data
for text, label in tqdm(zip(test_dataset [0], test_dataset [1])):
    # create prompt (in compliance with the one used during training)
    prompt = f'<startoftext>Sentence: {text}\nResearch Design:'
    # generate tokens
    generated = tokenizer(f"{prompt}", return_tensors="pt")
    input_id = torch.tensor(generated['input_ids']).cuda()
    attn_mask = torch.tensor(generated['attention_mask']).cuda()
    # perform prediction
    sample_outputs = model.generate(input_id, do_sample=False, top_k=50, max_length=512,
                                     top_p=0.90,temperature=0, num_return_sequences=0, pad_token_id=tokenizer.eos_token_id)
    # decode the predicted tokens into texts

    pred_text= tokenizer.decode (sample_outputs [0], skip_special_tokens=True)
    # extract the predicted sentiment
    try:
        pred_sentiment = re.findall("Research Design: (.*)", pred_text)[-1]
        pred_sentiment = pred_sentiment.split("<endoftext>")[0]
    except:
        pred_sentiment = "None"

    # append results
    original_label. append(label)
    predicted_label.append(pred_sentiment)
    original_text.append(text)
    predicted_text.append(pred_text)
# transform result into dataframe
df = pd.DataFrame({'original_text': original_text, 'predicted_label': predicted_label,'original_label': original_label,'predicted_text': predicted_text})
# predict the accuracy
print(f1_score(original_label, predicted_label, average= 'macro'))
print(classification_report(original_label, predicted_label))