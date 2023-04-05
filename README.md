## Sciplore: An Exploratory Study of Research Design Classification in Scientific Literature

Selecting an appropriate research design that aligns with the research question is crucial for any study. 
Identifying different designs used in scientific articles can aid researchers in recognizing the strengths and 
limitations of various approaches, assisting them in choosing the most suitable design for their research. However, 
with the overwhelming number of publications each year, it is challenging to keep track of them. Machine learning 
algorithms can be utilized to classify research designs in scientific articles, enabling researchers to locate existing 
techniques in their field and identify gaps in research designs. In this study, we investigated different input levels, 
including abstract, full text, section names, and selected section text, to classify the research design of scientific 
articles categorized as qualitative, quantitative, and mixed.


## Setup

Make sure to install all [required packages](requirements.txt) before running the project. It is recommended to work in
a conda virtual environment.

## Dataset

We utilized a dataset introduced in [Zolfagharian et al. (2019)](https://doi.org/10.1016/j.respol.2019.04.012), where 
authors presented a framework to evaluate 217 peer-reviewed papers in the area of transition studies. Due to data imbalance
we only used 57 articles for three different labels: Qualitative, Quantitative and Mixed (19 each). Full corpus can be 
found [here](data/full_corpus.json) and balanced dataset can be found [here](data/balance_corpus.json).
Full text can be converted into balanced dataset `src/utils/balance_dataset.py`.

In order to preprocess text by removing stop and noisy words, stemming etc. we created a 
[Data Preprocessor](src/utils/data_preprocessor.py) class. To preprocess data we used `src/utils/preprocess_text.py`
and preprocessed data can be found [here](data/dataset.json).



### Different Data types

One of the objectives of this exploratory study was to investigate how different levels of input affect the 
classification results. To accomplish this, we conducted experiments using four levels of input:

1. Abstract: This level uses only the abstract of a scientific article as input.
2. Full Text: This level uses the full text of a scientific article, including the abstract, as input.
3. Section Name: This level uses only the section names of a scientific article, such as Abstract, Introduction, 
Methodology, etc.
4. Section Text: This level uses text from specific sections of a scientific article, such as Introduction, Methodology, 
Conclusion, Discussion, Results, Concluding remarks, Method, and Data.

## Experiments

### 1. Basic Machine Learning Classifiers

To train basic machine learning classifiers like SVM, KNN and Naive Bayes, we created a [config file](configs/base_train.yaml)
which can be updated to select a classifier. Example of config file:

````yaml
data_path: data/dataset.json
feat_type: bow # bow, tfidf
model_type: bayes
params:
  alpha:
    - 0.1
    - 1.0
    - 10

bayes:
  alpha:
    - 0.1
    - 1.0
    - 10
svm:
  C:
    - 0.1


````
To train a classifier we create a [train script](src/train/base_models.py) which takes config path and data type as 
parameters. 

````bash
python src/train/base_models.py -config configs/base_train.yaml -data_type abstract
````

We also created an [experiments script](src/experiments/base.py) which can be used to perform experiments for all 
classifiers against all datatypes at once. 

````bash
python src/experiments/base.py -config configs/base_train.yaml
````

Results from this experiments can be found [here](experiments/base_models_report.csv).



### 2. Zero-shot Classification

We used Natural Language Inference (NLI) based zero-shot classification pipeline from 
[HuggingFace](https://huggingface.co/tasks/zero-shot-classification), which leverages NLI to classify 
unseen data. Similar to previous experiment, we can tune different parameters using a [config file](configs/zs_train.yaml).
To perform inference using zero shot model following command can be used:

````bash
python src/train/zero_shot.py -config configs/zs_train.yaml -data_type abstract
````

Experiment results for zero shot can be found [here](experiments/zero_shot_models_report.csv).


### 3. Few-shot Classification

For our experiment, we utilized [SetFit](https://github.com/huggingface/setfit), a framework for few-shot fine-tuning 
of Sentence Transformers that does not require prompts and is highly efficient.

To train few shot model, run following command:
````bash
python src/train/few_shot.py -config configs/fs_train.yaml -data_type abstract
````

Similarly, we can perform experiments on few shot model using `src/experiments/fs.py` and results from the experiment
can be found in `experiments` folder

### 4. Similarity based Classification

We created a small dataset for labels which can be found [here](data/labels.json). We used this dataset to perform
similarity based experiments. Firstly, we performed cosine similarity over the embeddings of labels and scientific articles.
Secondly, we created unigram and bigram based dictionary from label dataset to classify documents.

Cosine similarity based modelling can be performed using following command:
````bash
python src/train/similarity.py -config configs/sim_train.yaml -data_type abstract
````

Dictionary based modelling can be done using following command:
````bash
python src/train/dictionary.py -config configs/dict_train.yaml -data_type abstract
````

To perform experiments on these methods, we created experiment scripts for both [Cosine similarity based modelling](src/experiments/sim.py)
and [Dictionary based modelling](src/experiments/dict.py) and results from both experiments can be found in `experiments`
folder.


### 5. Citation network based Graph Neural Network (GNN)
We created citation network for 57 scientific articles using [CrossRef API](https://api.crossref.org/swagger-ui/index.html).
We collected [metadata](data/doi2metadata_abs.json) for each citation as well as parent article. We represented [citation
network](data/new_citation_net.json) as a dictionary with keys as DOI of parent article and value as list of DOIs of 
cited articles. Then we create [graph data](data/trans_feat_matrix.pkl) using citation network.

Train GNN model as:
````bash
python src/train/gnn.py -config configs/gnn_train.yaml
````

We also performed hyperparameter tuning using:

````bash
python src/train/gnn_hpt.py -count 5
````

Here `count` represents number of different experiments using parameter combination. Result of this tuning can be found 
[here](https://wandb.ai/hpi-dc/all_features_transductive/sweeps/xtd8upta).

Finally, using the best parameter combination we also performed experiments which included different sentence transformers
and different combination of features.
