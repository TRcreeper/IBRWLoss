# IBRWLoss

# A Loss Weighting Algorithm Based on In-batch Positive Sample Rankings for Dense Retrievers Code Repository

This repository stores the code for the paper "A Loss Weighting Algorithm Based on In-batch Positive Sample Rankings for Dense Retrievers".

## 1. Data Preparation
1. **Downloading the data**:
First, you need to download the data from the Passage ranking dataset on the webpage [https://microsoft.github.io/msmarco/Datasets.html](https://microsoft.github.io/msmarco/Datasets.html). Specifically, we will use `qrels.dev.tsv`, `qrels.train.tsv`, and `qidpidtriples.train.full.2.tsv.gz`.
For the remaining data, we will directly utilize the preprocessed data of RocketQA. You can download it from [https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz](https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz). After downloading, please save the data to the `data/msmarco_passage/raw/` directory.
2. **Preliminary data processing**:
Once the data is downloaded and saved in the correct location, run the `data.sh` script for preliminary data processing. 

## 2. Pre - trained Model
The pre - trained model `bert-base-uncased` should be placed in the `model` directory.

## 3. Execution Steps
1. First, run `tokenize.sh`.
2. Then, run `BM25_hard.sh` to train a standard DPR.
3. Run `retrieve_training_set.sh` to retrieve the training set, preparing for the extraction of hard negative samples.
4. Run `stdDPR_hard.sh` to train and evaluate our best model.

## 4. Parameter Modification
The three weighting parameters can be modified in `tevatron/modeling.py`. 
