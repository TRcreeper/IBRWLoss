# IBRWLoss

# A Loss Weighting Algorithm Based on In-batch Positive Sample Rankings for Dense Retrievers Code Repository

This repository stores the code for the paper "A Loss Weighting Algorithm Based on In-batch Positive Sample Rankings for Dense Retrievers".

## 1. Data Preparation
1. Download the data to `data/msmarco_passage/raw/`.
2. Run `data.sh` for preliminary data processing.

## 2. Pre - trained Model
The pre - trained model `bert - base - uncased` should be placed in the `model` directory.

## 3. Execution Steps
1. First, run `tokenize.sh`.
2. Then, run `BM25_hard.sh` to train a standard DPR.
3. Run `retrieve_training_set.sh` to retrieve the training set, preparing for the extraction of hard negative samples.
4. Run `stdDPR_hard.sh` to train and evaluate our best model.

## 4. Parameter Modification
The three weighting parameters can be modified in `tevatron/modeling.py`. 
