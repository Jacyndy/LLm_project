# LLM Project

## Project Task
This project aims to perform sentiment analysis on imdb data from hugging face dataset using LLM (NLP).

## Dataset
The datasets was sourced from hugging face dataset. It contains data on movie reviews on imdb and labels encoded as 1 for positive and 0 for negative. There are 25000 data points for bith train and test data respectively. 

## Pre-trained Model
metadata
library_name: transformers
base_model: distilbert-base-uncased
tags:
  - generated_from_trainer
metrics:
  - accuracy
  - f1
model-index:
  - name: finetuning-sentiment-model-1000-samples
  - finetuning-sentiment-model-1000-samples
This model is a fine-tuned version of distilbert-base-uncased on an imdb movie review dataset. It achieves the following results on the evaluation set:

Loss: 0.3777

## Performance Metrics
The performance metrics selected for this is accuracy and f1. the results are below:
Accuracy: 0.86
F1: 0.8608

## Hyperparameters
Training hyperparameters
The following hyperparameters were used during training:
learning_rate: 2e-05
train_batch_size: 16
eval_batch_size: 16
seed: 42
optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
lr_scheduler_type: linear
num_epochs: 2
Training results
Framework versions
Transformers 4.48.3
Pytorch 2.5.1+cu124
Datasets 3.3.2
Tokenizers 0.21.0

