# Hashtag Classification with BERT

This repository implements a BERT-based model to classify hashtags as either game-related or non-game-related. It leverages state-of-the-art NLP techniques to effectively handle the classification task.

## Key Features
- **BERT Model**: Utilizes `bert-large-uncased` for binary classification of hashtags.
- **Data Handling**: Loads hashtag data from CSV files for training and testing.
- **Oversampling**: Implements a method to oversample the minority class, addressing class imbalance in the dataset.
- **Evaluation Metrics**: Calculates various performance metrics, including:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC

## Requirements
To run this project, you need to have the following packages installed:
- torch
- transformers
- datasets
- sklearn
- accelerate
- numpy

## Usage
1. Set the paths to your dataset CSV files in the script.
2. Run the script to train the model and evaluate its performance.
