import os
import json
import torch
import numpy as np
import random
from collections import Counter
from torch.utils.data import Dataset
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    log_loss, brier_score_loss
)
from sklearn.model_selection import KFold
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Features, Value
from accelerate import Accelerator

# Set cache directory for transformers
cache_dir = 'C:\\Users\\paulz\\.cache\\huggingface\\transformers'
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', clean_up_tokenization_spaces=True)
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)

# Initialize accelerator
accelerator = Accelerator()

# Tokenize function
def tokenize_function(examples):
    texts = [str(example) for example in examples['text']]
    encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors=None)
    return {key: list(val) for key, val in encodings.items()}

# Define dataset features to ensure consistent column names and types
features = Features({
    'text': Value('string'),
    'labels': Value('int64'),
})

# Load dataset
data_dir = 'training_data'
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')
dataset = load_dataset(
    'csv',
    data_files={'train': train_file, 'test': test_file},
    column_names=['text', 'labels'],  # Explicitly set column names
    features=features,  # Ensure the data types match
    skiprows=1  # Skip the first row if it contains headers
)

print("Train dataset length:", len(dataset['train']))
print("Test dataset length:", len(dataset['test']))

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])

print("Tokenized train dataset length:", len(tokenized_datasets['train']))
print("Tokenized test dataset length:", len(tokenized_datasets['test']))

# Define a PyTorch Dataset class
class HashtagDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = [int(label) for label in labels]  # Ensure labels are integers
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # Convert to tensor

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Define a function to oversample the minority class
def oversample_minority_class(encodings, labels):
    class_counts = Counter(labels)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    samples_needed = class_counts[majority_class] - class_counts[minority_class]

    if samples_needed <= 0:
        return encodings, labels

    minority_indices = [i for i, label in enumerate(labels) if label == minority_class]
    if not minority_indices:
        return encodings, labels

    oversampled_indices = minority_indices * (samples_needed // len(minority_indices)) + random.choices(
        minority_indices, k=samples_needed % len(minority_indices)
    )

    new_encodings = {key: val + [val[i] for i in oversampled_indices] for key, val in encodings.items()}
    new_labels = labels + [minority_class] * samples_needed

    return new_encodings, new_labels

def data_collator(features):
    batch = {
        key: torch.stack([f[key] for f in features]) for key in features[0] if key != 'labels'
    }
    batch['labels'] = torch.tensor([f['labels'] for f in features])
    return batch

# Define the metrics computation function
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)

    confusion = confusion_matrix(labels, preds)
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    auroc = roc_auc_score(labels, probs[:, 1])
    auprc = average_precision_score(labels, probs[:, 1])
    logloss = log_loss(labels, probs)
    brier = brier_score_loss(labels, probs[:, 1])

    return {
        'confusion_matrix': confusion.tolist(),
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'mcc': mcc,
        'auroc': auroc,
        'auprc': auprc,
        'log_loss': logloss,
        'brier_score_loss': brier
    }

def single_run_training(tokenized_datasets, model):
    encodings = {key: tokenized_datasets['train'][key] for key in tokenized_datasets['train'].features.keys()}
    labels = tokenized_datasets['train']['labels']
    encodings, labels = oversample_minority_class(encodings, labels)

    train_dataset = HashtagDataset(encodings, labels)
    eval_dataset = HashtagDataset(
        {key: tokenized_datasets['test'][key] for key in tokenized_datasets['test'].features.keys()},
        tokenized_datasets['test']['labels']
    )

    training_args = TrainingArguments(
        output_dir='./results/single_run',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs/single_run',
        logging_steps=10,
        load_best_model_at_end=True,
        fp16=True
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,  # Use the modified data collator function
        callbacks=[early_stopping]
    )

    # Prepare for training with accelerator
    model, train_dataset, eval_dataset = accelerator.prepare(model, train_dataset, eval_dataset)
    trainer.model = model

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Save the trained model and tokenizer
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_tokenizer')

    # Save evaluation results
    results_folder = 'training_results'
    os.makedirs(results_folder, exist_ok=True)

    # Save evaluation results
    with open(os.path.join(results_folder, 'single_run_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)

    return eval_results

# Cross-validation function
def cross_validation(tokenized_datasets, model):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # Define a proper data collator function
    def collate_fn(features):
        batch = {
            key: torch.stack([f[key] for f in features]) for key in features[0] if key != 'labels'
        }
        batch['labels'] = torch.tensor([f['labels'] for f in features])
        return batch

    for fold, (train_idx, eval_idx) in enumerate(kfold.split(tokenized_datasets['train'])):
        print(f"Fold {fold + 1}")

        train_encodings = {key: [tokenized_datasets['train'][key][i] for i in train_idx] for key in tokenized_datasets['train'].features.keys()}
        train_labels = [tokenized_datasets['train']['labels'][i] for i in train_idx]
        eval_encodings = {key: [tokenized_datasets['train'][key][i] for i in eval_idx] for key in tokenized_datasets['train'].features.keys()}
        eval_labels = [tokenized_datasets['train']['labels'][i] for i in eval_idx]

        train_encodings, train_labels = oversample_minority_class(train_encodings, train_labels)

        train_dataset = HashtagDataset(train_encodings, train_labels)
        eval_dataset = HashtagDataset(eval_encodings, eval_labels)

        training_args = TrainingArguments(
            output_dir=f'./results/fold_{fold + 1}',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f'./logs/fold_{fold + 1}',
            logging_steps=10,
            load_best_model_at_end=True,
            fp16=True
        )

        early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,  # Use the defined data collator function
            callbacks=[early_stopping]
        )

        # Prepare for training with accelerator
        model, train_dataset, eval_dataset = accelerator.prepare(model, train_dataset, eval_dataset)
        trainer.model = model

        trainer.train()
        eval_results = trainer.evaluate()

        # Ensure the training_results folder exists
        results_folder = 'training_results'
        os.makedirs(results_folder, exist_ok=True)

        # Save evaluation results for this fold
        with open(os.path.join(results_folder, f'fold_{fold + 1}_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=4)

        results.append(eval_results)

    return results


# Main function
def main():
    mode = input("Choose mode (1: Single Run, 2: Cross-Validation): ").strip()
    if mode == '1':
        print("Starting single run training...")
        single_run_results = single_run_training(tokenized_datasets, model)
        print("Single run results:", single_run_results)
    elif mode == '2':
        print("Starting cross-validation...")
        cross_val_results = cross_validation(tokenized_datasets, model)
        print("Cross-validation results:", cross_val_results)
    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()
