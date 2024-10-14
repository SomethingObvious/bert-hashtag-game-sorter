# This is the old version. It works but will provide less accurate results.

from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import json
import os
from torch.amp import GradScaler, autocast
from accelerate import Accelerator

# Set cache directory
cache_dir = 'C:\\Users\\paulz\\.cache\\huggingface\\transformers'
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Device configuration using accelerate
accelerator = Accelerator()
device = accelerator.device

# Output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Load tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', clean_up_tokenization_spaces=True)
    model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)
    model.to(device)  # Move model to the appropriate device
except Exception as e:
    print(f"Error loading tokenizer or model: {e}")

# Tokenize function
def tokenize_function(examples):
    texts = [str(example) for example in examples['text']]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# Custom Dataset class for handling the tokenized data
class HashtagDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            if isinstance(val[idx], str):
                raise ValueError(f"Invalid data type 'str' found in encodings for key {key} at index {idx}")
            item[key] = torch.tensor(val[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Directory containing the training data
data_dir = 'training_data'

# Load dataset from the folder
train_file = os.path.join(data_dir, 'train1.csv')
test_file = os.path.join(data_dir, 'test1.csv')

try:
    dataset = load_dataset('csv', data_files={'train': train_file, 'test': test_file})
except Exception as e:
    print(f"Error loading dataset: {e}")

# Tokenize the dataset
try:
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['Hashtag'])
except Exception as e:
    print(f"Error during tokenization: {e}")

# Convert tokenized datasets to lists
def convert_to_list(dataset):
    encodings = {}
    for column in dataset.column_names:
        encodings[column] = dataset[column]
    return encodings

train_encodings = convert_to_list(tokenized_datasets['train'])
test_encodings = convert_to_list(tokenized_datasets['test'])

# Prepare datasets
train_labels = list(tokenized_datasets['train']['Label'])
test_labels = list(tokenized_datasets['test']['Label'])

train_dataset = HashtagDataset(train_encodings, train_labels)
test_dataset = HashtagDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Increased batch size
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Added L2 regularization
scaler = GradScaler()  # Mixed precision scaler

# Prepare the model and dataloaders with accelerator
model, train_loader, test_loader = accelerator.prepare(model, train_loader, test_loader)

# Training loop
early_stopping_patience = 2
best_val_loss = float('inf')
no_improvement = 0

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/3"):
        batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
        optimizer.zero_grad()

        with autocast('cuda'):  # Updated usage of autocast
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss}")

    # Validation
    model.eval()
    total_eval_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            with autocast('cuda'):  # Updated usage of autocast
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                total_eval_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += torch.sum(predictions == batch['labels'])

    avg_eval_loss = total_eval_loss / len(test_loader)
    eval_accuracy = correct_predictions.double() / len(test_dataset)
    print(f"Epoch {epoch + 1} - Average Validation Loss: {avg_eval_loss}")
    print(f"Epoch {epoch + 1} - Validation Accuracy: {eval_accuracy}")

    # Early stopping
    if avg_eval_loss < best_val_loss:
        best_val_loss = avg_eval_loss
        no_improvement = 0
        # Save the best model
        model.save_pretrained(os.path.join(output_dir, 'best_model'))
    else:
        no_improvement += 1
        if no_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Save the final model
model.save_pretrained(os.path.join(output_dir, 'final_model'))

# Write evaluation results to a JSON file
eval_results = {
    'eval_loss': avg_eval_loss,
    'eval_accuracy': eval_accuracy.item()
}

with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
    json.dump(eval_results, f, indent=4)

print("Training completed.")
