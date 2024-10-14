import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
script_dir = os.path.dirname(__file__)
input_file = os.path.join(script_dir, '../training_data/combined_output.csv')
output_dir = os.path.join(script_dir, '../training_data')

# Flag to control sampling
use_max_samples = True  # Set to False to use the entire dataset

# Maximum number of samples to take
max_samples = 100000000000000000000000

# Load the dataset
df = pd.read_csv(input_file)

# Sample a maximum of 10,000 rows from the dataset if the flag is set
if use_max_samples and len(df) > max_samples:
    df = df.sample(n=max_samples, random_state=41)

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)  # 90% train, 10% test

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the train and test CSV files
train_csv_path = os.path.join(output_dir, 'train.csv')
test_csv_path = os.path.join(output_dir, 'test.csv')

print(f"Input file path: {input_file}")
print(f"Output directory: {output_dir}")
print(f"Saving train CSV to: {train_csv_path}")
print(f"Saving test CSV to: {test_csv_path}")

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)
