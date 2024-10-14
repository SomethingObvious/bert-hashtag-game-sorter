import csv
import os
import string
from glob import glob


def clean_name(name):
    # Remove punctuation and spaces, and convert to lowercase
    return ''.join(c.lower() for c in name if c not in string.punctuation and c != ' ')


def process_and_append_csv(input_file, writer, seen):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)

        # Skip header
        next(reader, None)

        for row in reader:
            if len(row) == 0 or any(len(str(element).strip()) == 0 for element in row):
                # Skip empty rows or rows with empty fields
                continue

            try:
                name = row[0]
                cleaned_name = clean_name(name)

                if 3 < len(cleaned_name) < 50:
                    row_key = tuple([cleaned_name] + row[1:])
                    if row_key not in seen:
                        seen[row_key] = row  # Store the row if not seen
                    elif row[1] != '0' and seen[row_key][1] == '0':
                        # Replace the previous row if it has label 0 and the current one doesn't
                        seen[row_key] = row
            except Exception as e:
                # Skip rows that cause errors
                print(f"Error processing row {row}: {e}")
                continue


def write_to_output_file(output_file, seen):
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        for row in seen.values():
            writer.writerow(row)


def append_all_csvs_with_capital_D(output_file):
    # Find all CSV files starting with 'D'
    csv_files = glob('Data*.csv')
    seen = {}

    for csv_file in csv_files:
        process_and_append_csv(csv_file, writer=None, seen=seen)

    write_to_output_file(output_file, seen)


# Example usage
output_file = '../training_data/combined_output.csv'
append_all_csvs_with_capital_D(output_file)
