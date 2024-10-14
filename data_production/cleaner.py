import pandas as pd
import csv
import re


def clean_csv(file_path, column_to_keep, value_for_second_column):
    try:
        # Create a temporary list to hold the rows
        cleaned_rows = []

        # Read the CSV file and standardize the row lengths
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            max_columns = 0

            for row in reader:
                # Update max_columns if this row has more columns
                if len(row) > max_columns:
                    max_columns = len(row)

                # Normalize the row to the current max_columns
                row += [''] * (max_columns - len(row))
                cleaned_rows.append(row)

        # Convert cleaned rows to a DataFrame
        df = pd.DataFrame(cleaned_rows)

        # Select columns: the specified column and the second column if it exists
        column_to_keep_index = column_to_keep
        columns_to_keep = [column_to_keep_index]

        if df.shape[1] > 1:
            columns_to_keep.append(column_to_keep_index + 1)

        df_cleaned = df.iloc[:, columns_to_keep]

        # Remove the first row (headers)
        df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True)

        # Populate the second column with the specified value, if applicable
        if len(df_cleaned.columns) > 1:
            df_cleaned.iloc[:, 1] = value_for_second_column

        # Set column headers
        df_cleaned.columns = ['Hashtag', 'Label'] if len(df_cleaned.columns) > 1 else ['Hashtag']

        # Save the cleaned DataFrame to a temporary file
        temp_file_path = file_path
        df_cleaned.to_csv(temp_file_path, index=False)

        return temp_file_path

    except Exception as e:
        print(f"An error occurred during cleaning: {e}")
        return None


def sanitize_data(input_file):
    try:
        # Load CSV into a pandas DataFrame
        df = pd.read_csv(input_file)

        # Ensure 'Hashtag' column is of type string
        df['Hashtag'] = df['Hashtag'].astype(str)

        # Convert to lowercase
        df['Hashtag'] = df['Hashtag'].str.lower()

        # Remove punctuation and spaces
        df['Hashtag'] = df['Hashtag'].apply(lambda x: re.sub(r'[^\w]', '', x))

        # Remove rows with more than 50 characters
        df = df[df['Hashtag'].str.len() <= 50]

        # Remove any rows where 'Hashtag' is empty or not a string
        df = df[df['Hashtag'].str.strip().astype(bool)]

        # Optional: Remove any rows where 'Label' is not 0 or 1
        df = df[df['Label'].isin([0, 1])]

        # Define the final output file path with 'D' prefix
        output_file_path = 'D' + input_file

        # Save the sanitized DataFrame to a new CSV file
        df.to_csv(output_file_path, index=False)
        print(f"Sanitized data saved to '{output_file_path}'")

    except Exception as e:
        print(f"An error occurred during sanitization: {e}")


if __name__ == "__main__":
    # Get user inputs
    file_path = input("Enter the path to the CSV file: ")
    column_to_keep = int(input("Enter the column index to keep: "))
    value_for_second_column = input("Enter the value to populate in the second column: ")

    try:
        # Clean the CSV file
        temp_file_path = clean_csv(file_path, column_to_keep, value_for_second_column)

        if temp_file_path:
            # Sanitize the cleaned CSV
            sanitize_data(temp_file_path)

            print(f"Process completed. Sanitized file saved with 'D' prefix.")
        else:
            print("Cleaning process failed. No sanitization performed.")

    except Exception as e:
        print(f"An error occurred: {e}")
