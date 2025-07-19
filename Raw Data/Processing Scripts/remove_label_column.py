import pandas as pd

def remove_label_column(input_file, output_file):
    """Remove the 'label' column from a CSV file and save the result."""
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Check if 'label' column exists
        if 'label' in df.columns:
            # Remove the 'label' column
            df_cleaned = df.drop('label', axis=1)
            
            # Save the cleaned dataframe
            df_cleaned.to_csv(output_file, index=False)
            print(f"Successfully removed 'label' column from {input_file}")
            print(f"Cleaned data saved to {output_file}")
            print(f"Original shape: {df.shape}")
            print(f"New shape: {df_cleaned.shape}")
            print(f"Remaining columns: {list(df_cleaned.columns)}")
        else:
            print(f"No 'label' column found in {input_file}")
            
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

# Process both files
files_to_process = [
    ("processed-test-dataset.csv", "processed-test-dataset.csv"),
    ("processed-train-dataset.csv", "processed-train-dataset.csv")
]

for input_file, output_file in files_to_process:
    print(f"\n--- Processing {input_file} ---")
    remove_label_column(input_file, output_file)
