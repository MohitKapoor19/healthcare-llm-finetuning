import pandas as pd
import json

def csv_to_jsonl(input_csv, output_jsonl):
    """Convert CSV file to JSONL format."""
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv)
        
        # Convert to JSONL format - one JSON object per line
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for index, row in df.iterrows():
                data_dict = {
                    "text": row['text'],
                    "label_string": row['label_string']
                }
                json.dump(data_dict, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Successfully converted {input_csv} to JSONL format: {output_jsonl}")
        print(f"Total records: {len(df)}")
        
    except Exception as e:
        print(f"Error converting {input_csv} to JSONL: {str(e)}")

def jsonl_to_csv(input_jsonl, output_csv):
    """Convert JSONL file back to CSV format."""
    try:
        data_list = []
        
        # Read JSONL file
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data_dict = json.loads(line)
                    data_list.append(data_dict)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(data_list)
        df.to_csv(output_csv, index=False)
        
        print(f"Successfully converted {input_jsonl} to CSV format: {output_csv}")
        print(f"Total records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"Error converting {input_jsonl} to CSV: {str(e)}")

# Main conversion process
print("=== CSV to JSONL and back to CSV Conversion ===\n")

# Step 1: Convert CSV files to JSONL format
print("--- Step 1: Converting CSV files to JSONL ---")

print("\n1. Converting Train Dataset:")
csv_to_jsonl(
    input_csv="processed-train-dataset.csv",
    output_jsonl="train.jsonl"
)

print("\n2. Converting Test Dataset:")
csv_to_jsonl(
    input_csv="processed-test-dataset.csv",
    output_jsonl="test.jsonl"
)

# Step 2: Convert JSONL files back to CSV format
print("\n--- Step 2: Converting JSONL files back to CSV ---")

print("\n1. Converting train.jsonl to CSV:")
jsonl_to_csv(
    input_jsonl="train.jsonl",
    output_csv="train_from_jsonl.csv"
)

print("\n2. Converting test.jsonl to CSV:")
jsonl_to_csv(
    input_jsonl="test.jsonl",
    output_csv="test_from_jsonl.csv"
)

print("\n=== All conversions completed! ===")

# Verify the data integrity
print("\n--- Data Verification ---")
try:
    original_train = pd.read_csv("processed-train-dataset.csv")
    converted_train = pd.read_csv("train_from_jsonl.csv")
    
    original_test = pd.read_csv("processed-test-dataset.csv")
    converted_test = pd.read_csv("test_from_jsonl.csv")
    
    print(f"Original train shape: {original_train.shape}")
    print(f"Converted train shape: {converted_train.shape}")
    print(f"Train data integrity: {'✓ PASSED' if original_train.shape == converted_train.shape else '✗ FAILED'}")
    
    print(f"Original test shape: {original_test.shape}")
    print(f"Converted test shape: {converted_test.shape}")
    print(f"Test data integrity: {'✓ PASSED' if original_test.shape == converted_test.shape else '✗ FAILED'}")
    
except Exception as e:
    print(f"Error during verification: {str(e)}")
