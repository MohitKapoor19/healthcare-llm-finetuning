import pandas as pd
import json

def csv_to_json_format(input_csv, output_file, file_format='json'):
    """
    Convert CSV file to JSON or JSONL format.
    
    Args:
        input_csv (str): Path to input CSV file
        output_file (str): Path to output JSON/JSONL file
        file_format (str): 'json' for JSON array format, 'jsonl' for JSON Lines format
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv)
        
        # Convert to list of dictionaries
        data_list = []
        for index, row in df.iterrows():
            data_dict = {
                "text": row['text'],
                "label_string": row['label_string']
            }
            data_list.append(data_dict)
        
        # Save based on format
        if file_format.lower() == 'jsonl':
            # JSONL format - one JSON object per line
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data_list:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            print(f"Successfully converted {input_csv} to JSONL format: {output_file}")
        else:
            # JSON format - array of objects
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=2, ensure_ascii=False)
            print(f"Successfully converted {input_csv} to JSON format: {output_file}")
        
        print(f"Total records: {len(data_list)}")
        print(f"Sample record: {data_list[0] if data_list else 'No data'}")
        
    except Exception as e:
        print(f"Error processing {input_csv}: {str(e)}")

# Convert the files
print("=== Converting CSV files to JSON/JSONL format ===\n")

# Convert train dataset to JSON
print("--- Processing Train Dataset ---")
csv_to_json_format(
    input_csv="processed-train-dataset.csv",
    output_file="train.json",
    file_format="json"
)

print("\n--- Processing Test Dataset ---")
# Convert test dataset to JSONL
csv_to_json_format(
    input_csv="processed-test-dataset.csv", 
    output_file="test.jsonl",
    file_format="jsonl"
)

print("\n=== Conversion completed! ===")
