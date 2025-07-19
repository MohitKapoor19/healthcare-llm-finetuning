import pandas as pd
import json
import numpy as np # numpy is imported but not directly used in the provided snippet's logic

# Placeholder for the mapping.json file
# In a real scenario, you would load this directly:
with open('mapping.json', 'r') as f:
    label_mapping = json.load(f)

# Invert the dictionary for numerical label to string label conversion
string_to_label_map = {v: k for k, v in label_mapping.items()}

# Placeholder for symptom-disease-train-dataset.csv
# In a real scenario, you would load this directly:
train_df = pd.read_csv('symptom-disease-train-dataset.csv')

# Placeholder for symptom-disease-test-dataset.csv
# In a real scenario, you would load this directly:
test_df = pd.read_csv('symptom-disease-test-dataset.csv')
# The line 'test_df = pd.DataFrame(test_df)' is redundant here as test_df is already a DataFrame
# If you meant to re-create it with specific data, ensure 'test_data' is defined.
# Assuming you just want to load and modify the existing CSVs:
# test_df = pd.DataFrame(test_df) # This line can be removed if test_df is already loaded correctly from CSV

# Now you can apply the mapping to your DataFrames
train_df['label_string'] = train_df['label'].map(string_to_label_map)
test_df['label_string'] = test_df['label'].map(string_to_label_map)

print("Train DataFrame with string labels:")
print(train_df.head())
print("\nTest DataFrame with string labels:")
print(test_df.head())

# --- Option to Save DataFrames ---

# Save the modified training DataFrame to a new CSV file
# index=False prevents pandas from writing the DataFrame index as a column in the CSV
output_train_file = 'symptom-disease-train-dataset-with_strings.csv'
train_df.to_csv(output_train_file, index=False)
print(f"\nTrain DataFrame saved to: {output_train_file}")

# Save the modified testing DataFrame to a new CSV file
output_test_file = 'symptom-disease-test-dataset-with_strings.csv'
test_df.to_csv(output_test_file, index=False)
print(f"Test DataFrame saved to: {output_test_file}")

# You can also save them to Excel, JSON, etc., if needed:
# train_df.to_excel('symptom-disease-train-dataset-with_strings.xlsx', index=False)
# test_df.to_json('symptom-disease-test-dataset-with_strings.json', orient='records', indent=4)