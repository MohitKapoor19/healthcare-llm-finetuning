import pandas as pd
import re

def convert_symptoms_to_natural_language(symptoms_text):
    """
    Convert comma-separated symptoms to natural language.
    """
    # Check if it's already natural language (contains common words like 'I', 'have', 'been', etc.)
    natural_language_indicators = [
        'I have', 'I\'ve', 'I am', 'I\'m', 'My', 'The', 'Signs and symptoms',
        'Symptoms', 'A', 'An', 'This', 'These', 'Many', 'Some', 'When',
        'During', 'After', 'Before', 'Usually', 'Often', 'Sometimes'
    ]
    
    # If it already contains natural language indicators, return as is
    for indicator in natural_language_indicators:
        if indicator in symptoms_text:
            return symptoms_text
    
    # If it's comma-separated symptoms, convert to natural language
    if ',' in symptoms_text and not any(char in symptoms_text for char in '.!?'):
        symptoms = [symptom.strip() for symptom in symptoms_text.split(',')]
        
        # Clean up symptom names - replace underscores with spaces
        cleaned_symptoms = []
        for symptom in symptoms:
            # Replace underscores with spaces
            cleaned = symptom.replace('_', ' ')
            # Remove parentheses content like (typhos)
            cleaned = re.sub(r'\([^)]*\)', '', cleaned).strip()
            if cleaned:  # Only add non-empty symptoms
                cleaned_symptoms.append(cleaned)
        
        # Convert to natural language
        if len(cleaned_symptoms) == 1:
            return f"I have been experiencing {cleaned_symptoms[0]}."
        elif len(cleaned_symptoms) == 2:
            return f"I have been experiencing {cleaned_symptoms[0]} and {cleaned_symptoms[1]}."
        else:
            # Join all but last with commas, and last with 'and'
            symptom_list = ', '.join(cleaned_symptoms[:-1]) + f', and {cleaned_symptoms[-1]}'
            return f"I have been experiencing {symptom_list}."
    
    # If it doesn't match either pattern, return as is
    return symptoms_text

def process_csv_for_llm_fine_tuning(input_file, output_file):
    """
    Process the CSV file to convert one-hot encoded symptoms to natural language.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        print(f"Processing {input_file}")
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Apply the conversion to the text column
        print("Converting symptoms to natural language...")
        df['text'] = df['text'].apply(convert_symptoms_to_natural_language)
        
        # Save the processed dataset
        df.to_csv(output_file, index=False)
        
        print(f"Processed dataset saved to: {output_file}")
        print(f"Final dataset shape: {df.shape}")
        
        # Count how many entries were converted
        converted_count = 0
        natural_count = 0
        
        for text in df['text']:
            if 'I have been experiencing' in str(text):
                converted_count += 1
            else:
                natural_count += 1
        
        print(f"Conversion Summary:")
        print(f"Entries converted from symptoms to natural language: {converted_count}")
        print(f"Entries already in natural language: {natural_count}")
        print(f"Total entries: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None

# Process both files
print("=== Converting One-Hot Encoded Symptoms to Natural Language ===\n")

# Process training dataset
print("--- Processing Training Dataset ---")
train_df = process_csv_for_llm_fine_tuning(
    input_file="processed2-train.csv",
    output_file="llm_ready_train_dataset.csv"
)

print("\n--- Processing Test Dataset ---")
test_df = process_csv_for_llm_fine_tuning(
    input_file="processed2-test.csv",
    output_file="llm_ready_test_dataset.csv"
)

print("\n=== Processing completed! ===")

# Show some examples from both datasets
if train_df is not None and test_df is not None:
    print("\n--- Sample Examples ---")
    print("\nTrain Dataset Examples:")
    for i in range(min(3, len(train_df))):
        if 'I have been experiencing' in str(train_df.iloc[i]['text']):
            print(f"Text: {train_df.iloc[i]['text'][:100]}...")
            print(f"Label: {train_df.iloc[i]['label_string']}")
            print("-" * 50)
    
    print("\nTest Dataset Examples:")
    for i in range(min(3, len(test_df))):
        if 'I have been experiencing' in str(test_df.iloc[i]['text']):
            print(f"Text: {test_df.iloc[i]['text'][:100]}...")
            print(f"Label: {test_df.iloc[i]['label_string']}")
            print("-" * 50)
