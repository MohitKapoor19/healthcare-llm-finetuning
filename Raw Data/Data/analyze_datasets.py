import pandas as pd
import os
from collections import Counter
import numpy as np

def analyze_dataset(file_path):
    """Analyze a dataset file and return statistics."""
    try:
        df = pd.read_csv(file_path)
        
        # Basic statistics
        stats = {
            'file_name': os.path.basename(file_path),
            'total_samples': len(df),
            'columns': list(df.columns),
            'unique_labels': df['label_string'].nunique() if 'label_string' in df.columns else 0,
            'label_distribution': dict(df['label_string'].value_counts().head(10)) if 'label_string' in df.columns else {},
            'avg_text_length': df['text'].str.len().mean() if 'text' in df.columns else 0,
            'min_text_length': df['text'].str.len().min() if 'text' in df.columns else 0,
            'max_text_length': df['text'].str.len().max() if 'text' in df.columns else 0,
            'has_natural_language': False,
            'has_comma_symptoms': False,
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Check text format
        if 'text' in df.columns:
            sample_texts = df['text'].head(10).tolist()
            natural_language_count = 0
            comma_symptoms_count = 0
            
            for text in sample_texts:
                text_str = str(text)
                if any(indicator in text_str for indicator in ['I have', 'I\'ve been', 'Signs and symptoms', 'The']):
                    natural_language_count += 1
                elif ',' in text_str and not any(char in text_str for char in '.!?'):
                    comma_symptoms_count += 1
            
            stats['has_natural_language'] = natural_language_count > 5
            stats['has_comma_symptoms'] = comma_symptoms_count > 5
        
        return stats
    
    except Exception as e:
        return {'file_name': os.path.basename(file_path), 'error': str(e)}

def main():
    """Analyze all training and test datasets."""
    
    # Define file paths
    train_files = [
        'Training Data/llm_ready_train_dataset.csv',
        'Training Data/processed1-train-dataset.csv', 
        'Training Data/processed2-train.csv'
    ]
    
    test_files = [
        'Testing Data/llm_ready_test_dataset.csv',
        'Testing Data/processed1-test-dataset.csv',
        'Testing Data/processed2-test.csv'
    ]
    
    print("=" * 80)
    print("HEALTHCARE DATASET ANALYSIS FOR LLM FINE-TUNING")
    print("=" * 80)
    
    print("\n📊 TRAINING DATASETS ANALYSIS")
    print("-" * 50)
    
    train_stats = []
    for file_path in train_files:
        if os.path.exists(file_path):
            stats = analyze_dataset(file_path)
            train_stats.append(stats)
            
            print(f"\n🗂️  {stats['file_name']}")
            if 'error' in stats:
                print(f"   ❌ Error: {stats['error']}")
                continue
                
            print(f"   📈 Total Samples: {stats['total_samples']:,}")
            print(f"   🏷️  Unique Labels: {stats['unique_labels']}")
            print(f"   📝 Avg Text Length: {stats['avg_text_length']:.0f} chars")
            print(f"   📏 Text Length Range: {stats['min_text_length']} - {stats['max_text_length']}")
            print(f"   🗣️  Natural Language: {'✅' if stats['has_natural_language'] else '❌'}")
            print(f"   📋 Comma Symptoms: {'✅' if stats['has_comma_symptoms'] else '❌'}")
            
            if stats['missing_values']:
                missing = {k: v for k, v in stats['missing_values'].items() if v > 0}
                if missing:
                    print(f"   ⚠️  Missing Values: {missing}")
            
            print(f"   🔝 Top Labels: {list(stats['label_distribution'].keys())[:5]}")
    
    print("\n📊 TEST DATASETS ANALYSIS")
    print("-" * 50)
    
    test_stats = []
    for file_path in test_files:
        if os.path.exists(file_path):
            stats = analyze_dataset(file_path)
            test_stats.append(stats)
            
            print(f"\n🗂️  {stats['file_name']}")
            if 'error' in stats:
                print(f"   ❌ Error: {stats['error']}")
                continue
                
            print(f"   📈 Total Samples: {stats['total_samples']:,}")
            print(f"   🏷️  Unique Labels: {stats['unique_labels']}")
            print(f"   📝 Avg Text Length: {stats['avg_text_length']:.0f} chars")
            print(f"   🗣️  Natural Language: {'✅' if stats['has_natural_language'] else '❌'}")
            print(f"   📋 Comma Symptoms: {'✅' if stats['has_comma_symptoms'] else '❌'}")
    
    print("\n" + "=" * 80)
    print("🎯 RECOMMENDATIONS FOR LLM FINE-TUNING")
    print("=" * 80)
    
    # Find best datasets
    best_train = None
    best_test = None
    
    for stats in train_stats:
        if 'error' not in stats and stats['has_natural_language'] and stats['total_samples'] > 1000:
            if best_train is None or stats['total_samples'] > best_train['total_samples']:
                best_train = stats
    
    for stats in test_stats:
        if 'error' not in stats and stats['has_natural_language'] and stats['total_samples'] > 100:
            if best_test is None or stats['total_samples'] > best_test['total_samples']:
                best_test = stats
    
    if best_train:
        print(f"\n🏆 RECOMMENDED TRAINING DATASET: {best_train['file_name']}")
        print(f"   ✅ {best_train['total_samples']:,} samples with natural language format")
        print(f"   ✅ {best_train['unique_labels']} unique medical conditions")
        print(f"   ✅ Average text length: {best_train['avg_text_length']:.0f} characters")
        
    if best_test:
        print(f"\n🏆 RECOMMENDED TEST DATASET: {best_test['file_name']}")
        print(f"   ✅ {best_test['total_samples']:,} samples for evaluation")
        
    print(f"\n🤖 RECOMMENDED MODEL: microsoft/DialoGPT-medium or google/flan-t5-base")
    print(f"   ✅ Good for medical text classification and generation")
    print(f"   ✅ Supports LoRA fine-tuning efficiently")
    print(f"   ✅ Reasonable size for healthcare applications")
    
    print(f"\n📋 DATASET READINESS CHECK:")
    if best_train and best_test:
        print(f"   ✅ Training data ready: {best_train['file_name']}")
        print(f"   ✅ Test data ready: {best_test['file_name']}")
        print(f"   ✅ Format consistency: Both use natural language")
        print(f"   ✅ Ready for fine-tuning!")
    else:
        print(f"   ❌ No suitable datasets found for fine-tuning")
        
    return best_train, best_test

if __name__ == "__main__":
    best_train, best_test = main()
