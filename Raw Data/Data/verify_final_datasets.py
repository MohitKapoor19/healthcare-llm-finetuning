"""
Verify Final Combined Datasets
=============================
"""

import json
import os

def verify_dataset(file_path, dataset_name):
    """Verify a dataset file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n✅ {dataset_name}")
        print(f"   📊 Total samples: {len(data):,}")
        
        if data:
            # Check structure
            sample = data[0]
            print(f"   🔧 Structure: {list(sample.keys())}")
            print(f"   📝 Sample text: {sample.get('text', '')[:100]}...")
            print(f"   🏷️  Sample label: {sample.get('label_string', 'N/A')}")
            
            # Check unique labels
            labels = [item.get('label_string', '') for item in data]
            unique_labels = len(set(labels))
            print(f"   🎯 Unique labels: {unique_labels}")
            
            # Text length stats
            lengths = [len(item.get('text', '')) for item in data]
            avg_length = sum(lengths) / len(lengths)
            print(f"   📏 Avg text length: {avg_length:.0f} chars")
            
            # Top conditions
            from collections import Counter
            label_counts = Counter(labels)
            top_5 = label_counts.most_common(5)
            print(f"   🔝 Top conditions: {[f'{label}({count})' for label, count in top_5]}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error verifying {dataset_name}: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("🔍 VERIFYING FINAL COMBINED DATASETS")
    print("=" * 60)
    
    base_dir = "FInal Processed Data"
    
    # Verify training dataset
    train_file = os.path.join(base_dir, "final_fine_tune_train.json")
    verify_dataset(train_file, "FINAL TRAINING DATASET")
    
    # Verify test dataset
    test_file = os.path.join(base_dir, "final_fine_tune_test.json")
    verify_dataset(test_file, "FINAL TEST DATASET")
    
    print("\n" + "=" * 60)
    print("🎯 VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"✅ Files are ready for LLM fine-tuning!")
    print(f"📁 Training file: {train_file}")
    print(f"📁 Test file: {test_file}")

if __name__ == "__main__":
    main()
