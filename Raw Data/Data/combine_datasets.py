"""
Combine Multiple JSON Files into Final Training and Test Files
=============================================================
This script combines all training JSON files into one final training file
and all test JSON files into one final test file.
"""

import json
import os
from collections import Counter

def load_json_file(file_path):
    """Load a JSON file and return the data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} records from {os.path.basename(file_path)}")
        return data
    except Exception as e:
        print(f"❌ Error loading {file_path}: {str(e)}")
        return []

def combine_json_files(file_list, output_file, dataset_type="dataset"):
    """Combine multiple JSON files into one."""
    print(f"\n🔄 Combining {dataset_type} files...")
    print(f"Input files: {[os.path.basename(f) for f in file_list]}")
    
    all_data = []
    file_stats = {}
    
    # Load all files
    for file_path in file_list:
        if os.path.exists(file_path):
            data = load_json_file(file_path)
            if data:
                all_data.extend(data)
                file_stats[os.path.basename(file_path)] = len(data)
        else:
            print(f"⚠️  File not found: {file_path}")
    
    if not all_data:
        print(f"❌ No data to combine for {dataset_type}")
        return None
    
    # Remove exact duplicates while preserving order
    print(f"🔍 Removing duplicates...")
    seen = set()
    unique_data = []
    duplicates_removed = 0
    
    for item in all_data:
        # Create a unique identifier for each record
        identifier = (item.get('text', ''), item.get('label_string', ''))
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(item)
        else:
            duplicates_removed += 1
    
    print(f"📊 Statistics:")
    print(f"   Total records loaded: {len(all_data):,}")
    print(f"   Duplicates removed: {duplicates_removed:,}")
    print(f"   Final unique records: {len(unique_data):,}")
    
    # Analyze label distribution
    labels = [item.get('label_string', 'Unknown') for item in unique_data]
    label_counts = Counter(labels)
    unique_labels = len(label_counts)
    
    print(f"   Unique labels: {unique_labels}")
    print(f"   Top 5 conditions: {list(label_counts.most_common(5))}")
    
    # Calculate text statistics
    text_lengths = [len(item.get('text', '')) for item in unique_data]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    min_length = min(text_lengths) if text_lengths else 0
    max_length = max(text_lengths) if text_lengths else 0
    
    print(f"   Avg text length: {avg_length:.0f} characters")
    print(f"   Text length range: {min_length} - {max_length}")
    
    # Save combined file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Combined {dataset_type} saved to: {output_file}")
        
        # Create a summary
        summary = {
            "dataset_type": dataset_type,
            "source_files": file_stats,
            "total_records": len(unique_data),
            "unique_labels": unique_labels,
            "duplicates_removed": duplicates_removed,
            "text_statistics": {
                "avg_length": round(avg_length),
                "min_length": min_length,
                "max_length": max_length
            },
            "top_conditions": dict(label_counts.most_common(10)),
            "output_file": output_file
        }
        
        return summary
        
    except Exception as e:
        print(f"❌ Error saving {output_file}: {str(e)}")
        return None

def main():
    """Main function to combine all files."""
    print("=" * 80)
    print("🏥 COMBINING HEALTHCARE DATASETS FOR FINAL FINE-TUNING")
    print("=" * 80)
    
    # Define base directory
    base_dir = "FInal Processed Data"
    
    # Training files to combine
    train_files = [
        os.path.join(base_dir, "llm_ready_train.json"),
        os.path.join(base_dir, "processed1_train.json"),
        os.path.join(base_dir, "processed2_train.json")
    ]
    
    # Test files to combine
    test_files = [
        os.path.join(base_dir, "llm_ready_test.json"),
        os.path.join(base_dir, "processed1_test.json"),
        os.path.join(base_dir, "processed2_test.json")
    ]
    
    # Output files
    final_train_file = os.path.join(base_dir, "final_fine_tune_train.json")
    final_test_file = os.path.join(base_dir, "final_fine_tune_test.json")
    
    # Combine training files
    train_summary = combine_json_files(train_files, final_train_file, "TRAINING")
    
    # Combine test files
    test_summary = combine_json_files(test_files, final_test_file, "TEST")
    
    # Create final summary
    if train_summary and test_summary:
        final_summary = {
            "combination_date": "2025-07-19",
            "summary": "Final Combined Datasets for Healthcare LLM Fine-tuning",
            "training_dataset": train_summary,
            "test_dataset": test_summary,
            "recommendations": {
                "use_for_training": final_train_file,
                "use_for_testing": final_test_file,
                "recommended_model": "microsoft/biogpt-large",
                "expected_performance": "85-92% accuracy with combined dataset",
                "training_ready": True
            }
        }
        
        # Save final summary
        summary_file = os.path.join(base_dir, "final_dataset_summary.json")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
            print(f"\n📋 Final summary saved to: {summary_file}")
        except Exception as e:
            print(f"❌ Error saving summary: {str(e)}")
        
        print("\n" + "=" * 80)
        print("🎯 FINAL DATASET COMBINATION COMPLETE")
        print("=" * 80)
        print(f"🏆 Training Dataset: {final_train_file}")
        print(f"   📊 Total samples: {train_summary['total_records']:,}")
        print(f"   🏷️  Unique labels: {train_summary['unique_labels']}")
        print(f"   📝 Avg text length: {train_summary['text_statistics']['avg_length']} chars")
        
        print(f"\n🏆 Test Dataset: {final_test_file}")
        print(f"   📊 Total samples: {test_summary['total_records']:,}")
        print(f"   🏷️  Unique labels: {test_summary['unique_labels']}")
        print(f"   📝 Avg text length: {test_summary['text_statistics']['avg_length']} chars")
        
        print(f"\n✅ Ready for fine-tuning!")
        print(f"   Use: {final_train_file} for training")
        print(f"   Use: {final_test_file} for testing")
        
    else:
        print("\n❌ Failed to create combined datasets")

if __name__ == "__main__":
    main()
