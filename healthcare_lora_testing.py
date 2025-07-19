"""
Healthcare LLM Testing and Inference Script
==========================================
This script loads a fine-tuned healthcare LLM model and performs testing/inference
on new medical symptom descriptions for disease prediction.
"""

import os
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Hugging Face libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class HealthcareLLMTester:
    def __init__(self, model_dir="./healthcare_lora_model"):
        """
        Initialize the Healthcare LLM Tester.
        
        Args:
            model_dir (str): Directory containing the fine-tuned model
        """
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.label_mapping = None
        self.reverse_label_mapping = None
        
        print(f"🔍 Initializing Healthcare LLM Tester")
        print(f"📱 Device: {self.device}")
        print(f"📁 Model directory: {model_dir}")
        
        self.load_model()
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            print(f"\n🤖 Loading fine-tuned model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            print(f"✅ Tokenizer loaded")
            
            # Load label mapping
            label_mapping_path = os.path.join(self.model_dir, "label_mapping.json")
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, "r") as f:
                    self.label_mapping = json.load(f)
                # Convert string keys to integers
                self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
                self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
                print(f"✅ Label mapping loaded: {len(self.label_mapping)} conditions")
            else:
                print("⚠️  Label mapping not found, will need to be provided separately")
            
            # Load base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # Load LoRA weights if they exist
            adapter_path = os.path.join(self.model_dir, "adapter_model.bin")
            if os.path.exists(adapter_path):
                self.model = PeftModel.from_pretrained(base_model, self.model_dir)
                print(f"✅ LoRA model loaded with adapter weights")
            else:
                self.model = base_model
                print(f"✅ Base model loaded (no LoRA adapter found)")
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"🎯 Model ready for inference!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def predict_single(self, text, return_probabilities=False):
        """
        Predict medical condition for a single text input.
        
        Args:
            text (str): Medical symptom description
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            dict: Prediction results
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get predicted condition name
        predicted_condition = self.label_mapping.get(predicted_class, f"Unknown_Class_{predicted_class}")
        
        result = {
            'text': text,
            'predicted_condition': predicted_condition,
            'predicted_class_id': predicted_class,
            'confidence': confidence
        }
        
        if return_probabilities:
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities[0], k=min(5, len(self.label_mapping)))
            top_predictions = []
            
            for prob, idx in zip(top_probs, top_indices):
                condition = self.label_mapping.get(idx.item(), f"Unknown_Class_{idx.item()}")
                top_predictions.append({
                    'condition': condition,
                    'probability': prob.item(),
                    'class_id': idx.item()
                })
            
            result['top_predictions'] = top_predictions
        
        return result
    
    def predict_batch(self, texts, batch_size=8):
        """
        Predict medical conditions for a batch of texts.
        
        Args:
            texts (list): List of medical symptom descriptions
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of prediction results
        """
        print(f"🔄 Processing {len(texts)} samples in batches of {batch_size}")
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
            
            # Process results
            for j, (text, pred_class, confidence) in enumerate(zip(batch_texts, predicted_classes, confidences)):
                predicted_condition = self.label_mapping.get(pred_class.item(), f"Unknown_Class_{pred_class.item()}")
                
                results.append({
                    'text': text,
                    'predicted_condition': predicted_condition,
                    'predicted_class_id': pred_class.item(),
                    'confidence': confidence.item()
                })
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"✅ Processed {min(i + batch_size, len(texts))}/{len(texts)} samples")
        
        return results
    
    def test_on_csv(self, test_file, output_file=None):
        """
        Test the model on a CSV file and generate detailed evaluation.
        
        Args:
            test_file (str): Path to test CSV file
            output_file (str): Path to save results (optional)
            
        Returns:
            dict: Evaluation results
        """
        print(f"\n📊 Testing model on {test_file}")
        
        # Load test data
        test_df = pd.read_csv(test_file)
        print(f"📋 Test data loaded: {len(test_df)} samples")
        
        # Make predictions
        texts = test_df['text'].tolist()
        predictions = self.predict_batch(texts)
        
        # Add predictions to dataframe
        test_df['predicted_condition'] = [pred['predicted_condition'] for pred in predictions]
        test_df['predicted_class_id'] = [pred['predicted_class_id'] for pred in predictions]
        test_df['confidence'] = [pred['confidence'] for pred in predictions]
        
        # Calculate metrics if true labels are available
        if 'label_string' in test_df.columns:
            print(f"\n📈 Calculating evaluation metrics...")
            
            # Map true labels to class IDs
            true_labels = [self.reverse_label_mapping.get(label, -1) for label in test_df['label_string']]
            pred_labels = test_df['predicted_class_id'].tolist()
            
            # Filter out unknown labels
            valid_indices = [i for i, label in enumerate(true_labels) if label != -1]
            true_labels_filtered = [true_labels[i] for i in valid_indices]
            pred_labels_filtered = [pred_labels[i] for i in valid_indices]
            
            if len(true_labels_filtered) > 0:
                accuracy = accuracy_score(true_labels_filtered, pred_labels_filtered)
                
                # Get unique labels for classification report
                unique_labels = sorted(list(set(true_labels_filtered + pred_labels_filtered)))
                label_names = [self.label_mapping.get(label, f"Class_{label}") for label in unique_labels]
                
                report = classification_report(
                    true_labels_filtered, 
                    pred_labels_filtered,
                    labels=unique_labels,
                    target_names=label_names,
                    output_dict=True,
                    zero_division=0
                )
                
                print(f"🎯 Accuracy: {accuracy:.4f}")
                print(f"📊 Classification Report:")
                print(classification_report(
                    true_labels_filtered, 
                    pred_labels_filtered,
                    labels=unique_labels,
                    target_names=label_names,
                    zero_division=0
                ))
                
                # Calculate per-condition accuracy
                condition_accuracy = {}
                for condition in test_df['label_string'].unique():
                    if condition in self.reverse_label_mapping:
                        condition_mask = test_df['label_string'] == condition
                        condition_predictions = test_df[condition_mask]['predicted_condition']
                        condition_acc = (condition_predictions == condition).mean()
                        condition_accuracy[condition] = condition_acc
                
                # Create confusion matrix
                self._plot_confusion_matrix(
                    true_labels_filtered, 
                    pred_labels_filtered, 
                    label_names
                )
                
                results = {
                    'accuracy': accuracy,
                    'classification_report': report,
                    'condition_accuracy': condition_accuracy,
                    'total_samples': len(test_df),
                    'valid_samples': len(true_labels_filtered)
                }
            else:
                print("⚠️  No valid labels found for evaluation")
                results = {'total_samples': len(test_df), 'valid_samples': 0}
        else:
            print("ℹ️  No true labels provided, skipping accuracy calculation")
            results = {'total_samples': len(test_df)}
        
        # Save results
        if output_file:
            test_df.to_csv(output_file, index=False)
            print(f"💾 Results saved to: {output_file}")
            
            # Save evaluation metrics
            if 'accuracy' in results:
                eval_file = output_file.replace('.csv', '_evaluation.json')
                results['test_file'] = test_file
                results['output_file'] = output_file
                results['evaluation_date'] = datetime.now().isoformat()
                
                with open(eval_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"📊 Evaluation metrics saved to: {eval_file}")
        
        return test_df, results
    
    def _plot_confusion_matrix(self, y_true, y_pred, label_names):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names
        )
        plt.title('Confusion Matrix - Healthcare LLM Testing')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def interactive_demo(self):
        """Run an interactive demo for testing."""
        print(f"\n🎮 Interactive Healthcare LLM Demo")
        print(f"Type 'quit' to exit")
        print(f"-" * 50)
        
        while True:
            try:
                text = input("\n💬 Enter medical symptoms: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    continue
                
                # Make prediction
                result = self.predict_single(text, return_probabilities=True)
                
                print(f"\n🔍 Analysis Results:")
                print(f"📝 Input: {result['text']}")
                print(f"🎯 Predicted Condition: {result['predicted_condition']}")
                print(f"📊 Confidence: {result['confidence']:.4f}")
                
                if 'top_predictions' in result:
                    print(f"\n🏆 Top 5 Predictions:")
                    for i, pred in enumerate(result['top_predictions'], 1):
                        print(f"   {i}. {pred['condition']}: {pred['probability']:.4f}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main function for testing."""
    print("=" * 80)
    print("🏥 HEALTHCARE LLM TESTING AND INFERENCE")
    print("=" * 80)
    
    # Initialize tester
    tester = HealthcareLLMTester("./healthcare_lora_model")
    
    # Test on CSV file if available
    test_files = [
        "llm_ready_test_dataset.csv",
        "processed1-test-dataset.csv",
        "processed2-test.csv"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🧪 Testing on {test_file}")
            output_file = f"predictions_{test_file}"
            
            test_df, results = tester.test_on_csv(test_file, output_file)
            
            if 'accuracy' in results:
                print(f"🎯 Final Accuracy: {results['accuracy']:.4f}")
            break
    else:
        print("⚠️  No test files found, skipping CSV testing")
    
    # Example predictions
    example_symptoms = [
        "I have been experiencing high fever, cough, and difficulty breathing.",
        "I have severe headache, nausea, and sensitivity to light.",
        "I have been having chest pain, shortness of breath, and sweating.",
        "I have itchy skin, rash, and swelling around my eyes.",
        "I have been experiencing joint pain, stiffness, and swelling in my hands."
    ]
    
    print(f"\n📝 Example Predictions:")
    print(f"-" * 50)
    
    for i, symptom in enumerate(example_symptoms, 1):
        result = tester.predict_single(symptom)
        print(f"\n{i}. Symptoms: {symptom}")
        print(f"   Predicted: {result['predicted_condition']} (confidence: {result['confidence']:.4f})")
    
    # Interactive demo
    print(f"\n🎮 Starting interactive demo...")
    tester.interactive_demo()

if __name__ == "__main__":
    main()
