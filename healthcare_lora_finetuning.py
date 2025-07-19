"""
Healthcare LLM Fine-tuning with LoRA
====================================
This script fine-tunes a language model for medical diagnosis classification using LoRA (Low-Rank Adaptation).
Supports multiple model architectures and provides comprehensive evaluation metrics.
"""

import os
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Hugging Face libraries
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Set environment variables for better performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

class HealthcareLLMFineTuner:
    def __init__(self, model_name="microsoft/BioGPT-Large", max_length=512):
        """
        Initialize the Healthcare LLM Fine-tuner.
        
        Args:
            model_name (str): Hugging Face model name to fine-tune
            max_length (int): Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_labels = 0
        
        print(f"🔥 Initializing Healthcare LLM Fine-tuner")
        print(f"📱 Device: {self.device}")
        print(f"🤖 Model: {model_name}")
        print(f"📏 Max length: {max_length}")
        
    def load_and_prepare_data(self, train_file, test_file=None, validation_split=0.2):
        """
        Load and prepare training and testing data.
        
        Args:
            train_file (str): Path to training CSV file
            test_file (str): Path to testing CSV file (optional)
            validation_split (float): Validation split ratio if no test file provided
        """
        print(f"\n📊 Loading data from {train_file}")
        
        # Load training data
        train_df = pd.read_csv(train_file)
        print(f"✅ Training data loaded: {len(train_df)} samples")
        
        # Load test data or create validation split
        if test_file and os.path.exists(test_file):
            test_df = pd.read_csv(test_file)
            print(f"✅ Test data loaded: {len(test_df)} samples")
        else:
            print(f"📂 Creating validation split ({validation_split * 100}%)")
            train_df, test_df = train_test_split(
                train_df, test_size=validation_split, 
                stratify=train_df['label_string'], random_state=42
            )
        
        # Data quality checks
        self._validate_data(train_df, test_df)
        
        # Encode labels
        all_labels = pd.concat([train_df['label_string'], test_df['label_string']]).unique()
        self.label_encoder.fit(all_labels)
        self.num_labels = len(self.label_encoder.classes_)
        
        print(f"🏷️  Found {self.num_labels} unique medical conditions")
        print(f"🔝 Top conditions: {list(self.label_encoder.classes_[:10])}")
        
        # Encode labels
        train_df['labels'] = self.label_encoder.transform(train_df['label_string'])
        test_df['labels'] = self.label_encoder.transform(test_df['label_string'])
        
        self.train_df = train_df
        self.test_df = test_df
        
        return train_df, test_df
    
    def _validate_data(self, train_df, test_df):
        """Validate data quality and format."""
        print("\n🔍 Validating data quality...")
        
        # Check required columns
        required_cols = ['text', 'label_string']
        for df_name, df in [('Training', train_df), ('Test', test_df)]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"❌ {df_name} data missing columns: {missing_cols}")
        
        # Check for missing values
        train_missing = train_df[required_cols].isnull().sum()
        test_missing = test_df[required_cols].isnull().sum()
        
        if train_missing.any() or test_missing.any():
            print("⚠️  Warning: Found missing values")
            print(f"Training: {train_missing.to_dict()}")
            print(f"Test: {test_missing.to_dict()}")
        
        # Text length statistics
        train_lengths = train_df['text'].str.len()
        test_lengths = test_df['text'].str.len()
        
        print(f"📝 Text length stats:")
        print(f"   Training - Mean: {train_lengths.mean():.0f}, Max: {train_lengths.max()}")
        print(f"   Test - Mean: {test_lengths.mean():.0f}, Max: {test_lengths.max()}")
        
        if train_lengths.max() > self.max_length or test_lengths.max() > self.max_length:
            print(f"⚠️  Warning: Some texts exceed max_length ({self.max_length})")
    
    def setup_model_and_tokenizer(self):
        """Setup the model and tokenizer with LoRA configuration."""
        print(f"\n🤖 Setting up model and tokenizer...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=16,  # Rank
                lora_alpha=32,  # LoRA scaling parameter
                lora_dropout=0.1,
                target_modules=["query", "value", "key", "dense"] if "bert" in self.model_name.lower() else ["q_proj", "v_proj", "k_proj", "out_proj"]
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            self.model.to(self.device)
            
            print(f"✅ Model and tokenizer setup complete")
            print(f"📊 Model parameters: {self.model.num_parameters():,}")
            print(f"🔧 Trainable parameters: {self.model.num_parameters(only_trainable=True):,}")
            
        except Exception as e:
            print(f"❌ Error setting up model: {str(e)}")
            print(f"💡 Trying alternative model: google/flan-t5-base")
            
            # Fallback to T5
            self.model_name = "google/flan-t5-base"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
            
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.to(self.device)
    
    def tokenize_data(self):
        """Tokenize the training and test data."""
        print(f"\n🔤 Tokenizing data...")
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            tokenized['labels'] = examples['labels']
            return tokenized
        
        # Convert to datasets
        train_dataset = Dataset.from_pandas(self.train_df[['text', 'labels']])
        test_dataset = Dataset.from_pandas(self.test_df[['text', 'labels']])
        
        # Tokenize
        self.train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        print(f"✅ Tokenization complete")
        print(f"📊 Train dataset: {len(self.train_dataset)} samples")
        print(f"📊 Test dataset: {len(self.test_dataset)} samples")
    
    def train_model(self, output_dir="./healthcare_lora_model", epochs=3, batch_size=8, learning_rate=2e-5):
        """
        Train the model with LoRA fine-tuning.
        
        Args:
            output_dir (str): Directory to save the fine-tuned model
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for training
        """
        print(f"\n🚀 Starting LoRA fine-tuning...")
        print(f"📁 Output directory: {output_dir}")
        print(f"🔄 Epochs: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"📈 Learning rate: {learning_rate}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        print(f"🔥 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label encoder
        label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        with open(f"{output_dir}/label_mapping.json", "w") as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"✅ Training completed!")
        print(f"💾 Model saved to: {output_dir}")
        
        self.output_dir = output_dir
        return trainer
    
    def evaluate_model(self, trainer=None):
        """Evaluate the fine-tuned model."""
        print(f"\n📊 Evaluating model performance...")
        
        if trainer:
            # Get predictions
            predictions = trainer.predict(self.test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
        else:
            # Manual evaluation if no trainer provided
            self.model.eval()
            y_pred = []
            y_true = []
            
            with torch.no_grad():
                for batch in torch.utils.data.DataLoader(self.test_dataset, batch_size=8):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    y_pred.extend(predictions.cpu().numpy())
                    y_true.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))
        
        # Save detailed results
        if hasattr(self, 'output_dir'):
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'evaluation_date': datetime.now().isoformat()
            }
            
            with open(f"{self.output_dir}/evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Create and save confusion matrix
            self._plot_confusion_matrix(y_true, y_pred)
        
        return accuracy, report
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix - Healthcare LLM')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if hasattr(self, 'output_dir'):
            plt.savefig(f"{self.output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the fine-tuning process."""
    print("=" * 80)
    print("🏥 HEALTHCARE LLM FINE-TUNING WITH LORA")
    print("=" * 80)
    
    # Initialize fine-tuner
    fine_tuner = HealthcareLLMFineTuner(
        model_name="microsoft/BioGPT-Large",  # You can change this
        max_length=512
    )
    
    # Load and prepare data
    train_df, test_df = fine_tuner.load_and_prepare_data(
        train_file="llm_ready_train_dataset.csv",
        test_file="llm_ready_test_dataset.csv"
    )
    
    # Setup model and tokenizer
    fine_tuner.setup_model_and_tokenizer()
    
    # Tokenize data
    fine_tuner.tokenize_data()
    
    # Train model
    trainer = fine_tuner.train_model(
        output_dir="./healthcare_lora_model",
        epochs=3,
        batch_size=4,  # Adjust based on your GPU memory
        learning_rate=2e-5
    )
    
    # Evaluate model
    accuracy, report = fine_tuner.evaluate_model(trainer)
    
    print(f"\n🎉 Fine-tuning completed successfully!")
    print(f"🎯 Final Accuracy: {accuracy:.4f}")
    print(f"💾 Model saved to: ./healthcare_lora_model")

if __name__ == "__main__":
    main()
