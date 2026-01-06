"""
Fake News Detection - DistilBERT Training Script
=================================================
This script fine-tunes DistilBERT on the fake/real news dataset.

Usage:
    python 03_train_distilbert.py

For GPU training (recommended):
    - Use Google Colab with GPU runtime
    - Or local machine with CUDA

Training Time:
    - CPU: ~2-3 hours
    - GPU: ~15-20 minutes
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 512,
    'batch_size': 8,  # Reduce if OOM errors
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'output_dir': './fake_news_distilbert',
    'test_size': 0.2,
    'random_state': 42,
}

# =============================================================================
# Dataset Class
# =============================================================================

class NewsDataset(Dataset):
    """Custom dataset for news articles."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =============================================================================
# Metrics Function
# =============================================================================

def compute_metrics(pred):
    """Compute accuracy, precision, recall, and F1."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# =============================================================================
# Main Training Function
# =============================================================================

def main():
    print("=" * 60)
    print("🚀 Fake News Detection - DistilBERT Training")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Device: {device.upper()}")
    if device == 'cpu':
        print("⚠️  Training on CPU will be slow. Consider using Google Colab.")
    
    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    print("\n📂 Loading datasets...")
    
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    
    fake_df['label'] = 0  # Fake = 0
    true_df['label'] = 1  # Real = 1
    
    df = pd.concat([fake_df, true_df], axis=0)
    df = df.sample(frac=1, random_state=CONFIG['random_state']).reset_index(drop=True)
    
    print(f"   Total samples: {len(df):,}")
    print(f"   Fake: {len(fake_df):,} | Real: {len(true_df):,}")
    
    # -------------------------------------------------------------------------
    # Step 2: Prepare Text (use 'text' column, no manual cleaning needed)
    # -------------------------------------------------------------------------
    texts = df['text'].values
    labels = df['label'].values
    
    # -------------------------------------------------------------------------
    # Step 3: Train/Test Split
    # -------------------------------------------------------------------------
    print("\n✂️  Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=labels
    )
    
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # -------------------------------------------------------------------------
    # Step 4: Load Tokenizer
    # -------------------------------------------------------------------------
    print("\n🔤 Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'])
    
    # -------------------------------------------------------------------------
    # Step 5: Create Datasets
    # -------------------------------------------------------------------------
    print("📦 Creating datasets...")
    
    train_dataset = NewsDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
    test_dataset = NewsDataset(X_test, y_test, tokenizer, CONFIG['max_length'])
    
    # -------------------------------------------------------------------------
    # Step 6: Load Model
    # -------------------------------------------------------------------------
    print("🤖 Loading DistilBERT model...")
    
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=2
    )
    
    # -------------------------------------------------------------------------
    # Step 7: Training Arguments
    # -------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'] * 2,
        warmup_steps=CONFIG['warmup_steps'],
        weight_decay=CONFIG['weight_decay'],
        learning_rate=CONFIG['learning_rate'],
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        report_to='none',  # Disable wandb/tensorboard
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU
    )
    
    # -------------------------------------------------------------------------
    # Step 8: Create Trainer
    # -------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # -------------------------------------------------------------------------
    # Step 9: Train!
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🏋️  Starting Training...")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # -------------------------------------------------------------------------
    # Step 10: Evaluate
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 Evaluation Results")
    print("=" * 60)
    
    results = trainer.evaluate()
    
    print(f"\n   Accuracy:  {results['eval_accuracy']:.4f}")
    print(f"   F1 Score:  {results['eval_f1']:.4f}")
    print(f"   Precision: {results['eval_precision']:.4f}")
    print(f"   Recall:    {results['eval_recall']:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 11: Save Model
    # -------------------------------------------------------------------------
    print(f"\n💾 Saving model to '{CONFIG['output_dir']}'...")
    
    model.save_pretrained(CONFIG['output_dir'])
    tokenizer.save_pretrained(CONFIG['output_dir'])
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: {CONFIG['output_dir']}/")
    print("\nNext steps:")
    print("  1. Run the Streamlit app: streamlit run app_v2.py")
    print("  2. Test with modern news articles")

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    main()
