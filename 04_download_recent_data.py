"""
Download and Prepare Recent News Datasets
==========================================
This script helps you download and prepare additional training data
from recent fake news datasets.

Run: python 04_download_recent_data.py
"""

import os
import pandas as pd
import requests
from io import StringIO

def download_covid_fake_news():
    """
    Instructions to get COVID-19 Fake News Dataset:
    1. Go to: https://www.kaggle.com/datasets/arashnic/covid19-fake-news
    2. Download 'Constraint_Train.csv' and 'Constraint_Test.csv'
    3. Place them in this folder
    """
    print("=" * 60)
    print("📥 COVID-19 Fake News Dataset")
    print("=" * 60)
    print("""
To add recent (2020-2021) fake news data:

1. Visit: https://www.kaggle.com/datasets/arashnic/covid19-fake-news
2. Click 'Download' (requires free Kaggle account)
3. Extract and place these files in this folder:
   - Constraint_Train.csv
   - Constraint_Test.csv
4. Run this script again

This dataset contains ~10,000 COVID-related news articles
labeled as 'fake' or 'real'.
    """)

def prepare_covid_dataset():
    """Prepare COVID dataset for training."""
    
    train_file = 'Constraint_Train.csv'
    test_file = 'Constraint_Test.csv'
    
    if not os.path.exists(train_file):
        print(f"❌ {train_file} not found. Please download from Kaggle.")
        download_covid_fake_news()
        return None
    
    print(f"✅ Found {train_file}")
    
    # Load COVID data
    covid_train = pd.read_csv(train_file)
    
    if os.path.exists(test_file):
        covid_test = pd.read_csv(test_file)
        covid_df = pd.concat([covid_train, covid_test])
    else:
        covid_df = covid_train
    
    # Standardize columns
    # COVID dataset has: id, tweet, label (fake/real)
    covid_df = covid_df.rename(columns={'tweet': 'text'})
    covid_df['label'] = covid_df['label'].apply(lambda x: 0 if x.lower() == 'fake' else 1)
    
    print(f"✅ Loaded {len(covid_df)} COVID news samples")
    print(f"   Fake: {(covid_df['label'] == 0).sum()}")
    print(f"   Real: {(covid_df['label'] == 1).sum()}")
    
    return covid_df[['text', 'label']]

def combine_datasets():
    """Combine original dataset with COVID dataset."""
    
    print("\n" + "=" * 60)
    print("📊 Combining Datasets")
    print("=" * 60)
    
    # Load original data
    print("\n📂 Loading original dataset...")
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    
    fake_df['label'] = 0
    true_df['label'] = 1
    
    original_df = pd.concat([
        fake_df[['text', 'label']], 
        true_df[['text', 'label']]
    ])
    
    print(f"   Original samples: {len(original_df)}")
    
    # Load COVID data
    covid_df = prepare_covid_dataset()
    
    if covid_df is not None:
        # Combine
        combined_df = pd.concat([original_df, covid_df])
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined dataset
        combined_df.to_csv('combined_training_data.csv', index=False)
        
        print(f"\n✅ Combined dataset saved!")
        print(f"   Total samples: {len(combined_df)}")
        print(f"   Original: {len(original_df)}")
        print(f"   COVID: {len(covid_df)}")
        print(f"\n📄 Saved to: combined_training_data.csv")
        
        return combined_df
    
    return None

def create_custom_dataset():
    """Instructions for creating custom recent news dataset."""
    
    print("\n" + "=" * 60)
    print("📝 Creating Custom Recent News Dataset")
    print("=" * 60)
    print("""
To create your own recent news dataset:

1. REAL NEWS SOURCES (copy articles from):
   - Reuters: https://www.reuters.com
   - AP News: https://apnews.com
   - BBC: https://www.bbc.com/news
   - The Hindu: https://www.thehindu.com
   
2. FAKE NEWS SOURCES (known misinformation):
   - Fact-check sites debunked articles
   - Alt-News: https://www.altnews.in
   - Boom Live: https://www.boomlive.in
   
3. Create a CSV file with columns:
   - text: The full article text
   - label: 0 for fake, 1 for real
   
4. Save as 'custom_news_data.csv' in this folder

5. Run: python 03_train_distilbert.py --data custom_news_data.csv
    """)

def main():
    print("=" * 60)
    print("🔄 Recent News Data Preparation Tool")
    print("=" * 60)
    
    print("\nOptions:")
    print("1. Download COVID-19 Fake News Dataset (2020-2021)")
    print("2. Combine existing datasets")
    print("3. Create custom dataset (manual)")
    
    # Check what files exist
    print("\n📂 Checking existing files...")
    
    files = {
        'Fake.csv': os.path.exists('Fake.csv'),
        'True.csv': os.path.exists('True.csv'),
        'Constraint_Train.csv': os.path.exists('Constraint_Train.csv'),
        'combined_training_data.csv': os.path.exists('combined_training_data.csv')
    }
    
    for f, exists in files.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {f}")
    
    print("\n" + "-" * 60)
    
    # Try to combine if COVID data exists
    if files['Constraint_Train.csv']:
        combine_datasets()
    else:
        download_covid_fake_news()
        create_custom_dataset()

if __name__ == '__main__':
    main()
