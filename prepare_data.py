"""
Process COVID-19 Dataset and Combine with Original Data
========================================================
This script processes the COVID-19 fake news dataset and combines it
with the original 2016-2017 dataset for expanded training.
"""

import pandas as pd
import os

def main():
    print("=" * 60)
    print("📊 Processing COVID-19 Fake News Dataset")
    print("=" * 60)
    
    # Paths
    covid_dir = "./covid_data"
    
    # Load COVID Fake News
    print("\n📂 Loading COVID-19 fake news...")
    fake_files = [
        "NewsFakeCOVID-19.csv",
        "NewsFakeCOVID-19_5.csv",
        "NewsFakeCOVID-19_7.csv"
    ]
    
    fake_dfs = []
    for f in fake_files:
        path = os.path.join(covid_dir, f)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                if 'content' in df.columns:
                    fake_dfs.append(df[['content']].rename(columns={'content': 'text'}))
                elif 'title' in df.columns:
                    fake_dfs.append(df[['title']].rename(columns={'title': 'text'}))
                print(f"   ✅ {f}: {len(df)} rows")
            except Exception as e:
                print(f"   ⚠️ {f}: Error - {e}")
    
    # Load COVID Real News
    print("\n📂 Loading COVID-19 real news...")
    real_files = [
        "NewsRealCOVID-19.csv",
        "NewsRealCOVID-19_5.csv",
        "NewsRealCOVID-19_7.csv"
    ]
    
    real_dfs = []
    for f in real_files:
        path = os.path.join(covid_dir, f)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                if 'content' in df.columns:
                    real_dfs.append(df[['content']].rename(columns={'content': 'text'}))
                elif 'title' in df.columns:
                    real_dfs.append(df[['title']].rename(columns={'title': 'text'}))
                print(f"   ✅ {f}: {len(df)} rows")
            except Exception as e:
                print(f"   ⚠️ {f}: Error - {e}")
    
    # Combine COVID data
    covid_fake = pd.concat(fake_dfs, ignore_index=True) if fake_dfs else pd.DataFrame()
    covid_real = pd.concat(real_dfs, ignore_index=True) if real_dfs else pd.DataFrame()
    
    covid_fake['label'] = 0
    covid_real['label'] = 1
    
    covid_df = pd.concat([covid_fake, covid_real], ignore_index=True)
    
    # Clean
    covid_df = covid_df.dropna(subset=['text'])
    covid_df = covid_df[covid_df['text'].str.len() > 50]  # Keep only substantial text
    
    print(f"\n📊 COVID-19 Dataset Summary:")
    print(f"   Fake: {len(covid_fake)}")
    print(f"   Real: {len(covid_real)}")
    print(f"   Total (cleaned): {len(covid_df)}")
    
    # Load original data
    print("\n📂 Loading original dataset...")
    original_fake = pd.read_csv('Fake.csv')
    original_real = pd.read_csv('True.csv')
    
    original_fake['label'] = 0
    original_real['label'] = 1
    
    original_df = pd.concat([
        original_fake[['text', 'label']],
        original_real[['text', 'label']]
    ], ignore_index=True)
    
    print(f"   Original samples: {len(original_df)}")
    
    # Combine all
    print("\n🔄 Combining datasets...")
    combined_df = pd.concat([original_df, covid_df[['text', 'label']]], ignore_index=True)
    combined_df = combined_df.dropna(subset=['text'])
    combined_df = combined_df.drop_duplicates(subset=['text'])
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_file = 'combined_training_data.csv'
    combined_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("✅ COMBINED DATASET CREATED!")
    print("=" * 60)
    print(f"\n📊 Final Statistics:")
    print(f"   Total samples: {len(combined_df):,}")
    print(f"   Fake: {(combined_df['label'] == 0).sum():,}")
    print(f"   Real: {(combined_df['label'] == 1).sum():,}")
    print(f"\n📄 Saved to: {output_file}")
    print("\n🚀 Ready to train! Run: python3 03_train_distilbert.py")

if __name__ == '__main__':
    main()
