import pandas as pd
import numpy as np
import json
import yaml
import os
import pickle
from sklearn.preprocessing import LabelEncoder

# Import the custom BERT module located in the same 'src' folder
from bert import get_bert_embeddings 

# 1. Configuration Setup
# We assume config.yaml is in the project root (one level up from 'src')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # C:\Rec-SSP\src
ROOT_DIR = os.path.dirname(BASE_DIR)                   # C:\Rec-SSP

CONFIG_PATH = os.path.join(ROOT_DIR, 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# K is the sequence length (Paper optimal: 5)
K = config['model_config'].get('K', 5)  

# --- PATH CONFIGURATION (Stepping out of 'src' to find 'data') ---
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "data", "preprocessed")

# Ensure the output directory exists in the root data folder
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)
    print(f"Created directory: {PROCESSED_DATA_PATH}")

def load_and_merge_data():
    """
    Parses JSONL files from data/raw and merges reviews with metadata.
    Applies 5-core filtering for data quality.
    """
    def parse_json(file_name):
        # Construct absolute path to the raw data
        path = os.path.normpath(os.path.join(RAW_DATA_PATH, file_name))
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: Could not find '{file_name}' at {path}")
            
        data = []
        print(f"Reading {file_name}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)

    # Load raw datasets
    meta_df = parse_json('metadata.jsonl')
    review_df = parse_json('sampledata.jsonl')

    # Column selection and basic cleaning
    review_df = review_df[['rating', 'text', 'parent_asin', 'user_id', 'timestamp']].copy()
    review_df.rename(columns={
        'parent_asin': 'item', 
        'text': 'reviewText', 
        'user_id': 'user', 
        'timestamp': 'time'
    }, inplace=True)

    print("Merging datasets and applying 5-core filter...")
    df_merged = pd.merge(review_df, meta_df[['parent_asin', 'categories']], 
                         left_on='item', right_on='parent_asin', how='left')
    
    # Drop rows with missing values
    df_merged = df_merged.dropna(subset=['reviewText', 'categories']).drop(columns=['parent_asin'])
    
    # 5-core filtering (Eq 331): Keep users with >= 5 reviews
    user_counts = df_merged['user'].value_counts()
    df_merged = df_merged[df_merged['user'].isin(user_counts.index[user_counts >= 5])]
    
    return df_merged

def multi_hot_encode(categories, category_to_index, num_classes):
    """Encodes multiple categories into a single multi-hot vector (Eq 228)."""
    vector = np.zeros(num_classes)
    for cat in categories:
        if cat in category_to_index:
            vector[category_to_index[cat]] = 1
    return vector

def generate_multilevel_data(df):
    """
    Main Preprocessing Pipeline:
    1. Extract BERT [CLS] embeddings
    2. Calculate long-term user preferences (avg)
    3. Generate multi-hot category vectors
    4. Extract recent K-interaction sequences
    """
    # Step 1: BERT Feature Extraction (Eq 1)
    print("Extracting BERT embeddings (this might take time)...")
    review_map = get_bert_embeddings(df['reviewText'].tolist())
    df['bert_embedding'] = df['reviewText'].map(review_map)

    # Step 2: Global Averages (Eq 2 & 8)
    print("Calculating User/Item average embeddings...")
    # Long-term preference representation
    df['user_average_bert'] = df.groupby('user')['bert_embedding'].transform('mean')
    # Item characteristic representation
    df['item_average_bert'] = df.groupby('item')['bert_embedding'].transform('mean')

    # Step 3: Category Vectorization
    print("Encoding category multi-hot vectors...")
    unique_cats = df['categories'].explode().unique()
    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
    num_cats = len(unique_cats)
    df['multi_hot_encoding'] = df['categories'].apply(lambda x: multi_hot_encode(x, cat_to_idx, num_cats))

    # Step 4: Extract Recent K Sequences (Short-term dynamics)
    print(f"Extracting short-term sequence patterns (K={K})...")
    # Sort and group per user to find the latest K interactions
    result = (
        df.sort_values(by=['user', 'time'], ascending=[True, False])
          .groupby('user')
          .apply(lambda x: {
              'bert_embedding_seq': x.head(K)['bert_embedding'].tolist()[::-1],
              'multi_hot_encoding_seq': x.head(K)['multi_hot_encoding'].tolist()[::-1]
          })
          .reset_index()
    )

    # Map the sequences back to the main dataframe
    result_expanded = pd.json_normalize(result[0])
    result = pd.concat([result[['user']], result_expanded], axis=1)

    # Final Merge: Align all multilevel signals with target interactions
    df_all = pd.merge(
        df[['user', 'item', 'rating', 'bert_embedding', 'user_average_bert', 
            'item_average_bert', 'multi_hot_encoding']], 
        result, on='user', how='left'
    )

    return df_all

if __name__ == "__main__":
    try:
        # Step A: Data Loading & Initial Filtering
        raw_df = load_and_merge_data()
        
        # Step B: Multi-level Feature Processing
        final_df = generate_multilevel_data(raw_df)
        
        # Step C: Save final dataset as pickle
        save_file = os.path.join(PROCESSED_DATA_PATH, 'processed_data.pkl')
        final_df.to_pickle(save_file)
        
        print("-" * 40)
        print(f"Preprocessing Success!")
        print(f"Raw Data Path: {RAW_DATA_PATH}")
        print(f"Output Saved: {save_file}")
        print(f"Total Samples: {len(final_df)}")
        print("-" * 40)
        
        # Verification: Print first row sample
        print("\n[Data Inspection - First Row]")
        print(final_df.iloc[0])
        
    except Exception as e:
        print(f"An error occurred: {e}")