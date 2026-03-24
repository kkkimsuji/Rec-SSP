import sys
import os
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # src folder
ROOT_DIR = os.path.dirname(BASE_DIR)                  # root folder

# Add the 'model' folder to path so we can import 'proposed.py'
sys.path.insert(0, os.path.join(ROOT_DIR, 'model'))

# Import the model building function from proposed.py
from proposed import ModelBuild 

def load_config():
    """Loads configuration from config.yaml in the root."""
    config_path = os.path.join(ROOT_DIR, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def prepare_tensors(df, n):
    """Converts dataframe columns into numpy tensors for model input."""
    user_static = np.stack(df['user_average_bert'].values).astype('float32')
    item_bert = np.stack(df['item_average_bert'].values).astype('float32')
    item_label = np.stack(df['multi_hot_encoding'].values).astype('float32')
    
    # Sequential features
    user_dynamic = np.stack(df['bert_embedding_seq'].values).astype('float32')
    label_seq = np.stack(df['multi_hot_encoding_seq'].values).astype('float32')
    
    ratings = df['rating'].values.astype('float32')
    return [user_static, item_bert, item_label, user_dynamic, label_seq], ratings

def run_training():
    """Logic for splitting data, building, and training the model."""
    config = load_config()
    model_params = config.get('model_config', {})
    train_params = config.get('training_config', {})

    n = model_params.get('K', 5)
    lr = train_params.get('learning_rate', 0.0002)
    batch_size = train_params.get('batch_size', 64)
    epochs = train_params.get('epochs', 100)

    # Load the pickle file generated in Phase 1
    data_path = os.path.join(ROOT_DIR, "data", "preprocessed", "processed_data.pkl")
    df = pd.read_pickle(data_path)

    # Split data: 7(Train) : 1(Val) : 2(Test)
    train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)

    # Initialize model
    multi_hot_len = len(df.iloc[0]['multi_hot_encoding'])
    proposed_model = ModelBuild(n=n, multi_hot_len=multi_hot_len, config=model_params)
    
    proposed_model.compile(optimizer=Adam(learning_rate=lr), loss='MSE', metrics=["mse", "mae"])

    # Prepare data for training
    X_train, y_train = prepare_tensors(train_df, n)
    X_val, y_val = prepare_tensors(val_df, n)
    X_test, y_test = prepare_tensors(test_df, n)

    # Callbacks for optimization
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("\nModel training started...")
    proposed_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    # Final Evaluation
    print("\n" + "="*40)
    print("      FINAL TEST EVALUATION")
    print("="*40)
    
    preds = proposed_model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print("="*40)

    # Add this at the end of run_training() in src/train.py
    save_path = os.path.join(ROOT_DIR, "saved_models", "recssp_model.h5")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    proposed_model.save(save_path)
    print(f"✅ Model saved to: {save_path}")