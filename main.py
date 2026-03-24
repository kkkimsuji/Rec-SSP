import sys
import os

# Set the absolute path for the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add both 'src' and 'model' directories to the Python path
# This allows scripts to find modules in these specific folders
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, os.path.join(BASE_DIR, 'model'))

try:
    import data_processing  # Found in src/
    import train            # Found in src/
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    print("=" * 60)
    print("   Rec-SSP: Full Pipeline Execution   ")
    print("=" * 60)

    # --- PHASE 1: Data Preprocessing ---
    print("\n[PHASE 1] Starting Data Preprocessing...")
    try:
        # Load raw data and apply filtering
        raw_df = data_processing.load_and_merge_data()
        
        # Process multilevel features (BERT + Sequences)
        final_df = data_processing.generate_multilevel_data(raw_df)
        
        # Define path and save the processed pickle file
        save_dir = os.path.join(BASE_DIR, "data", "preprocessed")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, "processed_data.pkl")
        final_df.to_pickle(save_path)
        
        print(f"✅ Preprocessing Success! Saved to: {save_path}")
    except Exception as e:
        print(f"Error during Preprocessing: {e}")
        return

    # --- PHASE 2: Model Training & Evaluation ---
    print("\n" + "-" * 60)
    print("[PHASE 2] Starting Model Training & Evaluation...")
    print("-" * 60)
    
    try:
        # Execute the training logic defined in src/train.py
        train.run_training()
        print("\n" + "=" * 60)
        print("Full Pipeline Executed Successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"Error during Training: {e}")

if __name__ == "__main__":
    main()