"""
Dataset Download Script.

Downloads the CHB-MIT EEG dataset from Kaggle.
Requires Kaggle API credentials to be configured.

Usage:
    python download_data.py
    
Setup:
    1. Create a Kaggle account at https://www.kaggle.com
    2. Go to Account > API > Create New Token
    3. Place the downloaded kaggle.json in:
       - Windows: C:\\Users\\<user>\\.kaggle\\kaggle.json
       - Linux/Mac: ~/.kaggle/kaggle.json
"""

import os
import sys
import zipfile
import shutil


def download_dataset(target_dir="data/raw"):
    """Download CHB-MIT dataset from Kaggle."""
    
    # Check for Kaggle credentials
    kaggle_json_paths = [
        os.path.expanduser("~/.kaggle/kaggle.json"),
        os.path.join(os.environ.get("USERPROFILE", ""), ".kaggle", "kaggle.json"),
    ]
    
    cred_found = any(os.path.exists(p) for p in kaggle_json_paths)
    
    if not cred_found:
        print("=" * 60)
        print("KAGGLE API CREDENTIALS NOT FOUND")
        print("=" * 60)
        print()
        print("To download the CHB-MIT dataset, you need Kaggle API credentials.")
        print()
        print("Option 1: Set up kaggle.json")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' under API section")
        print("  3. Place the downloaded kaggle.json file in:")
        if sys.platform == "win32":
            print(f"     C:\\Users\\{os.getlogin()}\\.kaggle\\kaggle.json")
        else:
            print("     ~/.kaggle/kaggle.json")
        print()
        print("Option 2: Set environment variables")
        print("  set KAGGLE_USERNAME=your_username")
        print("  set KAGGLE_KEY=your_api_key")
        print()
        print("Option 3: Manual download")
        print("  1. Go to https://www.kaggle.com/datasets/borshalickson/chb-mit-data")
        print("  2. Click 'Download' button")
        print("  3. Extract the ZIP into the data/raw/ directory")
        print("  4. Ensure structure: data/raw/chb01/, data/raw/chb02/, etc.")
        print()
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Downloading CHB-MIT dataset to {target_dir}...")
        print("This is a ~25GB dataset. Download may take a while.")
        
        api.dataset_download_files(
            "borshalickson/chb-mit-data",
            path=target_dir,
            unzip=True,
        )
        
        print(f"Dataset downloaded and extracted to {target_dir}")
        
        # Check if files are in a subdirectory and move them up
        subdirs = [d for d in os.listdir(target_dir) 
                   if os.path.isdir(os.path.join(target_dir, d))]
        
        if len(subdirs) == 1 and not subdirs[0].startswith("chb"):
            # Files are inside a single subdirectory
            inner_dir = os.path.join(target_dir, subdirs[0])
            for item in os.listdir(inner_dir):
                src = os.path.join(inner_dir, item)
                dst = os.path.join(target_dir, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            os.rmdir(inner_dir)
            print(f"Moved files from {inner_dir} to {target_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease download manually from:")
        print("  https://www.kaggle.com/datasets/borshalickson/chb-mit-data")
        print(f"Extract to: {target_dir}")
        return False


if __name__ == "__main__":
    success = download_dataset()
    if success:
        print("\nDataset ready! Run the pipeline:")
        print("  python run_pipeline.py")
    else:
        print("\nPlease download the dataset manually and try again.")
