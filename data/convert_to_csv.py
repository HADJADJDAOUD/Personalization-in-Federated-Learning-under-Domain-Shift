"""
Convert UCI Heart Disease .data files to CSV
Team 8-HANTA — Week 2 Data Preparation
"""

import pandas as pd
import numpy as np
import os

# The 14 standard columns used in all 4 centers
COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Map center filenames to readable names
FILES = {
    'processed.cleveland.data':  'Cleveland',
    'processed.hungarian.data':  'Hungary',
    'processed.switzerland.data':'Switzerland',
    'processed.va.data':         'LongBeach',
}

def load_data_file(filepath, center_name):
    """Load a UCI .data file and return a clean DataFrame."""
    df = pd.read_csv(
        filepath,
        header=None,
        names=COLUMNS,
        na_values='?',      # UCI uses '?' for missing values
    )
    
    # Binarize target: 0 = no disease, 1 = disease (values 1-4 → 1)
    df['target'] = (df['target'] > 0).astype(int)
    
    # Add center label
    df['center'] = center_name
    
    return df


def main():
    # ── Set this to the folder containing your .data files ────────────────
    data_dir = r'./heart_disease_dataset'   # run the script from inside the heart_disease_dataset folder
    #   OR use the full path:
    # data_dir = r'D:\Study\2sc_2 semestre\MLA\Personalization-in-Federated-Learning-under-Domain_Shift\data\heart_disease_dataset'
    # ──────────────────────────────────────────────────────────────────────

    dfs = []
    for filename, center_name in FILES.items():
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found: {filepath}")
            continue
        
        df = load_data_file(filepath, center_name)
        
        print(f"\n{'='*50}")
        print(f"Center: {center_name}  ({filename})")
        print(f"  Rows: {len(df)}")
        print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")
        print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
        print(f"  Positive rate: {df['target'].mean():.3f}")
        
        # Save individual center CSV
        out_name = f'heart_{center_name.lower()}.csv'
        out_path = os.path.join(data_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_name}")
        
        dfs.append(df)

    # Combine all centers into one CSV
    df_all = pd.concat(dfs, ignore_index=True)
    combined_path = os.path.join(data_dir, 'heart_disease_all.csv')
    df_all.to_csv(combined_path, index=False)
    
    print(f"\n{'='*50}")
    print(f"COMBINED DATASET")
    print(f"  Total rows: {len(df_all)}")
    print(f"  Columns: {list(df_all.columns)}")
    print(f"  Missing values:")
    print(df_all.isnull().sum()[df_all.isnull().sum()>0].to_string())
    print(f"\nSamples per center:")
    print(df_all.groupby('center')['target'].agg(['count','mean']).rename(columns={'mean':'pos_rate'}).round(3))
    print(f"\nSaved combined → heart_disease_all.csv")
    print(f"\nDone! ✓")


if __name__ == '__main__':
    main()