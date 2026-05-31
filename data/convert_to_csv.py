
"""
Convert UCI Heart Disease .data files to CSV.
Thin wrapper — actual loading logic lives in data_pipeline.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import load_client_data_detailed, CENTER_FILES, COLUMNS
import pandas as pd

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'heart_disease_dataset')

    centers = load_client_data_detailed(data_dir=data_dir, scale=False, test_size=0.20)

    dfs = []
    for filename, center_name in CENTER_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found: {filepath}")
            continue
        # Re-read raw (unscaled) for the CSV export
        df = pd.read_csv(filepath, header=None, names=COLUMNS, na_values='?')
        df['target'] = (df['target'] > 0).astype(int)
        df['center'] = center_name

        out_path = os.path.join(data_dir, f'heart_{center_name.lower()}.csv')
        df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(os.path.join(data_dir, 'heart_disease_all.csv'), index=False)
    print(f"\nSaved combined → heart_disease_all.csv  ({len(df_all)} rows)")

if __name__ == '__main__':
    main()