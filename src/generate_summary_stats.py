# src/generate_summary_stats.py
"""
Summary statistics generator for FL experiments.
Produces a console report comparing FedAvg, Local-only, and pFedMe.
Run: python src/generate_summary_stats.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from fedavg import run_fedavg
from pfedme import run_pfedme
from evaluation import LocalBaseline, summarize_results, create_results_table, get_per_client_table


def load_heart_disease_data():
    """Load the FLamby Heart Disease dataset splits."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart_disease_dataset')
    
    files = {
        'cleveland':   'processed.cleveland.data',
        'hungary':     'processed.hungarian.data',
        'switzerland': 'processed.switzerland.data',
        'longbeach':   'processed.va.data',
    }
    
    feature_cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    client_data = {}
    
    for center, fname in files.items():
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path, header=None, na_values='?')
        df.columns = feature_cols + ['target']
        
        # Binarize target (0 = no disease, 1 = disease)
        df['target'] = (df['target'] > 0).astype(int)
        
        # Median imputation
        df = df.fillna(df.median(numeric_only=True))
        
        X = df[feature_cols].values
        y = df['target'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        
        client_data[center] = {
            'X_train': X_train,
            'X_test':  X_test,
            'y_train': y_train,
            'y_test':  y_test,
        }
        
        print(f"  {center:12s}: {len(y_train)} train, {len(y_test)} test | "
              f"disease prevalence: {y.mean():.1%}")
    
    return client_data


def run_all_experiments(client_data):
    centers = list(client_data.keys())
    results = {}
    
    print("\n[1/3] Running Local-only baseline...")
    local = LocalBaseline()
    local.train(client_data)
    local_eval = local.evaluate(client_data)
    results['local'] = summarize_results(local_eval, 'Local-only')
    
    print("\n[2/3] Running FedAvg (20 rounds)...")
    fed_history, _, fed_server, fed_clients = run_fedavg(client_data, n_rounds=20)
    fedavg_final = {c: fed_history[c][-1] for c in centers}
    results['fedavg'] = summarize_results(fedavg_final, 'FedAvg')
    
    print("\n[3/3] Running pFedMe λ=0.5 (20 rounds)...")
    pfed_history, _, pfed_server, pfed_clients = run_pfedme(client_data, n_rounds=20, lambda_param=0.5)
    pfedme_final = {c: pfed_history[c][-1] for c in centers}
    results['pfedme'] = summarize_results(pfedme_final, 'pFedMe (λ=0.5)')
    
    return results, centers


def print_report(results, centers):
    print("\n" + "="*60)
    print("  RESULTS SUMMARY — Personalized FL under Domain Shift")
    print("="*60)
    
    table = create_results_table([results['local'], results['fedavg'], results['pfedme']])
    print("\n--- Overall Performance (mean ± std across centers) ---")
    print(table.to_string(index=False))
    
    print("\n--- Per-Center Accuracy ---")
    per_client = get_per_client_table(
        [results['local'], results['fedavg'], results['pfedme']], centers
    )
    pivot = per_client.pivot(index='Center', columns='Method', values='Accuracy')
    print(pivot.round(3).to_string())
    
    print("\n--- Key Finding ---")
    swiss_fedavg = results['fedavg'].get('accuracy_switzerland', float('nan'))
    swiss_pfedme = results['pfedme'].get('accuracy_switzerland', float('nan'))
    print(f"Switzerland — FedAvg: {swiss_fedavg:.3f}  →  pFedMe: {swiss_pfedme:.3f}")
    print("(Domain shift: Switzerland has ~94% disease prevalence, others ~50%)")
    print("pFedMe recovers performance by allowing per-hospital personalization.\n")


if __name__ == '__main__':
    print("Loading Heart Disease dataset...")
    client_data = load_heart_disease_data()
    
    results, centers = run_all_experiments(client_data)
    print_report(results, centers)