# Evaluation utilities for FL experiments
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class LocalBaseline:
    """Local-only baseline: each client trains independently"""
    
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.models = {}
        
    def train(self, client_data):
        """Train independent model for each client"""
        for center, data in client_data.items():
            model = LogisticRegression(
                max_iter=self.max_iter,
                random_state=42,
                class_weight='balanced'
            )
            model.fit(data['X_train'], data['y_train'])
            self.models[center] = model
    
    def evaluate(self, client_data):
        """Evaluate local models"""
        results = {}
        for center, data in client_data.items():
            model = self.models[center]
            y_pred = model.predict(data['X_test'])
            y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
            
            results[center] = {
                'accuracy': accuracy_score(data['y_test'], y_pred),
                'precision': precision_score(data['y_test'], y_pred, zero_division=0),
                'recall': recall_score(data['y_test'], y_pred, zero_division=0),
                'f1': f1_score(data['y_test'], y_pred, zero_division=0),
                'auc': roc_auc_score(data['y_test'], y_pred_proba)
            }
        
        return results


def summarize_results(results_dict, method_name):
    """
    Summarize results across clients
    
    Args:
        results_dict: {center: {metric: value}}
        method_name: name of method
    
    Returns:
        summary dict with mean and std
    """
    centers = list(results_dict.keys())
    metrics = list(results_dict[centers[0]].keys())
    
    summary = {
        'method': method_name,
        'n_clients': len(centers)
    }
    
    for metric in metrics:
        values = [results_dict[c][metric] for c in centers]
        summary[f'{metric}_mean'] = np.mean(values)
        summary[f'{metric}_std'] = np.std(values)
        
        for center in centers:
            summary[f'{metric}_{center}'] = results_dict[center][metric]
    
    return summary


def create_results_table(all_results):
    """
    Create comparison table for methods
    
    Args:
        all_results: list of result dicts from summarize_results
    
    Returns:
        pandas DataFrame
    """
    data = []
    for result in all_results:
        method = result['method']
        row = {
            'Method': method,
            'Accuracy': f"{result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}",
            'F1': f"{result['f1_mean']:.3f} ± {result['f1_std']:.3f}",
            'AUC': f"{result['auc_mean']:.3f} ± {result['auc_std']:.3f}",
        }
        data.append(row)
    
    return pd.DataFrame(data)


def get_per_client_table(all_results, centers):
    """
    Create per-client breakdown table
    
    Args:
        all_results: list of result dicts
        centers: list of center names
    
    Returns:
        pandas DataFrame
    """
    data = []
    for result in all_results:
        method = result['method']
        for center in centers:
            row = {
                'Method': method,
                'Center': center,
                'Accuracy': result.get(f'accuracy_{center}', np.nan),
                'F1': result.get(f'f1_{center}', np.nan),
                'AUC': result.get(f'auc_{center}', np.nan),
            }
            data.append(row)
    
    return pd.DataFrame(data)
