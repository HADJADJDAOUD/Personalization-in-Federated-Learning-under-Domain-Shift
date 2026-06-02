# Evaluation utilities for FL experiments
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba):
    """Compute a consistent set of binary classification metrics."""
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = np.nan

    try:
        auc_pr = average_precision_score(y_true, y_proba)
    except ValueError:
        auc_pr = np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "auc_pr": auc_pr,
        "brier": brier_score_loss(y_true, y_proba),
    }

class LocalBaseline:
    """Local-only baseline: each client trains independently."""

    def __init__(self, max_iter=1000, model_params=None):
        self.max_iter = max_iter
        self.model_params = model_params or {}
        self.models = {}
        
    def train(self, client_data):
        """Train independent model for each client"""
        for center, data in client_data.items():
            model = LogisticRegression(
                max_iter=self.max_iter,
                random_state=42,
                class_weight="balanced",
                **self.model_params,
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
            
            results[center] = compute_metrics(data["y_test"], y_pred, y_pred_proba)
        
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
            'Balanced Acc': f"{result['balanced_accuracy_mean']:.3f} ± {result['balanced_accuracy_std']:.3f}",
            'F1': f"{result['f1_mean']:.3f} ± {result['f1_std']:.3f}",
            'AUC': f"{result['auc_mean']:.3f} ± {result['auc_std']:.3f}",
            'AUC-PR': f"{result['auc_pr_mean']:.3f} ± {result['auc_pr_std']:.3f}",
            'Brier': f"{result['brier_mean']:.3f} ± {result['brier_std']:.3f}",
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
                'Balanced Acc': result.get(f'balanced_accuracy_{center}', np.nan),
                'F1': result.get(f'f1_{center}', np.nan),
                'AUC': result.get(f'auc_{center}', np.nan),
                'AUC-PR': result.get(f'auc_pr_{center}', np.nan),
                'Brier': result.get(f'brier_{center}', np.nan),
            }
            data.append(row)
    
    return pd.DataFrame(data)


def tune_logreg_c(client_data, c_values, metric="auc_pr", max_iter=1000):
    """Select a global C by averaging validation performance across clients."""
    scores = {}
    for c_value in c_values:
        metrics_per_center = []
        for _, data in client_data.items():
            model = LogisticRegression(
                max_iter=max_iter,
                random_state=42,
                class_weight="balanced",
                C=c_value,
            )
            model.fit(data["X_train"], data["y_train"])
            y_pred = model.predict(data["X_val"])
            y_proba = model.predict_proba(data["X_val"])[:, 1]
            metrics = compute_metrics(data["y_val"], y_pred, y_proba)
            metrics_per_center.append(metrics[metric])
        scores[c_value] = float(np.mean(metrics_per_center))

    best_c = max(scores, key=scores.get)
    return best_c, scores
