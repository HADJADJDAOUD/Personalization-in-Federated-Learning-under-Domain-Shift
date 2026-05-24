# FedAvg: Federated Averaging (McMahan et al., AISTATS 2017)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import copy

class FedAvgServer:
    """Federated Averaging Server"""
    
    def __init__(self, n_features, learning_rate=0.01, max_iter=1000):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.global_model = LogisticRegression(max_iter=max_iter, random_state=42, class_weight='balanced')
        self.global_weights = None
        
    def initialize(self, X_dummy, y_dummy):
        """Initialize global model"""
        self.global_model.fit(X_dummy, y_dummy)
        self.global_weights = {
            'coef': self.global_model.coef_.copy(),
            'intercept': self.global_model.intercept_.copy()
        }
    
    def aggregate(self, client_weights, client_sizes):
        """FedAvg aggregation: weighted average of client models"""
        total_size = sum(client_sizes)
        
        # Weighted average of coefficients
        avg_coef = np.zeros_like(client_weights[0]['coef'])
        avg_intercept = np.zeros_like(client_weights[0]['intercept'])
        
        for w, size in zip(client_weights, client_sizes):
            weight_factor = size / total_size
            avg_coef += weight_factor * w['coef']
            avg_intercept += weight_factor * w['intercept']
        
        self.global_weights = {
            'coef': avg_coef,
            'intercept': avg_intercept
        }
        
        return self.global_weights
    
    def evaluate_on_client(self, X_test, y_test):
        """Evaluate global model on client test data"""
        self.global_model.coef_ = self.global_weights['coef']
        self.global_model.intercept_ = self.global_weights['intercept']
        
        y_pred = self.global_model.predict(X_test)
        y_pred_proba = self.global_model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }


class FedAvgClient:
    """Federated Averaging Client"""
    
    def __init__(self, client_id, X_train, y_train, X_test, y_test, max_iter=1000):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.max_iter = max_iter
        self.model = LogisticRegression(max_iter=max_iter, random_state=42, class_weight='balanced')
        
    def train_with_global_init(self, global_weights, epochs=1):
        """Train local model initialized from global weights"""
        self.model.coef_ = global_weights['coef'].copy()
        self.model.intercept_ = global_weights['intercept'].copy()
        
        # Further train on local data
        self.model.fit(self.X_train, self.y_train)
        
        return {
            'coef': self.model.coef_,
            'intercept': self.model.intercept_
        }
    
    def get_weights(self):
        """Return local model weights"""
        return {
            'coef': self.model.coef_,
            'intercept': self.model.intercept_
        }
    
    def evaluate(self):
        """Evaluate local model"""
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(self.y_test, y_pred_proba)
        }


def run_fedavg(client_data, n_rounds=20, max_iter=1000):
    """
    Run FedAvg algorithm
    
    Args:
        client_data: dict {center: {'X_train', 'X_test', 'y_train', 'y_test'}}
        n_rounds: number of communication rounds
        max_iter: max iterations for LogReg
    
    Returns:
        history: dict with metrics per round
    """
    
    centers = list(client_data.keys())
    n_clients = len(centers)
    n_features = client_data[centers[0]]['X_train'].shape[1]
    
    # Initialize server
    server = FedAvgServer(n_features, max_iter=max_iter)
    X_dummy = client_data[centers[0]]['X_train'][:10]
    y_dummy = client_data[centers[0]]['y_train'][:10]
    server.initialize(X_dummy, y_dummy)
    
    # Initialize clients
    clients = {}
    client_sizes = {}
    for center in centers:
        clients[center] = FedAvgClient(
            center,
            client_data[center]['X_train'],
            client_data[center]['y_train'],
            client_data[center]['X_test'],
            client_data[center]['y_test'],
            max_iter=max_iter
        )
        client_sizes[center] = len(client_data[center]['y_train'])
    
    # Training loop
    history = {center: [] for center in centers}
    global_history = []
    
    for round_idx in range(n_rounds):
        # 1. Distribute global model to clients
        # 2. Clients train locally
        client_weights = []
        for center in centers:
            weights = clients[center].train_with_global_init(server.global_weights, epochs=1)
            client_weights.append(weights)
        
        # 3. Server aggregates
        weights_list = [w for w in client_weights]
        sizes_list = [client_sizes[c] for c in centers]
        server.aggregate(weights_list, sizes_list)
        
        # 4. Evaluate
        for center in centers:
            metrics = server.evaluate_on_client(
                client_data[center]['X_test'],
                client_data[center]['y_test']
            )
            history[center].append(metrics)
        
        global_acc = np.mean([history[c][-1]['accuracy'] for c in centers])
        global_history.append(global_acc)
        
        if (round_idx + 1) % 5 == 0:
            print(f"Round {round_idx+1}/{n_rounds} | Global Acc: {global_acc:.3f}")
    
    return history, global_history, server, clients
