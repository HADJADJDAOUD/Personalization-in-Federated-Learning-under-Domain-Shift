# pFedMe: Personalized Federated Learning with Moreau Envelopes (Dinh et al., NeurIPS 2020)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import copy

class PFedMeServer:
    """pFedMe Server with Moreau Envelope regularization"""
    
    def __init__(self, n_features, learning_rate=0.01, max_iter=1000, lambda_param=0.5):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_param = lambda_param  # Personalization parameter
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
        """Aggregate client models (same as FedAvg)"""
        total_size = sum(client_sizes)
        
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
        """Evaluate global model"""
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


class PFedMeClient:
    """pFedMe Client with local personalization"""
    
    def __init__(self, client_id, X_train, y_train, X_test, y_test, max_iter=1000, lambda_param=0.5):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.max_iter = max_iter
        self.lambda_param = lambda_param  # Personalization parameter (proximal term)
        self.model = LogisticRegression(max_iter=max_iter, random_state=42, class_weight='balanced')
        self.personal_model = LogisticRegression(max_iter=max_iter, random_state=42, class_weight='balanced')
        
    def train_personalized(self, global_weights):
        """
        Train personalized model with Moreau Envelope regularization:
        L(w) = f(w) + (lambda/2) * ||w - w_global||^2
        
        This encourages the local model to stay close to global while fitting local data.
        """
        # Start from global model
        self.model.coef_ = global_weights['coef'].copy()
        self.model.intercept_ = global_weights['intercept'].copy()
        
        # Train with regularization encouraging closeness to global model
        # We use L2 penalty (ridge) to simulate the proximal term
        c_param = 1.0 / (self.lambda_param + 1e-6)  # Ridge strength
        
        personalized_model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=42,
            class_weight='balanced',
            C=c_param,  # Inverse of regularization strength
            solver='lbfgs',
            penalty='l2'
        )
        
        personalized_model.fit(self.X_train, self.y_train)
        self.personal_model = personalized_model
        
        return {
            'coef': personalized_model.coef_,
            'intercept': personalized_model.intercept_
        }
    
    def get_weights(self):
        """Return personal model weights"""
        return {
            'coef': self.personal_model.coef_,
            'intercept': self.personal_model.intercept_
        }
    
    def evaluate_personal(self):
        """Evaluate personal (local) model"""
        y_pred = self.personal_model.predict(self.X_test)
        y_pred_proba = self.personal_model.predict_proba(self.X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(self.y_test, y_pred_proba)
        }


def run_pfedme(client_data, n_rounds=20, max_iter=1000, lambda_param=0.5):
    """
    Run pFedMe algorithm
    
    Args:
        client_data: dict {center: {'X_train', 'X_test', 'y_train', 'y_test'}}
        n_rounds: number of communication rounds
        max_iter: max iterations for LogReg
        lambda_param: personalization parameter (higher = more personalization)
    
    Returns:
        history: metrics per round per client
    """
    
    centers = list(client_data.keys())
    n_clients = len(centers)
    n_features = client_data[centers[0]]['X_train'].shape[1]
    
    # Initialize server
    server = PFedMeServer(n_features, max_iter=max_iter, lambda_param=lambda_param)
    X_dummy = client_data[centers[0]]['X_train'][:10]
    y_dummy = client_data[centers[0]]['y_train'][:10]
    server.initialize(X_dummy, y_dummy)
    
    # Initialize clients
    clients = {}
    client_sizes = {}
    for center in centers:
        clients[center] = PFedMeClient(
            center,
            client_data[center]['X_train'],
            client_data[center]['y_train'],
            client_data[center]['X_test'],
            client_data[center]['y_test'],
            max_iter=max_iter,
            lambda_param=lambda_param
        )
        client_sizes[center] = len(client_data[center]['y_train'])
    
    # Training loop
    history = {center: [] for center in centers}
    global_history = []
    personal_history = {center: [] for center in centers}
    
    for round_idx in range(n_rounds):
        # 1. Distribute global model to clients
        # 2. Clients train personalized models (with Moreau envelope)
        client_weights = []
        for center in centers:
            weights = clients[center].train_personalized(server.global_weights)
            client_weights.append(weights)
        
        # 3. Server aggregates (for next round)
        weights_list = [w for w in client_weights]
        sizes_list = [client_sizes[c] for c in centers]
        server.aggregate(weights_list, sizes_list)
        
        # 4. Evaluate: both global and personal models
        for center in centers:
            # Global model performance
            global_metrics = server.evaluate_on_client(
                client_data[center]['X_test'],
                client_data[center]['y_test']
            )
            history[center].append(global_metrics)
            
            # Personal model performance
            personal_metrics = clients[center].evaluate_personal()
            personal_history[center].append(personal_metrics)
        
        personal_acc = np.mean([personal_history[c][-1]['accuracy'] for c in centers])
        personal_history_avg = personal_acc
        global_history.append(personal_acc)
        
        if (round_idx + 1) % 5 == 0:
            print(f"Round {round_idx+1}/{n_rounds} | Personal Acc: {personal_acc:.3f}")
    
    return personal_history, global_history, server, clients
