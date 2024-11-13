from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import time

class TextClassifierANN:
    def __init__(self, hidden_layer_size=100):
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size,),
            max_iter=500,
            random_state=42,
            verbose=True,
            early_stopping=True,
            validation_fraction=0.1
        )
        
    def train(self, X, y, progress_bar=None):
        """Entraîne le réseau de neurones"""
        start_time = time.time()
        self.model.fit(X, y)
        if progress_bar:
            progress_bar.set_description(f"Durée: {time.time() - start_time:.2f}s")
            progress_bar.update(1)
        
    def predict(self, X):
        """Fait des prédictions"""
        return self.model.predict(X)
        
    def evaluate(self, X, y_true, pbar=None):
        """Évalue le modèle et retourne les métriques"""
        start_time = time.time()
        y_pred = self.predict(X)
        if pbar:
            pbar.set_description(f"Évaluation ({time.time() - start_time:.2f}s)")
            pbar.update(len(y_true))
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }