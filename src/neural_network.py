from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

class TextClassifierANN:
    def __init__(self, hidden_layer_size=100):
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size,),
            max_iter=500,
            random_state=42
        )
        
    def train(self, X, y):
        """Entraîne le réseau de neurones"""
        self.model.fit(X, y)
        
    def predict(self, X):
        """Fait des prédictions"""
        return self.model.predict(X)
        
    def evaluate(self, X, y_true):
        """Évalue le modèle et retourne les métriques"""
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }