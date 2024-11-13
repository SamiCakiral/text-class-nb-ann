import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

class CustomNaiveBayes:
    def __init__(self):
        self.models = {}
        self.vocabularies = {}

    def train_model(self, X, y, n_gram):
        """Entraîne le modèle Naive Bayes pour un n-gram spécifique"""
        model = MultinomialNB(alpha=1.0)  # Lissage de Laplace
        model.fit(X, y)
        self.models[n_gram] = model
        
    def predict(self, X, n_gram):
        """Fait des prédictions avec le modèle n-gram spécifié"""
        if n_gram not in self.models:
            raise ValueError(f"Pas de modèle entraîné pour {n_gram}-gram")
        return self.models[n_gram].predict(X)

    def evaluate(self, X, y_true, n_gram):
        """Évalue le modèle et retourne les métriques"""
        y_pred = self.predict(X, n_gram)
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_true, y_pred)
        return {
            'accuracy': accuracy,
            'recall': recall,
            'confusion_matrix': conf_matrix
        }