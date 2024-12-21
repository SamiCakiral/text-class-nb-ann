import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from collections import defaultdict
from tqdm import tqdm

class CustomNaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Initialisation du classifieur Naive Bayes
        
        Args:
            alpha (float): Paramètre de lissage Laplace (default: 1.0)
        """
        self.models = {}
        self.vocabularies = {}
        self.class_priors = {}
        self.word_counts = {}
        self.n1 = {}  # Pour Good-Turing: nombre de mots vus une seule fois
        self.n0 = {}  # Pour Good-Turing: estimation des mots non vus
        self.alpha = alpha
        self.interpolation_weights = {
            1: 0.5,  # Poids pour unigrams
            2: 0.3,  # Poids pour bigrams
            3: 0.2   # Poids pour trigrams
        }

    def _calculate_good_turing_counts(self, X, n_gram):
        """Calcule les statistiques nécessaires pour Good-Turing smoothing de manière vectorisée"""
        # Calcul vectorisé des fréquences
        word_counts = X.sum(axis=0).A1  # .A1 convertit la matrice sparse en array 1D
        unique_counts, count_freq = np.unique(word_counts, return_counts=True)
        
        # Création du dictionnaire de fréquences
        word_freq = dict(zip(unique_counts, count_freq))
        
        # Calcule N1 (nombre de mots vus une seule fois)
        self.n1[n_gram] = word_freq.get(1, 0)
        
        # Estime N0 (nombre de mots non vus)
        self.n0[n_gram] = self.n1[n_gram] ** 2 / (2 * word_freq.get(2, 1))
        
        return word_counts.sum()

    def train_model(self, X, y, n_gram, pbar=None):
        """
        Entraîne le modèle Naive Bayes avec Laplace smoothing
        
        Args:
            X: Matrice de caractéristiques
            y: Labels
            n_gram: Taille du n-gram
            pbar: Barre de progression
        """
        # Calcul des probabilités a priori des classes
        self.class_priors[n_gram] = {}
        classes = np.unique(y)
        total_docs = len(y)
        
        # Stockage des comptes pour Good-Turing
        self._calculate_good_turing_counts(X, n_gram)
        
        # Pour chaque classe avec barre de progression
        for c in tqdm(classes, desc="Entraînement par classe"):
            # Probabilité a priori de la classe
            class_docs = (y == c).sum()
            self.class_priors[n_gram][c] = class_docs / total_docs
            
            # Sélection des documents de la classe
            X_c = X[y == c]
            
            # Calcul des probabilités des mots avec Laplace smoothing
            word_counts = X_c.sum(axis=0).A1 + self.alpha  # Convertit en array 1D
            total_words = word_counts.sum()
            
            # Éviter la division par zéro et le log(0)
            word_probs = np.maximum(word_counts / total_words, np.finfo(float).tiny)
            
            # Stockage des probabilités
            if n_gram not in self.models:
                self.models[n_gram] = {}
            self.models[n_gram][c] = np.log(word_probs)
            
            # Stockage des comptes pour Good-Turing
            self.word_counts[(n_gram, c)] = word_counts
        
        if pbar:
            pbar.update(100)

    def predict(self, X, n_gram):
        """Fait des prédictions en utilisant Good-Turing pour les mots non vus"""
        if n_gram not in self.models:
            raise ValueError(f"Pas de modèle entraîné pour {n_gram}-gram")
        
        predictions = []
        classes = list(self.models[n_gram].keys())
        
        # Prédiction pour chaque document avec barre de progression
        for i in tqdm(range(X.shape[0]), desc="Prédiction"):
            scores = {}
            for c in classes:
                # Score initial avec la probabilité a priori
                score = np.log(self.class_priors[n_gram][c])
                
                # Ajout des scores des mots avec Good-Turing smoothing
                present_words = X[i].nonzero()[1]
                for word_idx in present_words:
                    if word_idx < len(self.models[n_gram][c]):  # Vérification de l'index
                        if X[i, word_idx] > 0:
                            score += self.models[n_gram][c][word_idx]
                        else:
                            # Utilisation de Good-Turing pour les mots non vus
                            score += np.log(self.n1[n_gram] / 
                                          (self.n0[n_gram] * self.word_counts[(n_gram, c)].sum()))
                
                scores[c] = score
            
            predictions.append(max(scores.items(), key=lambda x: x[1])[0])
        
        return np.array(predictions)

    def predict_with_interpolation(self, X_dict, labels):
        """
        Fait des prédictions en utilisant l'interpolation des modèles n-gram
        
        Args:
            X_dict: Dictionnaire contenant les matrices de caractéristiques pour chaque n-gram
            labels: Labels possibles
        """
        predictions = []
        
        # Prédiction pour chaque document avec barre de progression
        for i in tqdm(range(X_dict[1].shape[0]), desc="Prédiction avec interpolation"):
            scores = {label: 0 for label in labels}
            
            for label in labels:
                # Combinaison des scores de chaque modèle n-gram
                for n_gram in [1, 2, 3]:
                    if n_gram in X_dict:
                        X = X_dict[n_gram]
                        score = np.log(self.class_priors[n_gram][label])
                        
                        present_words = X[i].nonzero()[1]
                        for word_idx in present_words:
                            if X[i, word_idx] > 0:
                                score += self.models[n_gram][label][word_idx]
                            else:
                                score += np.log(self.n1[n_gram] / 
                                              (self.n0[n_gram] * self.word_counts[(n_gram, label)].sum()))
                        
                        scores[label] += self.interpolation_weights[n_gram] * score
            
            predictions.append(max(scores.items(), key=lambda x: x[1])[0])
        
        return np.array(predictions)

    def evaluate(self, X, y_true, n_gram, pbar=None):
        """Évalue le modèle et retourne les métriques"""
        y_pred = self.predict(X, n_gram)
        if pbar:
            pbar.update(len(y_true))
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    def evaluate_interpolation(self, X_dict, y_true, pbar=None):
        """Évalue le modèle avec interpolation et retourne les métriques"""
        y_pred = self.predict_with_interpolation(X_dict, np.unique(y_true))
        if pbar:
            pbar.update(len(y_true))
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }