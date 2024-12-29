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
        """Prédiction vectorisée standard"""
        if n_gram not in self.models:
            raise ValueError(f"Pas de modèle entraîné pour {n_gram}-gram")
        
        classes = list(self.models[n_gram].keys())
        n_samples = X.shape[0]
        n_classes = len(classes)
        
        # Initialisation de la matrice de scores
        scores = np.zeros((n_samples, n_classes))
        
        # Calcul vectorisé pour chaque classe
        for c_idx, c in enumerate(classes):
            # Log probabilité a priori
            scores[:, c_idx] = np.log(self.class_priors[n_gram][c])
            
            # Préparation du modèle
            model_matrix = np.zeros(X.shape[1])
            model_matrix[:len(self.models[n_gram][c])] = self.models[n_gram][c]
            
            # Calcul vectorisé des scores
            scores[:, c_idx] += X.multiply(model_matrix).sum(axis=1).A1
        
        # Retourne les classes prédites
        return np.array(classes)[np.argmax(scores, axis=1)]

    def predict_with_interpolation(self, X_dict, labels):
        """Prédiction vectorisée avec interpolation"""
        n_samples = X_dict[1].shape[0]
        n_classes = len(labels)
        
        # Initialisation des scores finaux
        final_scores = np.zeros((n_samples, n_classes))
        
        # Pour chaque n-gram
        for n_gram in [1, 2, 3]:
            if n_gram not in X_dict:
                continue
            
            X = X_dict[n_gram]
            weight = self.interpolation_weights[n_gram]
            
            # Calcul des scores pour ce n-gram
            scores = np.zeros((n_samples, n_classes))
            
            for c_idx, label in enumerate(labels):
                # Log probabilité a priori
                scores[:, c_idx] = np.log(self.class_priors[n_gram][label])
                
                # Préparation du modèle
                model_matrix = np.zeros(X.shape[1])
                model_matrix[:len(self.models[n_gram][label])] = self.models[n_gram][label]
                
                # Calcul vectorisé des scores
                present_words = X.multiply(model_matrix)
                scores[:, c_idx] += present_words.sum(axis=1).A1
                
                # Good-Turing pour les mots non vus - Version optimisée
                missing_words = X.multiply(model_matrix != 0)
                missing_words.data = 1 - missing_words.data  # Inverse les valeurs non nulles
                gt_score = np.log(self.n1[n_gram] / 
                                (self.n0[n_gram] * self.word_counts[(n_gram, label)].sum()))
                scores[:, c_idx] += gt_score * missing_words.sum(axis=1).A1
            
            # Ajout pondéré des scores de ce n-gram
            final_scores += weight * scores
        
        # Retourne les classes prédites
        return np.array(labels)[np.argmax(final_scores, axis=1)]

    def evaluate(self, X, y_true, n_gram):
        """Évaluation vectorisée"""
        
        y_pred = self.predict(X, n_gram)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    def evaluate_interpolation(self, X_dict, y_true):
        """Évaluation vectorisée pour l'interpolation"""
        
        y_pred = self.predict_with_interpolation(X_dict, np.unique(y_true))
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }