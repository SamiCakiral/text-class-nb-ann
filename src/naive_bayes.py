import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from collections import defaultdict
from tqdm import tqdm

class CustomNaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Initialise le classifieur Naive Bayes avec lissage de Laplace et Good-Turing.

        Attributs:
            models (dict): Stocke les modèles de probabilités pour chaque n-gram et classe
            vocabularies (dict): Stocke le vocabulaire pour chaque n-gram
            class_priors (dict): Probabilités a priori des classes pour chaque n-gram
            word_counts (dict): Compte des mots pour chaque n-gram et classe
            n1 (dict): Nombre de mots vus une seule fois (pour Good-Turing)
            n0 (dict): Estimation des mots jamais vus (pour Good-Turing)
            alpha (float): Paramètre de lissage de Laplace
            interpolation_weights (dict): Poids pour l'interpolation des différents n-grams

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
        """
        Calcule les statistiques nécessaires pour le lissage Good-Turing de manière vectorisée.
        
        Algorithme:
        1. Calcule la fréquence de chaque mot dans le corpus (word_counts)
        2. Compte combien de mots apparaissent exactement k fois (word_freq)
        3. Calcule N1 (nombre de mots vus une seule fois)
        4. Estime N0 (nombre de mots jamais vus) avec la formule: N0 = N1²/(2*N2)
        
        Args:
            X (sparse matrix): Matrice de caractéristiques
            n_gram (int): Taille du n-gram

        Returns:
            int: Nombre total de mots dans le corpus
        """
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
        Entraîne le modèle Naive Bayes avec lissage de Laplace et Good-Turing.

        Algorithme:
        1. Calcul des probabilités a priori P(classe)
        2. Pour chaque classe:
           - Sélectionne les documents de la classe
           - Applique le lissage de Laplace aux comptes de mots
           - Calcule P(mot|classe) = (count + alpha) / (total + alpha*|V|)
           - Stocke les log-probabilités pour optimiser les calculs
        3. Calcule les statistiques Good-Turing pour le n-gram

        Args:
            X (sparse matrix): Matrice de caractéristiques (documents × mots)
            y (array): Labels des documents
            n_gram (int): Taille du n-gram
            pbar (tqdm): Barre de progression optionnelle
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
        """
        Effectue la prédiction vectorisée standard sans interpolation.

        Algorithme:
        1. Pour chaque classe:
           - Calcule log P(classe)
           - Pour chaque document:
             * Somme log P(mot|classe) pour les mots présents
        2. Retourne la classe avec le score maximum

        Args:
            X (sparse matrix): Matrice de caractéristiques à prédire
            n_gram (int): Taille du n-gram

        Returns:
            array: Classes prédites pour chaque document
        """
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
        """
        Effectue la prédiction avec interpolation linéaire des n-grams.

        Algorithme:
        1. Pour chaque n-gram (1, 2, 3):
           - Calcule les scores comme dans predict()
           - Applique le lissage Good-Turing pour les mots non vus
           - Pondère les scores selon interpolation_weights
        2. Combine les scores de tous les n-grams
        3. Retourne la classe avec le score maximum

        Args:
            X_dict (dict): Dictionnaire des matrices de caractéristiques par n-gram
            labels (array): Liste des labels possibles

        Returns:
            array: Classes prédites pour chaque document
        """
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
        """
        Évalue les performances du modèle sans interpolation.

        Métriques calculées:
        - Accuracy: Proportion de prédictions correctes
        - Recall pondéré: Moyenne du recall par classe, pondérée par leur fréquence
        - Matrice de confusion: Visualisation des prédictions vs réalité

        Args:
            X (sparse matrix): Matrice de caractéristiques à évaluer
            y_true (array): Labels réels
            n_gram (int): Taille du n-gram

        Returns:
            dict: Dictionnaire contenant accuracy, recall et matrice de confusion
        """
        
        y_pred = self.predict(X, n_gram)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    def evaluate_interpolation(self, X_dict, y_true):
        """
        Évalue les performances du modèle avec interpolation.

        Métriques calculées:
        - Accuracy: Proportion de prédictions correctes
        - Recall pondéré: Moyenne du recall par classe, pondérée par leur fréquence
        - Matrice de confusion: Visualisation des prédictions vs réalité

        Args:
            X_dict (dict): Dictionnaire des matrices de caractéristiques par n-gram
            y_true (array): Labels réels

        Returns:
            dict: Dictionnaire contenant accuracy, recall et matrice de confusion
        """
        
        y_pred = self.predict_with_interpolation(X_dict, np.unique(y_true))
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }