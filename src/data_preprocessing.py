import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import re
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import numpy as np

class DataPreprocessor:
    def __init__(self, n_splits=5, random_state=42):
        # Télécharger les ressources NLTK nécessaires
        nltk.download('punkt') # Tokenize les phrases en mots ex: "Hello, how are you?" -> ["Hello", "how", "are", "you"]
        nltk.download('stopwords') # Telecharge des mots tels que "the", "is", "at", etc.
        nltk.download('averaged_perceptron_tagger') # Tag les mots avec leur partie du discours ex: "The cat sleeps" -> [("The", "NN"), ("cat", "NN"), ("sleeps", "VBZ")]
        self.stop_words = set(stopwords.words('english')) # Supprime les mots communs comme "the", "is", "at", etc.
        # Initialisation des paramètres de cross-validation
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.folds = None
        self.train_data = None  # Ajout d'un attribut pour les données d'entraînement
        self.test_data = None   # Ajout d'un attribut pour les données de test
        self.preprocessed_data = None

    def load_data(self, file_path):
        """Charge les données à partir du fichier CSV"""
        try:
            # Utilisation de pandas pour lire le CSV avec le bon séparateur et les bonnes colonnes
            df = pd.read_csv(file_path, 
                            names=['Class', 'Title', 'Description'],  # noms des colonnes
                            skiprows=1)  # skip l'en-tête
            
            # Gestion des valeurs manquantes si nécessaire
            df = df.fillna('')
            
            # Combine title et description pour le texte complet
            df['Text'] = df['Title'] + ' ' + df['Description']
            
            # Conversion de la colonne Class en type numérique
            df['Class'] = pd.to_numeric(df['Class'])
            
            print(f"Données chargées : {len(df)} lignes")
            print(f"Distribution des classes :\n{df['Class'].value_counts()}")
            
            return df
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return None

    def clean_text(self, text):
        """Nettoie le texte"""
        # Convertir en minuscules
        text = text.lower()
        # Supprimer les caractères spéciaux
        text = re.sub(r'[^\w\s]', ' ', text) # \w : lettres, chiffres, underscores, \s : espaces
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, df, is_train=True):
        """Prétraitement complet des données"""
        print("Nettoyage des textes...")
        df['Cleaned_Text'] = [self.clean_text(text) for text in tqdm(df['Text'], desc="Nettoyage")]
        
        print("Tokenisation...")
        df['Tokens'] = [word_tokenize(text) for text in tqdm(df['Cleaned_Text'], desc="Tokenisation")]
        
        print("Filtrage des stop words...")
        df['Tokens'] = [
            [token for token in tokens if token not in self.stop_words]
            for tokens in tqdm(df['Tokens'], desc="Filtrage stop words")
        ]
        
        if is_train:
            self.train_data = df
        else:
            self.test_data = df
            
        self.preprocessed_data = df
        return df

    def create_folds(self):
        """Crée les folds pour la validation croisée"""
        if self.preprocessed_data is None:
            raise ValueError("Les données doivent d'abord être prétraitées")
            
        print("\nCréation des folds pour la validation croisée...")
        X = self.preprocessed_data['Cleaned_Text'].values
        y = self.preprocessed_data['Class'].values
        
        self.folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            fold_info = {
                'train_idx': train_idx,
                'val_idx': val_idx,
                'train_dist': pd.Series(y[train_idx]).value_counts().to_dict(),
                'val_dist': pd.Series(y[val_idx]).value_counts().to_dict()
            }
            self.folds.append(fold_info)
            
            print(f"\nFold {fold_idx + 1}/{self.n_splits}")
            print(f"Distribution train: {fold_info['train_dist']}")
            print(f"Distribution validation: {fold_info['val_dist']}")
        
        return self.folds

    def get_fold(self, fold_idx):
        """Récupère les données pour un fold spécifique"""
        if self.folds is None:
            raise ValueError("Les folds doivent d'abord être créés avec create_folds()")
        
        if fold_idx >= self.n_splits:
            raise ValueError(f"L'index du fold doit être < {self.n_splits}")
            
        fold = self.folds[fold_idx]
        
        # Utilisation des données d'entraînement pour la validation croisée
        if self.train_data is None:
            raise ValueError("Les données d'entraînement doivent d'abord être prétraitées")
        
        # Récupération des données d'entraînement uniquement
        texts = self.train_data['Cleaned_Text'].values
        labels = self.train_data['Class'].values
        
        return {
            'X_train': texts[fold['train_idx']],
            'X_val': texts[fold['val_idx']],
            'y_train': labels[fold['train_idx']],
            'y_val': labels[fold['val_idx']]
        }

    def get_all_data(self):
        """Retourne toutes les données prétraitées"""
        if self.preprocessed_data is None:
            raise ValueError("Les données doivent d'abord être prétraitées")
        
        return {
            'X': self.preprocessed_data['Cleaned_Text'].values,
            'y': self.preprocessed_data['Class'].values
        }

    def get_pos_tags(self, tokens):
        """Obtient les tags POS pour les tokens"""
        return pos_tag(tokens)