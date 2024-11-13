import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import re
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self):
        # Télécharger les ressources NLTK nécessaires
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        self.stop_words = set(stopwords.words('english'))

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
        text = re.sub(r'[^\w\s]', ' ', text)
        # Supprimer les chiffres
        text = re.sub(r'\d+', '', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, df):
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
        return df

    def get_pos_tags(self, tokens):
        """Obtient les tags POS pour les tokens"""
        return pos_tag(tokens)