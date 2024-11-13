from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizers = {}
        self.top_features = {}

    def create_ngram_vectors(self, texts, n):
        """Crée les vecteurs n-gram"""
        with tqdm(total=2, desc=f"Création vecteurs {n}-gram") as pbar:
            vectorizer = CountVectorizer(ngram_range=(n, n))
            pbar.set_description(f"Fitting {n}-gram vectorizer")
            vectors = vectorizer.fit_transform(texts)
            pbar.update(1)
            
            pbar.set_description(f"Sauvegarde {n}-gram vectorizer")
            self.count_vectorizers[n] = vectorizer
            pbar.update(1)
            
        return vectors

    def extract_tfidf_features(self, texts, max_features=1000):
        """Extrait les caractéristiques TF-IDF"""
        with tqdm(total=2, desc="Extraction TF-IDF") as pbar:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
            pbar.set_description("Fitting TF-IDF vectorizer")
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            pbar.update(1)
            
            pbar.set_description("Finalisation TF-IDF")
            pbar.update(1)
            
        return tfidf_matrix

    def get_top_words(self, n_words=15):
        """Obtient les mots avec les scores TF-IDF les plus élevés"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer n'a pas encore été entraîné")
        
        with tqdm(total=3, desc="Extraction top mots") as pbar:
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            pbar.update(1)
            
            scores = self.tfidf_vectorizer.idf_
            word_scores = list(zip(feature_names, scores))
            pbar.update(1)
            
            pbar.set_description("Tri des mots")
            word_scores.sort(key=lambda x: x[1], reverse=True)
            pbar.update(1)
            
        return word_scores[:n_words]