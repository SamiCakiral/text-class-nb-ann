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
        vectorizer = CountVectorizer(ngram_range=(n, n))
        vectors = vectorizer.fit_transform(texts)
        self.count_vectorizers[n] = vectorizer
        return vectors

    def extract_tfidf_features(self, texts, max_features=1000):
        """Extrait les caractéristiques TF-IDF"""
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        return tfidf_matrix

    def get_top_words(self, n_words=15):
        """Obtient les mots avec les scores TF-IDF les plus élevés"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer n'a pas encore été entraîné")
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        scores = self.tfidf_vectorizer.idf_
        word_scores = list(zip(feature_names, scores))
        
        print("Tri des mots par score TF-IDF...")
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:n_words]