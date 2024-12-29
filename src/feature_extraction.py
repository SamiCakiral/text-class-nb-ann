from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np


class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizers = {}
        self.top_features = {}

    def create_ngram_vectors(self, texts, n):
        """Crée les vecteurs n-gram"""
        vectorizer = CountVectorizer(ngram_range=(n, n)) # Création du vectorizer
        vectors = vectorizer.fit_transform(texts) # Création des vecteurs
        self.count_vectorizers[n] = vectorizer # Sauvegarde du vectorizer
        
        return vectors

    def extract_tfidf_features(self, texts, max_features=1000):
        """Extrait les caractéristiques TF-IDF"""
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features) # Création du vectorizer
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts) # Création des vecteurs
        
        return tfidf_matrix

    def get_top_words(self, n_words=15):
        """Obtient les mots avec les scores TF-IDF les plus élevés"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer n'a pas encore été entraîné")
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out() # Récupération des noms des caractéristiques
        scores = self.tfidf_vectorizer.idf_ # Récupération des scores TF-IDF
        word_scores = list(zip(feature_names, scores)) # Création de la liste des mots avec leurs scores
        word_scores.sort(key=lambda x: x[1], reverse=True) # Tri des mots par score décroissant
        
        return word_scores[:n_words] # Retour des n_words mots avec les scores les plus élevés

class EnhancedFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.tfidf_scaler = StandardScaler()
        self.feature_weights = None
        
    def compute_feature_weights(self, tfidf_features, statistical_features):
        """Calcule les poids des features basés sur leur complexité"""
        # Calcul de la complexité des features TF-IDF (basé sur la variance)
        tfidf_complexity = np.var(tfidf_features, axis=0)
        stat_complexity = np.var(statistical_features, axis=0)
        
        # Normalisation des complexités
        total_tfidf_complexity = np.sum(tfidf_complexity)
        total_stat_complexity = np.sum(stat_complexity)
        
        # Calcul des poids (plus la complexité est élevée, plus le poids est important)
        tfidf_weight = total_tfidf_complexity / (total_tfidf_complexity + total_stat_complexity)
        stat_weight = total_stat_complexity / (total_tfidf_complexity + total_stat_complexity)
        
        return tfidf_weight, stat_weight
    
    def extract_statistical_features(self, texts):
        """Extrait exactement les 6 caractéristiques statistiques spécifiées"""
        features = []
        
        for text in texts:
            # 1-2. Text length and word count
            text_length = len(text)
            words = text.split()
            word_count = len(words)
            
            # 3-4. Space ratio and average word length
            space_ratio = text.count(' ') / text_length if text_length > 0 else 0
            avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
            
            # 5-6. Long/medium sentence ratios
            sentences = text.split('.')
            sentence_lengths = [len(s.strip().split()) for s in sentences if s.strip()]
            total_sentences = len(sentences) if sentences else 1
            
            long_sentences = sum(1 for length in sentence_lengths if length > 20)
            medium_sentences = sum(1 for length in sentence_lengths if 10 <= length <= 20)
            
            long_ratio = long_sentences / total_sentences
            medium_ratio = medium_sentences / total_sentences
            
            features.append([
                text_length,        # 1
                word_count,         # 2
                space_ratio,        # 3
                avg_word_length,    # 4
                long_ratio,         # 5
                medium_ratio        # 6
            ])
            
        return np.array(features)
    
    def extract_enhanced_features(self, texts, max_features=1000):
        """Combine TF-IDF (5, 10, ou 15 dimensions) et les 6 features statistiques"""
        # Extraction TF-IDF avec le nombre spécifié de features
        tfidf_features = self.extract_tfidf_features(texts, max_features)
        tfidf_dense = tfidf_features.toarray()
        
        # Normalisation des features TF-IDF
        tfidf_normalized = self.tfidf_scaler.fit_transform(tfidf_dense)
        
        # Extraction et normalisation des 6 features statistiques
        statistical_features = self.extract_statistical_features(texts)
        statistical_normalized = self.scaler.fit_transform(statistical_features)
        
        # Calcul des poids des features
        if self.feature_weights is None:
            tfidf_weight, stat_weight = self.compute_feature_weights(tfidf_normalized, statistical_normalized)
            self.feature_weights = (tfidf_weight, stat_weight)
        
        # Application des poids
        weighted_tfidf = tfidf_normalized * self.feature_weights[0]
        weighted_stats = statistical_normalized * self.feature_weights[1]
        
        # Combinaison des features
        combined_features = np.hstack((weighted_tfidf, weighted_stats))
        
        return combined_features
