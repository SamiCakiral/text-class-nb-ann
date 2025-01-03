from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler


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
        """Extrait les features statistiques optimales"""
        features = []
        for text in texts:
            words = text.split()
            sentences = text.split('.')
            
            # Features optimisées
            feature_dict = {
                # Métriques fondamentales de taille
                'text_length': len(text), # Longueur du texte
                'word_count': len(words), # Nombre de mots
                
                # Patterns structurels
                'space_ratio': text.count(' ') / max(len(text), 1), # Ratio de mots par rapport à la longueur du texte
                'avg_word_length': np.mean([len(w) for w in words]), # Longueur moyenne des mots
                
                # Complexité syntaxique
                'long_sentence_ratio': len([s for s in sentences if len(s.split()) >= 15])/max(len(sentences), 1), # Ratio de phrases longues
                'medium_sentence_ratio': len([s for s in sentences if 5 < len(s.split()) < 15])/max(len(sentences), 1) # Ratio de phrases moyennes
            }
            
            features.append(list(feature_dict.values()))
        
        return np.array(features)
    
    def extract_enhanced_features(self, texts, max_features=1000):
        """Combine TF-IDF et les features statistiques avec normalisation appropriée"""
        # Extraction TF-IDF avec le nombre spécifié de features
        tfidf_features = self.extract_tfidf_features(texts, max_features)
        tfidf_dense = tfidf_features.toarray()
        
        # Normalisation L2 pour TF-IDF (meilleure pour le texte)
        tfidf_normalized = normalize(tfidf_dense, norm='l2')
        
        # Extraction et normalisation MinMax pour les features statistiques
        if not hasattr(self, 'stats_scaler'):
            self.stats_scaler = MinMaxScaler()
            statistical_features = self.extract_statistical_features(texts)
            statistical_normalized = self.stats_scaler.fit_transform(statistical_features)
        else:
            statistical_features = self.extract_statistical_features(texts)
            statistical_normalized = self.stats_scaler.transform(statistical_features)
        
        # Application des poids (70% TF-IDF, 30% stats)
        tfidf_weight, stat_weight = 0.7, 0.3
        
        # Combinaison des features
        combined_features = np.hstack((
            tfidf_normalized * tfidf_weight,
            statistical_normalized * stat_weight
        ))
        
        return combined_features
