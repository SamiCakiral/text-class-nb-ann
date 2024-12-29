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
        """Extrait les features statistiques optimisées selon l'analyse"""
        features = []
        for text in texts:
            words = text.split()
            sentences = text.split('.')
            
            # Calcul des features selon FEATURES_FINALES
            feature_dict = {
                # 1. Métriques fondamentales
                'text_length': len(text),
                'unique_words_ratio': len(set(words)) / max(len(words), 1),
                'std_word_length': np.std([len(w) for w in words]),
                
                # 2. Style d'écriture
                'avg_word_length': np.mean([len(w) for w in words]),
                'flesch_reading_ease': 206.835 - 1.015 * (len(words)/max(len(sentences), 1)) - 84.6 * (sum(sum(1 for c in w if c.lower() in 'aeiouy') for w in words)/max(len(words), 1)),
                
                # 3. Structure du texte
                'short_sentences_ratio': len([s for s in sentences if len(s.split()) <= 5]) / max(len(sentences), 1),
                'long_sentences_ratio': len([s for s in sentences if len(s.split()) >= 15]) / max(len(sentences), 1),
                
                # 4. Caractéristiques spécifiques aux news
                'starts_with_number': int(any(c.isdigit() for c in words[0])) if words else 0,
                'contains_date': int(any(w.replace("/", "").replace("-", "").isdigit() for w in words)),
                'contains_money': int(any(w.startswith('$') or w.endswith('€') for w in words)),
                
                # 5. Style narratif
                'third_person_pronouns': len([w for w in words if w.lower() in ['he', 'she', 'it', 'they', 'his', 'her', 'their']]) / max(len(words), 1),
                'quote_ratio': (text.count('"') + text.count("'")) / max(len(text), 1)
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
