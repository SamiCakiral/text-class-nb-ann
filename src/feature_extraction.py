from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
import numpy as np
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizers = {}  # Un vectorizer par classe
        self.top_words_per_class = {}  # Stockage des mots importants par classe
        self.count_vectorizers = {}  # Pour Naive Bayes
        
    def is_noun_or_adj(self, word, tag):
        """Vérifie si le mot est un nom ou un adjectif"""
        return (tag.startswith('NN') or tag.startswith('JJ'))
    
    def create_ngram_vectors(self, texts, n):
        """Crée les vecteurs n-gram pour Naive Bayes"""
        vectorizer = CountVectorizer(ngram_range=(n, n))
        vectors = vectorizer.fit_transform(texts)
        self.count_vectorizers[n] = vectorizer
        return vectors
    
    def extract_tfidf_features_by_class(self, texts, labels, n_words):
        """
        Extrait les n mots les plus importants par classe selon TF-IDF.
        Ne garde que les noms et adjectifs.
        """
        unique_classes = np.unique(labels)
        all_important_words = set()
        
        # Pour chaque classe
        for class_label in unique_classes:
            # Sélectionner les textes de cette classe
            class_mask = labels == class_label
            class_texts = [texts[i] for i in range(len(texts)) if class_mask[i]]
            
            # Créer et entraîner le vectorizer pour cette classe
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(class_texts)
            
            # Récupérer les mots et leurs scores
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            
            # Filtrer pour ne garder que les noms et adjectifs
            word_pos_pairs = pos_tag(feature_names)
            valid_words = [(word, score) for (word, tag), score 
                          in zip(word_pos_pairs, tfidf_scores) 
                          if self.is_noun_or_adj(word, tag)]
            
            # Trier par score TF-IDF et prendre les n_words premiers
            valid_words.sort(key=lambda x: x[1], reverse=True)
            top_words = [word for word, _ in valid_words[:n_words]]
            
            # Sauvegarder pour cette classe
            self.top_words_per_class[class_label] = top_words
            all_important_words.update(top_words)
            
        return list(all_important_words)
    
    def create_bow_features(self, texts, n_words):
        """
        Crée la matrice de features basée sur les bags of words.
        """
        # Créer un vocabulaire à partir de tous les mots importants
        all_words = []
        for words in self.top_words_per_class.values():
            all_words.extend(words)
        vocabulary = list(set(all_words))
        
        # Créer la matrice de features
        features = np.zeros((len(texts), len(vocabulary)))
        
        for i, text in enumerate(texts):
            words = word_tokenize(text.lower())
            for j, vocab_word in enumerate(vocabulary):
                features[i, j] = words.count(vocab_word)
                
        return features
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
        tfidf_features = super().extract_tfidf_features_by_class(texts, np.zeros(len(texts)), max_features)
        tfidf_matrix = TfidfVectorizer(vocabulary=tfidf_features).fit_transform(texts)
        tfidf_dense = tfidf_matrix.toarray()
        
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

