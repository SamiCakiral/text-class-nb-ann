from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


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
