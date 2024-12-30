import numpy as np
import pandas as pd
from collections import Counter
import re
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

class NewsFeatureAnalyzer:
    def __init__(self):
        self.features_df = None
        self.class_stats = {}
        
    def calculate_pmi(self, text, window_size=5):
        """Calcule la PMI moyenne du texte"""
        words = text.split()
        if len(words) < window_size:
            return 0
        
        # Calcul des fréquences individuelles
        word_freq = Counter(words)
        total_words = len(words)
        
        # Calcul des co-occurrences
        cooccurrence = Counter()
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            for j in range(len(window)):
                for k in range(j + 1, len(window)):
                    cooccurrence[(window[j], window[k])] += 1
        
        # Calcul PMI
        pmi_scores = []
        for (word1, word2), cooc_count in cooccurrence.items():
            p_xy = cooc_count / total_words
            p_x = word_freq[word1] / total_words
            p_y = word_freq[word2] / total_words
            
            if p_xy > 0 and p_x > 0 and p_y > 0:
                pmi = np.log2(p_xy / (p_x * p_y))
                pmi_scores.append(pmi)
        
        return np.mean(pmi_scores) if pmi_scores else 0
    
    def extract_statistical_features(self, text):
        """Extrait toutes les caractéristiques statistiques d'un texte"""
        # Préparation du texte
        if pd.isna(text):  # Gestion des valeurs NaN
            text = ""
        text = str(text)  # Conversion en string pour être sûr
        
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        try:
            # 1. Statistiques de base
            basic_stats = {
                'char_count': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'paragraph_count': len(paragraphs),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'std_word_length': np.std([len(w) for w in words]) if words else 0,
            }
            
            # 2. Statistiques de phrases
            sentence_stats = {
                'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
                'std_sentence_length': np.std([len(s.split()) for s in sentences]) if sentences else 0,
                'min_sentence_length': min([len(s.split()) for s in sentences]) if sentences else 0,
                'max_sentence_length': max([len(s.split()) for s in sentences]) if sentences else 0,
            }
            
            # 3. Ratios et densités
            density_stats = {
                'unique_words_ratio': len(set(words)) / len(words) if words else 0,
                'chars_per_word': len(text.replace(" ", "")) / len(words) if words else 0,
                'words_per_sentence': len(words) / len(sentences) if sentences else 0,
                'sentences_per_paragraph': len(sentences) / len(paragraphs) if paragraphs else 0,
            }
            
            # 4. Distribution des mots
            word_distribution = {
                'short_words_ratio': len([w for w in words if len(w) <= 3]) / len(words) if words else 0,
                'medium_words_ratio': len([w for w in words if 3 < len(w) < 7]) / len(words) if words else 0,
                'long_words_ratio': len([w for w in words if len(w) >= 7]) / len(words) if words else 0,
            }
            
            # 5. Caractéristiques structurelles
            structural_stats = {
                'starts_with_capital': int(text[0].isupper()) if text else 0,
                'ends_with_punct': int(text[-1] in '.!?') if text else 0,
                'avg_paragraph_length': np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0,
                'std_paragraph_length': np.std([len(p.split()) for p in paragraphs]) if paragraphs else 0,
            }
            
            # 6. Statistiques de ponctuation
            punctuation_stats = {
                'comma_ratio': text.count(',') / len(text) if text else 0,
                'period_ratio': text.count('.') / len(text) if text else 0,
                'question_ratio': text.count('?') / len(text) if text else 0,
                'exclamation_ratio': text.count('!') / len(text) if text else 0,
                'quote_ratio': (text.count('"') + text.count("'")) / len(text) if text else 0,
            }
            
            # 7. Statistiques avancées
            advanced_stats = {
                'entropy': stats.entropy(pd.Series(words).value_counts()) if words else 0,
                'unique_char_ratio': len(set(text)) / len(text) if text else 0,
                'digit_ratio': sum(c.isdigit() for c in text) / len(text) if text else 0,
                'uppercase_ratio': sum(c.isupper() for c in text) / len(text) if text else 0,
            }
            
            # 8. Statistiques PMI
            pmi_stats = {
                'avg_pmi': self.calculate_pmi(text),
                'avg_pmi_short': self.calculate_pmi(text, window_size=3),
                'avg_pmi_long': self.calculate_pmi(text, window_size=7),
            }
            
            # Combiner toutes les statistiques
            return {**basic_stats, **sentence_stats, **density_stats, 
                    **word_distribution, **structural_stats, 
                    **punctuation_stats, **advanced_stats, **pmi_stats}
            
        except Exception as e:
            print(f"Erreur lors du traitement du texte: {str(e)}")
            return {k: 0 for k in ['char_count', 'word_count', 'sentence_count',
                                  'avg_pmi', 'avg_pmi_short', 'avg_pmi_long']}
    
    def analyze_dataset(self, texts, labels):
        """Analyse complète du dataset"""
        print("Extraction des features...")
        features = []
        for text in tqdm(texts):
            features.append(self.extract_statistical_features(text))
        
        self.features_df = pd.DataFrame(features)
        self.labels = labels
        
        # Analyse par classe
        print("\nAnalyse par classe...")
        for class_id in set(labels):
            class_mask = labels == class_id
            class_features = self.features_df[class_mask]
            self.class_stats[class_id] = {
                'mean': class_features.mean(),
                'std': class_features.std(),
                'size': len(class_features)
            }
    
    def get_discriminative_features(self, min_importance=0.05):
        """Identifie les features les plus discriminantes"""
        importance = mutual_info_classif(self.features_df, self.labels)
        features_importance = list(zip(self.features_df.columns, importance))
        features_importance.sort(key=lambda x: x[1], reverse=True)
        
        return [f for f, imp in features_importance if imp > min_importance]
    
    def plot_analysis(self, output_dir='results/feature_analysis'):
        """Génère des visualisations"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Distribution des features principales
        plt.figure(figsize=(15, 10))
        self.features_df.boxplot()
        plt.xticks(rotation=45)
        plt.title('Distribution des features')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/features_distribution.png')
        plt.close()
        
        # 2. Matrice de corrélation
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.features_df.corr(), cmap='coolwarm', center=0)
        plt.title('Corrélation entre features')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/features_correlation.png')
        plt.close()
        
        # 3. Importance des features
        importance = mutual_info_classif(self.features_df, self.labels)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance)), sorted(importance, reverse=True))
        plt.title('Importance des features')
        plt.xlabel('Features')
        plt.ylabel('Importance (mutual information)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/features_importance.png')
        plt.close()

def main():
    """Main entry point for analysis"""
    # Chargement des données
    from data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data('data/raw/train.csv')
    
    print("\nColonnes disponibles:", data.columns.tolist())
    
    # Analyse des features
    analyzer = NewsFeatureAnalyzer()
    # Utiliser 'Description' et 'Class' qui sont les vrais noms des colonnes
    analyzer.analyze_dataset(data['Description'].values, data['Class'].values)
    
    # Identification des features discriminantes
    important_features = analyzer.get_discriminative_features()
    print("\nFeatures les plus discriminantes:")
    for feature in important_features:
        print(f"- {feature}")
    
    # Génération des visualisations
    analyzer.plot_analysis()

if __name__ == "__main__":
    main()
