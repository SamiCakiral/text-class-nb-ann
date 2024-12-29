import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import pickle
from tqdm import tqdm
from data_preprocessing import DataPreprocessor  # Adjust path according to your project

class TextFeatureAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.99)  # On veut 99% de la variance
        self.features_df = None
        
    def extract_text_features(self, texts):
        """Extract all possible statistical features from texts"""
        print("Extracting features...")
        features_dict = []
        
        for text in tqdm(texts):
            words = text.split()
            sentences = text.split('.')
            
            # Calculate all possible statistics
            stats = {
                # 1. Métriques de base
                'text_length': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': np.mean([len(w) for w in words]),
                'std_word_length': np.std([len(w) for w in words]),
                
                # 2. Ratios de caractères
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
                'lowercase_ratio': sum(1 for c in text if c.islower()) / max(len(text), 1),
                'digit_ratio': sum(c.isdigit() for c in text) / max(len(text), 1),
                'whitespace_ratio': sum(c.isspace() for c in text) / max(len(text), 1),
                'punctuation_ratio': len([c for c in text if c in '.,!?;:()[]{}"\'-']) / max(len(text), 1),
                
                # 3. Ponctuation spécifique
                'comma_ratio': text.count(',') / max(len(text), 1),
                'period_ratio': text.count('.') / max(len(text), 1),
                'exclamation_ratio': text.count('!') / max(len(text), 1),
                'question_ratio': text.count('?') / max(len(text), 1),
                'quote_ratio': (text.count('"') + text.count("'")) / max(len(text), 1),
                
                # 4. Ratios de mots
                'unique_words_ratio': len(set(words)) / max(len(words), 1),
                'short_words_ratio': len([w for w in words if len(w) <= 3]) / max(len(words), 1),
                'medium_words_ratio': len([w for w in words if 3 < len(w) < 7]) / max(len(words), 1),
                'long_words_ratio': len([w for w in words if len(w) >= 7]) / max(len(words), 1),
                
                # 5. Structure des phrases
                'short_sentences_ratio': len([s for s in sentences if len(s.split()) <= 5]) / max(len(sentences), 1),
                'medium_sentences_ratio': len([s for s in sentences if 5 < len(s.split()) < 15]) / max(len(sentences), 1),
                'long_sentences_ratio': len([s for s in sentences if len(s.split()) >= 15]) / max(len(sentences), 1),
                
                # 6. Densité lexicale
                'lexical_density': len(set(words)) / max(len(words), 1),
                'content_density': len([w for w in words if len(w) > 3]) / max(len(words), 1),
                
                # 7. Statistiques avancées
                'avg_sentence_length': len(words) / max(len(sentences), 1),
                'std_sentence_length': np.std([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0,
                'words_per_sentence': len(words) / max(len(sentences), 1),
                'chars_per_word': len(text.replace(" ", "")) / max(len(words), 1),
                
                # 8. Features spécifiques aux news
                'starts_with_number': int(any(c.isdigit() for c in words[0])) if words else 0,
                'contains_date': int(any(w.replace("/", "").replace("-", "").isdigit() for w in words)),
                'contains_money': int(any(w.startswith('$') or w.endswith('€') for w in words)),
                'contains_percent': int('%' in text),
                
                # 9. Ratios de position
                'first_person_pronouns': len([w for w in words if w.lower() in ['i', 'we', 'my', 'our', 'me', 'us']]) / max(len(words), 1),
                'third_person_pronouns': len([w for w in words if w.lower() in ['he', 'she', 'it', 'they', 'his', 'her', 'their']]) / max(len(words), 1),
                
                # 10. Complexité du texte
                'avg_syllables_per_word': np.mean([sum(1 for c in w if c.lower() in 'aeiouy') for w in words]) if words else 0,
                'flesch_reading_ease': 206.835 - 1.015 * (len(words)/max(len(sentences), 1)) - 84.6 * (sum(sum(1 for c in w if c.lower() in 'aeiouy') for w in words)/max(len(words), 1))
            }
            features_dict.append(stats)
            
        self.features_df = pd.DataFrame(features_dict)
        return self.features_df
    
    def analyze_features(self, n_components=0.99):
        """Analyze features with PCA and print detailed analysis"""
        if self.features_df is None:
            raise ValueError("No features extracted. Call extract_text_features first.")
            
        print("\nAnalyse détaillée des features...")
        
        # 1. Analyse statistique basique
        print("\n1. Statistiques descriptives:")
        stats = self.features_df.describe()
        print(stats)
        
        # 2. Calcul des corrélations
        print("\n2. Features hautement corrélées (>0.8):")
        corr_matrix = self.features_df.corr().abs()
        
        # Création d'un masque pour le triangle supérieur
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        high_corr_matrix = corr_matrix.where(mask)
        
        # Trouver les paires avec corrélation > 0.8
        high_corr = []
        for col in high_corr_matrix.columns:
            for idx in high_corr_matrix.index:
                if high_corr_matrix.loc[idx, col] > 0.8:
                    high_corr.append((idx, col, high_corr_matrix.loc[idx, col]))
        
        for f1, f2, corr in high_corr:
            if not np.isnan(corr):  # Éviter les NaN
                print(f"{f1} - {f2}: {corr:.3f}")

        # 3. Analyse de variance
        print("\n3. Variance des features:")
        variances = self.features_df.var().sort_values(ascending=False)
        print(variances)
        
        # 4. PCA avec analyse détaillée
        X_scaled = self.scaler.fit_transform(self.features_df)
        self.pca.fit(X_scaled)
        
        # Créer un dictionnaire de résultats PCA
        pca_results = {
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.pca.explained_variance_ratio_),
            'components': pd.DataFrame(
                self.pca.components_.T,
                columns=[f'PC{i+1}' for i in range(len(self.pca.components_))],
                index=self.features_df.columns
            )
        }
        
        print("\n4. Analyse PCA:")
        cumulative_var = np.cumsum(self.pca.explained_variance_ratio_)
        components_needed = len(cumulative_var[cumulative_var <= 0.99])
        
        print(f"\nNombre de composantes nécessaires pour 99% de variance: {components_needed}")
        
        # 5. Contribution des features
        feature_importance = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(self.pca.components_))],
            index=self.features_df.columns
        )
        
        print("\n5. Features les plus importantes par composante:")
        for i in range(min(3, len(self.pca.components_))):
            pc = f'PC{i+1}'
            var_exp = self.pca.explained_variance_ratio_[i] * 100
            print(f"\n{pc} (explique {var_exp:.1f}% de la variance):")
            
            # Trier les features par importance absolue
            importance = abs(feature_importance[pc]).sort_values(ascending=False)
            cumsum_importance = importance.cumsum() / importance.sum()
            
            # Afficher les features qui contribuent à 80% de l'importance
            important_features = importance[cumsum_importance <= 0.8]
            for feat, imp in important_features.items():
                print(f"  - {feat}: {imp:.3f}")
        
        # 6. Recommandations finales
        print("\n6. Recommandations des features à conserver:")
        important_features = set()
        
        # Ajouter les features avec haute variance
        top_variance = variances[variances > variances.median()].index
        important_features.update(top_variance)
        
        # Ajouter les features importantes des premières composantes PCA
        for i in range(min(3, len(self.pca.components_))):
            pc = f'PC{i+1}'
            importance = abs(feature_importance[pc])
            top_features = importance[importance > importance.median()].index
            important_features.update(top_features)
        
        # Retirer les features très corrélées
        for f1, f2, _ in high_corr:
            if f1 in important_features and f2 in important_features:
                # Garder celle avec la plus grande variance
                if variances[f1] < variances[f2]:
                    important_features.remove(f1)
                else:
                    important_features.remove(f2)
        
        print("\nFeatures recommandées:")
        for feature in sorted(important_features):
            print(f"- {feature}")
        
        # À la fin de la méthode, retourner à la fois les features importantes et les résultats PCA
        return {
            'important_features': important_features,
            'pca_results': pca_results
        }
    
    def plot_analysis(self, results, output_dir='results/analysis'):
        """Generate all visualizations"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        pca_results = results['pca_results']
        
        # 1. Variance expliquée cumulative
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(pca_results['explained_variance_ratio']) + 1),
                pca_results['cumulative_variance_ratio'], 'bo-')
        plt.xlabel('Nombre de composantes')
        plt.ylabel('Variance expliquée cumulative')
        plt.title('Variance expliquée par composante')
        plt.grid(True)
        plt.savefig(f'{output_dir}/variance_explained.png')
        plt.close()
        
        # 2. Heatmap des composantes principales
        plt.figure(figsize=(20, 12))
        sns.heatmap(pca_results['components'].iloc[:5], cmap='coolwarm', center=0,
                    xticklabels=True, yticklabels=True)
        plt.title('Contribution des features aux 5 premières composantes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance_heatmap.png')
        plt.close()
        
        # 3. Matrice de corrélation avec clustering
        plt.figure(figsize=(20, 16))
        # 1. Nettoyage approfondi des données
        features_clean = self.features_df.copy()
        
        # Remplacer les valeurs infinies par NaN
        features_clean = features_clean.replace([np.inf, -np.inf], np.nan)
        
        # Remplacer les NaN par 0 (ou la médiane si préféré)
        features_clean = features_clean.fillna(0)
        
        # Supprimer les colonnes avec variance nulle
        variance = features_clean.var()
        features_clean = features_clean.loc[:, variance > 0]
        
        # Calculer la corrélation sur les données nettoyées
        corr = features_clean.corr()
        
        # S'assurer que la matrice est propre
        corr = np.nan_to_num(corr, nan=0.0)
        
        # Créer le masque pour le triangle supérieur
        mask = np.triu(np.ones_like(corr), k=1)
        
        # Plot avec les données nettoyées
        g = sns.clustermap(
            corr,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            mask=mask,
            figsize=(20, 16),
            dendrogram_ratio=(.1, .2),
            cbar_pos=(0.02, .32, .03, .2),
            xticklabels=True,
            yticklabels=True
        )
        
        # Rotation des labels
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
        
        # Sauvegarder
        plt.savefig(f'{output_dir}/feature_correlation_clustered.png', 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()
        
        # 4. Distribution des features par catégorie
        categories = {
            'Métriques de base': ['text_length', 'word_count', 'sentence_count', 'avg_word_length', 'std_word_length'],
            'Ratios de caractères': ['uppercase_ratio', 'lowercase_ratio', 'digit_ratio', 'whitespace_ratio', 'punctuation_ratio'],
            'Ponctuation': ['comma_ratio', 'period_ratio', 'exclamation_ratio', 'question_ratio', 'quote_ratio'],
            'Structure': ['short_sentences_ratio', 'medium_sentences_ratio', 'long_sentences_ratio'],
            'Complexité': ['lexical_density', 'content_density', 'flesch_reading_ease', 'avg_syllables_per_word']
        }
        
        for cat_name, features in categories.items():
            # Vérifier que toutes les features existent dans le DataFrame
            valid_features = [f for f in features if f in self.features_df.columns]
            if valid_features:  # Ne créer le plot que s'il y a des features valides
                plt.figure(figsize=(15, 6))
                self.features_df[valid_features].boxplot()
                plt.title(f'Distribution des features - {cat_name}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/distribution_{cat_name.lower().replace(" ", "_")}.png')
                plt.close()
    
    def get_important_features(self, results, n_top=5):
        """Identify the most important features"""
        pca_results = results['pca_results']  # Extraire les résultats PCA du dictionnaire
        important_features = {}
        
        for i in range(3):  # First 3 components
            component_importance = abs(pca_results['components'].iloc[:, i])
            top_features = component_importance.nlargest(n_top)
            important_features[f'PC{i+1}'] = list(zip(top_features.index, top_features.values))
        
        return important_features
    
    def save_results(self, important_features, output_dir='results/analysis'):
        """Save results to a file"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save important features
        with open(f'{output_dir}/important_features.txt', 'w') as f:
            f.write("Most important features by principal component:\n\n")
            for pc, features in important_features.items():
                f.write(f"\n{pc}:\n")
                for feature, importance in features:
                    f.write(f"  - {feature}: {importance:.3f}\n")
        
        # Save descriptive statistics
        stats_description = self.features_df.describe()
        stats_description.to_csv(f'{output_dir}/feature_statistics.csv')

def main():
    """Main entry point for analysis"""
    
    
    # Load data
    print("Loading data...")
    preprocessor = DataPreprocessor()
    train_df = preprocessor.load_data('data/raw/train.csv')
    train_df = preprocessor.preprocess(train_df, is_train=True)
    
    # Check available columns
    print("\nAvailable columns:", train_df.columns.tolist())
    
    # Use the 'Cleaned_Text' column
    text_column = 'Cleaned_Text'  # Exact column name in your DataFrame
    
    # Feature analysis
    analyzer = TextFeatureAnalyzer()
    features_df = analyzer.extract_text_features(train_df[text_column])
    results = analyzer.analyze_features()  # Retourne un dict avec 'important_features' et 'pca_results'
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_analysis(results)
    
    # Identify important features
    important_features = analyzer.get_important_features(results)
    
    # Save results
    print("\nSaving results...")
    analyzer.save_results(important_features)
    
    print("\nAnalysis complete! Results are in the 'results/analysis' folder")

if __name__ == "__main__":
    main() 