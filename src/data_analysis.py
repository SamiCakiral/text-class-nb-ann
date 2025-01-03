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
        self.pca = PCA()
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
                # Basic
                'text_length': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                
                # Averages
                'avg_word_length': np.mean([len(w) for w in words]),
                'avg_sentence_length': len(words) / max(len(sentences), 1),
                'std_word_length': np.std([len(w) for w in words]),
                
                # Character ratios
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
                'punctuation_ratio': len([c for c in text if c in '.,!?;:']) / max(len(text), 1),
                'space_ratio': text.count(' ') / max(len(text), 1),
                'digit_ratio': sum(c.isdigit() for c in text) / max(len(text), 1),
                'letter_ratio': sum(c.isalpha() for c in text) / max(len(text), 1),
                
                # Word ratios
                'unique_words_ratio': len(set(words)) / max(len(words), 1),
                'short_words_ratio': len([w for w in words if len(w) <= 3]) / max(len(words), 1),
                'medium_words_ratio': len([w for w in words if 3 < len(w) < 7]) / max(len(words), 1),
                'long_words_ratio': len([w for w in words if len(w) >= 7]) / max(len(words), 1),
                
                # Sentence ratios
                'short_sentences_ratio': len([s for s in sentences if len(s.split()) <= 5]) / max(len(sentences), 1),
                'medium_sentences_ratio': len([s for s in sentences if 5 < len(s.split()) < 15]) / max(len(sentences), 1),
                'long_sentences_ratio': len([s for s in sentences if len(s.split()) >= 15]) / max(len(sentences), 1),
                
                # Distribution
                'words_per_sentence': len(words) / max(len(sentences), 1),
                'chars_per_word': len(text) / max(len(words), 1),
            }
            features_dict.append(stats)
            
        self.features_df = pd.DataFrame(features_dict)
        return self.features_df
    
    def analyze_features(self, n_components=0.95):
        """Analyze features with PCA"""
        if self.features_df is None:
            raise ValueError("No features extracted. Call extract_text_features first.")
            
        print("\nPCA feature analysis...")
        X_scaled = self.scaler.fit_transform(self.features_df)
        self.pca.fit(X_scaled)
        
        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.pca.explained_variance_ratio_),
            'components': pd.DataFrame(
                self.pca.components_,
                columns=self.features_df.columns
            )
        }
    
    def plot_analysis(self, pca_results, output_dir='results/analysis'):
        """Generate all visualizations"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pca_results['explained_variance_ratio']) + 1),
                pca_results['cumulative_variance_ratio'], 'bo-')
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title('Variance Explained by Component')
        plt.grid(True)
        plt.savefig(f'{output_dir}/variance_explained.png')
        plt.close()
        
        # 2. Principal component heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(pca_results['components'].iloc[:3], cmap='coolwarm', center=0)
        plt.title('Feature Contribution to First 3 Components')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance_heatmap.png')
        plt.close()
        
        # 3. Feature correlation
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.features_df.corr(), cmap='coolwarm', center=0)
        plt.title('Feature Correlation')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_correlation.png')
        plt.close()
        
        # 4. Distribution des features avec normalisation et gestion des outliers
        plt.figure(figsize=(15, 6))
        
        # Normaliser les données
        features_normalized = pd.DataFrame(
            self.scaler.fit_transform(self.features_df),
            columns=self.features_df.columns
        )
        
        # Clipper les valeurs extrêmes à ±5 écarts-types
        features_normalized = features_normalized.clip(-5, 5)
        
        # Créer le boxplot
        features_normalized.boxplot(figsize=(15, 6))
        plt.xticks(rotation=90)
        plt.title('Feature Distribution (Normalized with Clip at ±5σ)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_distribution.png')
        plt.close()
    
    def get_important_features(self, pca_results, n_top=5):
        """Identify the most important features"""
        important_features = {}
        
        for i in range(3):  # First 3 components
            component_importance = abs(pca_results['components'].iloc[i])
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
    pca_results = analyzer.analyze_features()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_analysis(pca_results)
    
    # Identify important features
    important_features = analyzer.get_important_features(pca_results)
    
    # Save results
    print("\nSaving results...")
    analyzer.save_results(important_features)
    
    print("\nAnalysis complete! Results are in the 'results/analysis' folder")
if __name__ == "__main__":
    main() 