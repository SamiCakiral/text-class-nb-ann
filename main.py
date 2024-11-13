import os
from pathlib import Path
import argparse
from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor
from src.naive_bayes import CustomNaiveBayes
from src.neural_network import TextClassifierANN
from src.utils import plot_confusion_matrix, save_metrics
from tqdm import tqdm

def setup_argparse():
    parser = argparse.ArgumentParser(description='NLP Classification Project')
    parser.add_argument('--train_path', type=str, required=True, help='Chemin vers le fichier d\'entraînement')
    parser.add_argument('--test_path', type=str, required=True, help='Chemin vers le fichier de test')
    parser.add_argument('--output_dir', type=str, default='results', help='Dossier de sortie pour les résultats')
    return parser.parse_args()

def ensure_directories(base_dir):
    """Crée les dossiers nécessaires s'ils n'existent pas"""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(base_dir, 'metrics')).mkdir(exist_ok=True)
    Path(os.path.join(base_dir, 'models')).mkdir(exist_ok=True)

def run_naive_bayes_experiments(train_texts, test_texts, train_labels, test_labels):
    """Exécute les expériences avec Naive Bayes"""
    feature_extractor = FeatureExtractor()
    nb_classifier = CustomNaiveBayes()
    results = {}

    # Entraînement et évaluation pour chaque n-gram
    for n in tqdm(range(1, 4), desc="Progression globale n-grams"):
        print(f"\nTraitement des {n}-grams...")
        
        # Création des vecteurs avec barre de progression
        print("Création des vecteurs d'entraînement...")
        train_vectors = feature_extractor.create_ngram_vectors(train_texts, n)
        
        print("Création des vecteurs de test...")
        test_vectors = feature_extractor.count_vectorizers[n].transform(test_texts)
        
        # Entraînement avec barre de progression
        print(f"Entraînement du modèle {n}-gram...")
        with tqdm(total=100, desc=f"Entraînement {n}-gram") as pbar:
            nb_classifier.train_model(train_vectors, train_labels, n, pbar)
            
        # Évaluation avec barre de progression
        print(f"Évaluation du modèle {n}-gram...")
        with tqdm(total=len(test_labels), desc=f"Évaluation {n}-gram") as pbar:
            results[f'{n}-gram'] = nb_classifier.evaluate(test_vectors, test_labels, n, pbar)
        
        print(f"Accuracy pour {n}-gram: {results[f'{n}-gram']['accuracy']:.4f}")
        print(f"Recall pour {n}-gram: {results[f'{n}-gram']['recall']:.4f}")
        
    return results

def run_ann_experiments(train_texts, test_texts, train_labels, test_labels):
    """Exécute les expériences avec le réseau de neurones"""
    feature_extractor = FeatureExtractor()
    results = {}

    # Extraction des caractéristiques TF-IDF avec barre de progression
    print("\nCalcul des TF-IDF...")
    with tqdm(total=2, desc="Extraction TF-IDF") as pbar:
        train_tfidf = feature_extractor.extract_tfidf_features(train_texts)
        pbar.update(1)
        test_tfidf = feature_extractor.tfidf_vectorizer.transform(test_texts)
        pbar.update(1)

    # Obtention des top mots pour différentes tailles de sacs de mots
    for n_words in [5, 10, 15]:
        print(f"\nTraitement ANN avec {n_words} mots...")
        
        # Extraction des top mots
        top_words = feature_extractor.get_top_words(n_words)
        
        # Création et entraînement du modèle
        ann_classifier = TextClassifierANN(hidden_layer_size=100)
        
        # Entraînement avec affichage du temps
        print(f"Entraînement du modèle ANN ({n_words} mots)...")
        with tqdm(total=1, desc="Entraînement") as pbar:
            ann_classifier.train(train_tfidf, train_labels, progress_bar=pbar)
        
        # Évaluation avec barre de progression
        print("Évaluation du modèle...")
        with tqdm(total=1, desc="Évaluation") as pbar:
            results[f'ann_{n_words}_words'] = ann_classifier.evaluate(test_tfidf, test_labels, pbar)
        
        print(f"Accuracy pour {n_words} mots: {results[f'ann_{n_words}_words']['accuracy']:.4f}")
        print(f"Recall pour {n_words} mots: {results[f'ann_{n_words}_words']['recall']:.4f}")
    
    return results

def main():
    # Configuration
    args = setup_argparse()
    ensure_directories(args.output_dir)
    
    print("Démarrage du processus de classification...")
    
    # Préparation des données
    preprocessor = DataPreprocessor()
    
    print("Chargement et prétraitement des données d'entraînement...")
    train_df = preprocessor.load_data(args.train_path)
    train_df = preprocessor.preprocess(train_df)
    
    print("Chargement et prétraitement des données de test...")
    test_df = preprocessor.load_data(args.test_path)
    test_df = preprocessor.preprocess(test_df)
    
    train_texts = train_df['Cleaned_Text']
    test_texts = test_df['Cleaned_Text']
    train_labels = train_df['Class']
    test_labels = test_df['Class']
    
    # Expériences Naive Bayes
    print("\nDémarrage des expériences Naive Bayes...")
    nb_results = run_naive_bayes_experiments(train_texts, test_texts, train_labels, test_labels)
    
    # Expériences ANN
    print("\nDémarrage des expériences ANN...")
    ann_results = run_ann_experiments(train_texts, test_texts, train_labels, test_labels)
    
    # Sauvegarde des résultats
    metrics_path = os.path.join(args.output_dir, 'metrics', 'results.txt')
    save_metrics(nb_results, 'Naive Bayes', metrics_path)
    save_metrics(ann_results, 'ANN', metrics_path)
    
    print(f"\nTraitement terminé. Résultats sauvegardés dans {metrics_path}")

if __name__ == "__main__":
    main()

    