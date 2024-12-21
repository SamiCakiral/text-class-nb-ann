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
    """Configuration des arguments en ligne de commande
    Permet de spécifier les chemins des fichiers d'entrée et de sortie"""
    parser = argparse.ArgumentParser(description='NLP Classification Project')
    parser.add_argument('--train_path', type=str, required=True, help='Chemin vers le fichier d\'entraînement')
    parser.add_argument('--test_path', type=str, required=True, help='Chemin vers le fichier de test')
    parser.add_argument('--output_dir', type=str, default='results', help='Dossier de sortie pour les résultats')
    return parser.parse_args()

def ensure_directories(base_dir):
    """Crée la structure des dossiers pour sauvegarder les résultats
    - base_dir/metrics : pour les métriques de performance
    - base_dir/models : pour sauvegarder les modèles"""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(base_dir, 'metrics')).mkdir(exist_ok=True)
    Path(os.path.join(base_dir, 'models')).mkdir(exist_ok=True)

def run_naive_bayes_experiments(train_texts, test_texts, train_labels, test_labels):
    """Exécute les expériences avec l'algorithme Naive Bayes avec différentes méthodes
    
    1. Naive Bayes standard
    2. Naive Bayes avec Laplace smoothing
    3. Naive Bayes avec Good-Turing
    4. (Bonus) Naive Bayes avec interpolation
    """
    feature_extractor = FeatureExtractor()
    results = {}
    X_dict = {}  # Pour l'interpolation

    # Test différentes tailles de n-grams (1, 2, 3)
    for n in tqdm(range(1, 4), desc="Progression globale n-grams"):
        print(f"\nTraitement des {n}-grams...")
        
        # Création des vecteurs de caractéristiques
        train_vectors = feature_extractor.create_ngram_vectors(train_texts, n)
        test_vectors = feature_extractor.count_vectorizers[n].transform(test_texts)
        X_dict[n] = test_vectors  # Stockage pour l'interpolation
        
        # 1. Test Naive Bayes standard (sans smoothing)
        print(f"\nTest Naive Bayes standard pour {n}-gram...")
        nb_standard = CustomNaiveBayes(alpha=0)
        with tqdm(total=100, desc=f"Entraînement standard {n}-gram") as pbar:
            nb_standard.train_model(train_vectors, train_labels, n, pbar)
        with tqdm(total=len(test_labels), desc=f"Évaluation standard {n}-gram") as pbar:
            results[f'{n}-gram_standard'] = nb_standard.evaluate(test_vectors, test_labels, n, pbar)
        
        # 2. Test avec Laplace smoothing
        print(f"\nTest Naive Bayes avec Laplace smoothing pour {n}-gram...")
        nb_laplace = CustomNaiveBayes(alpha=1.0)
        with tqdm(total=100, desc=f"Entraînement Laplace {n}-gram") as pbar:
            nb_laplace.train_model(train_vectors, train_labels, n, pbar)
        with tqdm(total=len(test_labels), desc=f"Évaluation Laplace {n}-gram") as pbar:
            results[f'{n}-gram_laplace'] = nb_laplace.evaluate(test_vectors, test_labels, n, pbar)
        
        # 3. Test avec Good-Turing (utilise le même modèle que Laplace mais avec Good-Turing pour les mots inconnus)
        print(f"\nTest Naive Bayes avec Good-Turing pour {n}-gram...")
        nb_goodturing = CustomNaiveBayes(alpha=1.0)
        with tqdm(total=100, desc=f"Entraînement Good-Turing {n}-gram") as pbar:
            nb_goodturing.train_model(train_vectors, train_labels, n, pbar)
        with tqdm(total=len(test_labels), desc=f"Évaluation Good-Turing {n}-gram") as pbar:
            results[f'{n}-gram_goodturing'] = nb_goodturing.evaluate(test_vectors, test_labels, n, pbar)
        
        # Affichage des résultats pour ce n-gram
        for method in ['standard', 'laplace', 'goodturing']:
            print(f"\nRésultats pour {n}-gram avec {method}:")
            print(f"Accuracy: {results[f'{n}-gram_{method}']['accuracy']:.4f}")
            print(f"Recall: {results[f'{n}-gram_{method}']['recall']:.4f}")
    
    # 4. Bonus: Test avec interpolation (combine les résultats des différents n-grams)
    print("\nTest Naive Bayes avec interpolation...")
    nb_interpolation = CustomNaiveBayes(alpha=1.0)
    # Entraînement sur tous les n-grams
    for n in range(1, 4):
        train_vectors = feature_extractor.create_ngram_vectors(train_texts, n)
        with tqdm(total=100, desc=f"Entraînement interpolation {n}-gram") as pbar:
            nb_interpolation.train_model(train_vectors, train_labels, n, pbar)
    
    # Évaluation avec interpolation
    with tqdm(total=len(test_labels), desc="Évaluation interpolation") as pbar:
        results['interpolation'] = nb_interpolation.evaluate_interpolation(X_dict, test_labels, pbar)
    
    print("\nRésultats avec interpolation:")
    print(f"Accuracy: {results['interpolation']['accuracy']:.4f}")
    print(f"Recall: {results['interpolation']['recall']:.4f}")
    
    return results

def run_ann_experiments(train_texts, test_texts, train_labels, test_labels):
    """Exécute les expériences avec le réseau de neurones artificiel (ANN)
    
    Processus :
    1. Extraction des caractéristiques TF-IDF
    2. Test différentes configurations (5, 10, 15 mots)
    3. Entraînement et évaluation pour chaque configuration
    
    Returns:
        dict: Résultats pour chaque configuration
    """
    feature_extractor = FeatureExtractor()
    results = {}

    # Extraction des caractéristiques TF-IDF
    print("\nCalcul des TF-IDF...")
    with tqdm(total=2, desc="Extraction TF-IDF") as pbar:
        train_tfidf = feature_extractor.extract_tfidf_features(train_texts)
        pbar.update(1)
        test_tfidf = feature_extractor.tfidf_vectorizer.transform(test_texts)
        pbar.update(1)

    # Test différentes tailles de vocabulaire
    for n_words in [5, 10, 15]:
        print(f"\nANN avec {n_words} mots...")
        top_words = feature_extractor.get_top_words(n_words)
        
        # Création et entraînement du modèle
        ann_classifier = TextClassifierANN(hidden_layer_size=100)
        
        # Entraînement avec suivi de progression
        with tqdm(total=500, desc="Entraînement") as pbar:
            ann_classifier.train(train_tfidf, train_labels, progress_bar=pbar)
        
        # Évaluation avec suivi de progression
        with tqdm(total=1, desc="Évaluation") as pbar:
            results[f'ann_{n_words}_words'] = ann_classifier.evaluate(test_tfidf, test_labels, pbar)
        
        # Affichage des résultats
        print(f"Accuracy: {results[f'ann_{n_words}_words']['accuracy']:.4f}")
        print(f"Recall: {results[f'ann_{n_words}_words']['recall']:.4f}")
    
    return results

def main():
    """Fonction principale qui orchestre tout le processus
    
    Étapes :
    1. Configuration et préparation des dossiers
    2. Chargement et prétraitement des données
    3. Exécution des expériences Naive Bayes
    4. Exécution des expériences ANN
    5. Sauvegarde des résultats
    """
    # Configuration initiale
    args = setup_argparse()
    ensure_directories(args.output_dir)
    
    print("Démarrage du processus de classification...")
    
    # Préparation des données
    preprocessor = DataPreprocessor()
    
    # Chargement et prétraitement des données d'entraînement
    print("Chargement et prétraitement des données d'entraînement...")
    train_df = preprocessor.load_data(args.train_path)
    train_df = preprocessor.preprocess(train_df)
    
    # Chargement et prétraitement des données de test
    print("Chargement et prétraitement des données de test...")
    test_df = preprocessor.load_data(args.test_path)
    test_df = preprocessor.preprocess(test_df)
    
    # Extraction des textes et labels
    train_texts = train_df['Cleaned_Text']
    test_texts = test_df['Cleaned_Text']
    train_labels = train_df['Class']
    test_labels = test_df['Class']
    
    # Exécution des expériences
    print("\nDémarrage des expériences Naive Bayes...")
    nb_results = run_naive_bayes_experiments(train_texts, test_texts, train_labels, test_labels)
    
    print("\nDémarrage des expériences ANN...")
    ann_results = run_ann_experiments(train_texts, test_texts, train_labels, test_labels)
    
    # Sauvegarde des résultats
    metrics_path = os.path.join(args.output_dir, 'metrics', 'results.txt')
    
    # Sauvegarde des résultats Naive Bayes par méthode
    save_metrics(nb_results, 'Naive Bayes - Standard', metrics_path)
    save_metrics({k: v for k, v in nb_results.items() if 'laplace' in k}, 
                'Naive Bayes - Laplace', metrics_path)
    save_metrics({k: v for k, v in nb_results.items() if 'goodturing' in k}, 
                'Naive Bayes - Good-Turing', metrics_path)
    if 'interpolation' in nb_results:
        save_metrics({'interpolation': nb_results['interpolation']}, 
                    'Naive Bayes - Interpolation', metrics_path)
    
    # Sauvegarde des résultats ANN
    save_metrics(ann_results, 'ANN', metrics_path)
    
    print(f"\nTraitement terminé. Résultats sauvegardés dans {metrics_path}")

# Point d'entrée du script
if __name__ == "__main__":
    main()

    