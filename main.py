import os
from pathlib import Path
import argparse
from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor
from src.naive_bayes import CustomNaiveBayes
from src.neural_network import TextClassifierANN
from src.utils import plot_confusion_matrix, save_metrics, save_detailed_metrics
from tqdm import tqdm
import numpy as np

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
    train_tfidf = feature_extractor.extract_tfidf_features(train_texts)
    test_tfidf = feature_extractor.tfidf_vectorizer.transform(test_texts)


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
        results[f'ann_{n_words}_words'] = ann_classifier.evaluate(test_tfidf, test_labels)
        
        # Affichage des résultats
        print(f"Accuracy: {results[f'ann_{n_words}_words']['accuracy']:.4f}")
        print(f"Recall: {results[f'ann_{n_words}_words']['recall']:.4f}")
    
    return results

def main():
    """Fonction principale qui orchestre tout le processus avec k-fold validation
    
    Étapes :
    1. Configuration et préparation des dossiers
    2. Chargement et prétraitement des données avec création des folds
    3. Exécution des expériences Naive Bayes sur chaque fold
    4. Exécution des expériences ANN sur chaque fold
    5. Sauvegarde des résultats moyens et par fold
    """
    # Configuration initiale
    args = setup_argparse()
    ensure_directories(args.output_dir)
    
    print("Démarrage du processus de classification...")
    
    # Préparation des données avec 5 folds
    preprocessor = DataPreprocessor(n_splits=5)
    
    # Chargement et prétraitement des données d'entraînement
    print("Chargement et prétraitement des données d'entraînement...")
    train_df = preprocessor.load_data(args.train_path)
    train_df = preprocessor.preprocess(train_df, is_train=True)
    
    # Création des folds
    folds = preprocessor.create_folds()
    
    # Chargement et prétraitement des données de test
    print("Chargement et prétraitement des données de test...")
    test_df = preprocessor.load_data(args.test_path)
    test_df = preprocessor.preprocess(test_df, is_train=False)
    
    # Dictionnaires pour stocker tous les résultats
    nb_results = {
        'fold_results': [],
        'test_results': {},
        'mean_results': {}
    }
    
    ann_results = {
        'fold_results': [],
        'test_results': {},
        'mean_results': {}
    }
    
    # Pour chaque fold
    for fold_idx in range(preprocessor.n_splits):
        print(f"\n=== Traitement du fold {fold_idx + 1}/{preprocessor.n_splits} ===")
        
        # Récupération des données du fold
        fold_data = preprocessor.get_fold(fold_idx)
        
        # Expériences Naive Bayes sur ce fold
        print("\nDémarrage des expériences Naive Bayes...")
        fold_nb_results = run_naive_bayes_experiments(
            fold_data['X_train'],
            fold_data['X_val'],  # Utilisation des données de validation au lieu du test
            fold_data['y_train'],
            fold_data['y_val']   # Utilisation des données de validation au lieu du test
        )
        nb_results['fold_results'].append({
            'fold_idx': fold_idx,
            'results': fold_nb_results
        })
        
        # Expériences ANN sur ce fold
        print("\nDémarrage des expériences ANN...")
        fold_ann_results = run_ann_experiments(
            fold_data['X_train'],
            fold_data['X_val'],  # Utilisation des données de validation au lieu du test
            fold_data['y_train'],
            fold_data['y_val']   # Utilisation des données de validation au lieu du test
        )
        ann_results['fold_results'].append({
            'fold_idx': fold_idx,
            'results': fold_ann_results
        })
    
    # Calcul des moyennes sur tous les folds
    print("\nCalcul des moyennes sur tous les folds...")
    
    # Pour Naive Bayes
    for method in ['standard', 'laplace', 'goodturing']:
        for n_gram in [1, 2, 3]:
            accuracies = []
            recalls = []
            
            for fold_result in nb_results['fold_results']:
                results = fold_result['results']
                key = f'{n_gram}-gram_{method}'
                accuracies.append(results[key]['accuracy'])
                recalls.append(results[key]['recall'])
            
            nb_results['mean_results'][f'{n_gram}-gram_{method}'] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'recall_mean': np.mean(recalls),
                'recall_std': np.std(recalls)
            }
    
    # Pour ANN
    for n_words in [5, 10, 15]:
        accuracies = []
        recalls = []
        
        for fold_result in ann_results['fold_results']:
            results = fold_result['results']
            key = f'ann_{n_words}_words'
            accuracies.append(results[key]['accuracy'])
            recalls.append(results[key]['recall'])
        
        ann_results['mean_results'][f'ann_{n_words}_words'] = {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls)
        }
    
    # Sauvegarde des résultats détaillés
    save_detailed_metrics(nb_results, 'Naive Bayes', args.output_dir)
    save_detailed_metrics(ann_results, 'ANN', args.output_dir)
    
    print(f"\nTraitement terminé. Résultats sauvegardés dans {args.output_dir}/metrics/")

if __name__ == "__main__":
    main()

    