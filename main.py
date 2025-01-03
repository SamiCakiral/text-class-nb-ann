import os
from pathlib import Path
import argparse
from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor, EnhancedFeatureExtractor
from src.naive_bayes import CustomNaiveBayes
from src.neural_network import TextClassifierANN, EnhancedTextClassifierANN
import src.utils as utils
from tqdm import tqdm
import numpy as np

def setup_argparse():
    """
    Configuration des arguments en ligne de commande.
    
    Arguments disponibles:
    - mode: Type d'expérience à exécuter (ann, nbayes, annspe, all)
    - output_dir: Dossier de sortie pour les résultats
    - n_folds: Nombre de folds pour la validation croisée
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(description='NLP Classification Project')
    parser.add_argument('--mode', type=str, choices=['ann', 'nbayes', 'annspe', 'all'], 
                       default='all', help='Mode d\'exécution (ann: réseaux de neurones, nbayes: naive bayes, annspe: réseau de neurones spécialisé, all: tous)')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Dossier de sortie pour les résultats')
    parser.add_argument('--n_folds', type=int, default=5, 
                       help='Nombre de folds pour la validation croisée')
    
    return parser.parse_args()

def ensure_directories(base_dir):
    """
    Crée la structure des dossiers pour sauvegarder les résultats.
    
    Structure:
    base_dir/
    ├── metrics/  # Métriques de performance
    └── models/   # Modèles sauvegardés
    
    Args:
        base_dir (str): Chemin du dossier racine
    """
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(base_dir, 'metrics')).mkdir(exist_ok=True)
    Path(os.path.join(base_dir, 'models')).mkdir(exist_ok=True)

def run_naive_bayes_experiments(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels):
    """
    Exécute les expériences avec l'algorithme Naive Bayes.
    
    Algorithme:
    1. Pour chaque n-gram (1, 2, 3):
       - Crée les vecteurs de caractéristiques
       - Teste Naive Bayes standard (sans lissage)
       - Teste avec lissage de Laplace
       - Teste avec Good-Turing
    2. Test final avec interpolation des n-grams
    
    Args:
        train_texts (list): Textes d'entraînement
        val_texts (list): Textes de validation
        test_texts (list): Textes de test
        train_labels (array): Labels d'entraînement
        val_labels (array): Labels de validation
        test_labels (array): Labels de test
    
    Returns:
        dict: Résultats des différentes expériences
    """
    feature_extractor = FeatureExtractor()
    results = {}
    X_dict = {}  # Pour l'interpolation

    # Test différentes tailles de n-grams (1, 2, 3)
    for n in tqdm(range(1, 4), desc="Progression globale n-grams"):
        print(f"\nTraitement des {n}-grams...")
        
        # Création des vecteurs de caractéristiques
        train_vectors = feature_extractor.create_ngram_vectors(train_texts, n)
        val_vectors = feature_extractor.count_vectorizers[n].transform(val_texts)
        test_vectors = feature_extractor.count_vectorizers[n].transform(test_texts)
        X_dict[n] = test_vectors  # Stockage pour l'interpolation
        
        # 1. Test Naive Bayes standard (sans smoothing)
        print(f"\nTest Naive Bayes standard pour {n}-gram...")
        nb_standard = CustomNaiveBayes(alpha=0)
        with tqdm(total=2000, desc=f"Entraînement standard {n}-gram") as pbar:
            nb_standard.train_model(train_vectors, train_labels, n, pbar)
            # Validation sur val_vectors
            val_results = nb_standard.evaluate(val_vectors, val_labels, n)
            print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        results[f'{n}-gram_standard'] = nb_standard.evaluate(test_vectors, test_labels, n)
        
        # 2. Test avec Laplace smoothing
        print(f"\nTest Naive Bayes avec Laplace smoothing pour {n}-gram...")
        nb_laplace = CustomNaiveBayes(alpha=1.0)
        with tqdm(total=2000, desc=f"Entraînement Laplace {n}-gram") as pbar:
            nb_laplace.train_model(train_vectors, train_labels, n, pbar)
            val_results = nb_laplace.evaluate(val_vectors, val_labels, n)
            print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        results[f'{n}-gram_laplace'] = nb_laplace.evaluate(test_vectors, test_labels, n)
        
        # 3. Test avec Good-Turing
        print(f"\nTest Naive Bayes avec Good-Turing pour {n}-gram...")
        nb_goodturing = CustomNaiveBayes(alpha=1.0)
        with tqdm(total=100, desc=f"Entraînement Good-Turing {n}-gram") as pbar:
            nb_goodturing.train_model(train_vectors, train_labels, n, pbar)
            val_results = nb_goodturing.evaluate(val_vectors, val_labels, n)
            print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        results[f'{n}-gram_goodturing'] = nb_goodturing.evaluate(test_vectors, test_labels, n)
        
        # Affichage des résultats de test pour ce n-gram
        for method in ['standard', 'laplace', 'goodturing']:
            print(f"\nRésultats de test pour {n}-gram avec {method}:")
            print(f"Accuracy: {results[f'{n}-gram_{method}']['accuracy']:.4f}")
            print(f"Recall: {results[f'{n}-gram_{method}']['recall']:.4f}")
    
    # 4. Test avec interpolation (combine les résultats des différents n-grams)
    print("\nTest Naive Bayes avec interpolation...")
    nb_interpolation = CustomNaiveBayes(alpha=1.0)
    # Entraînement sur tous les n-grams
    for n in range(1, 4):
        train_vectors = feature_extractor.create_ngram_vectors(train_texts, n)
        with tqdm(total=100, desc=f"Entraînement interpolation {n}-gram") as pbar:
            nb_interpolation.train_model(train_vectors, train_labels, n, pbar)
    
    # Validation avec interpolation
    val_dict = {n: feature_extractor.count_vectorizers[n].transform(val_texts) for n in range(1, 4)}
    val_results = nb_interpolation.evaluate_interpolation(val_dict, val_labels)
    print(f"\nValidation avec interpolation:")
    print(f"Accuracy: {val_results['accuracy']:.4f}")
    print(f"Recall: {val_results['recall']:.4f}")
    
    # Évaluation finale avec interpolation
    results['interpolation'] = nb_interpolation.evaluate_interpolation(X_dict, test_labels)
    print("\nRésultats de test avec interpolation:")
    print(f"Accuracy: {results['interpolation']['accuracy']:.4f}")
    print(f"Recall: {results['interpolation']['recall']:.4f}")
    
    return results

def run_ann_experiments(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels):
    """
    Exécute les expériences avec le réseau de neurones standard.
    
    Algorithme:
    1. Pour chaque taille de vocabulaire (5, 10, 15 mots):
       - Extrait les caractéristiques
       - Entraîne le modèle avec early stopping
       - Évalue sur validation et test
    
    Args:
        train_texts (list): Textes d'entraînement
        val_texts (list): Textes de validation
        test_texts (list): Textes de test
        train_labels (array): Labels d'entraînement
        val_labels (array): Labels de validation
        test_labels (array): Labels de test
    
    Returns:
        dict: Résultats des différentes configurations
    """
    feature_extractor = FeatureExtractor()
    results = {}
    
    # Extraction des caractéristiques TF-IDF
    print("\nCalcul des TF-IDF...")
    train_tfidf = feature_extractor.extract_tfidf_features(train_texts)
    val_tfidf = feature_extractor.tfidf_vectorizer.transform(val_texts)
    test_tfidf = feature_extractor.tfidf_vectorizer.transform(test_texts)

    # Test différentes tailles de vocabulaire
    for n_words in [5, 10, 15]:
        print(f"\nANN avec {n_words} mots...")
        top_words = feature_extractor.get_top_words(n_words)
        
        # Création et entraînement du modèle
        ann_classifier = TextClassifierANN(hidden_layer_size=100)
        
        # Entraînement avec validation
        with tqdm(total=100, desc="Entraînement") as pbar:
            ann_classifier.train(train_tfidf, train_labels, 
                               val_tfidf, val_labels, 
                               progress_bar=pbar)
        
        # Évaluation finale sur les données de test
        results[f'ann_{n_words}_words'] = ann_classifier.evaluate(test_tfidf, test_labels)
        results[f'ann_{n_words}_words']['top_words'] = top_words
        results[f'ann_{n_words}_words']['loss_history'] = ann_classifier.loss_history
        
        print(f"Accuracy: {results[f'ann_{n_words}_words']['accuracy']:.4f}")
        print(f"Recall: {results[f'ann_{n_words}_words']['recall']:.4f}")
    
    return results

def run_ann_spe_experiments(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels):
    """
    Exécute les expériences avec le réseau de neurones spécialisé.
    
    Algorithme:
    1. Extraction des caractéristiques améliorées
    2. Entraînement avec architecture spécialisée
    3. Évaluation avec métriques détaillées
    
    Args:
        train_texts (list): Textes d'entraînement
        val_texts (list): Textes de validation
        test_texts (list): Textes de test
        train_labels (array): Labels d'entraînement
        val_labels (array): Labels de validation
        test_labels (array): Labels de test
    
    Returns:
        dict: Résultats des expériences
    """
    feature_extractor = EnhancedFeatureExtractor()
    results = {}
    
    # Normalisation des labels
    train_labels = np.array(train_labels).ravel()
    val_labels = np.array(val_labels).ravel()
    test_labels = np.array(test_labels).ravel()
    
    for n_words in [15]:
        print(f"\nANN-SPE avec {n_words} mots et features optimisées...")
        
        # Extraction des features
        train_features = feature_extractor.extract_enhanced_features(train_texts, max_features=n_words)
        val_features = feature_extractor.extract_enhanced_features(val_texts, max_features=n_words)
        test_features = feature_extractor.extract_enhanced_features(test_texts, max_features=n_words)
        
        # Configuration et entraînement
        ann_classifier = EnhancedTextClassifierANN(
            hidden_layer_size=200,
            input_size=train_features.shape[1],
            tfidf_dim=n_words,
            stats_dim=6
        )
        
        with tqdm(total=100, desc="Entraînement") as pbar:
            ann_classifier.train(train_features, train_labels,
                               val_features, val_labels,
                               progress_bar=pbar)
        
        # Évaluation
        evaluation = ann_classifier.evaluate(test_features, test_labels)
        
        # Stockage des résultats dans le même format que Naive Bayes
        results[f'ann_spe_{n_words}_words'] = {
            'accuracy': evaluation['accuracy'],
            'recall': evaluation['recall'],
            'confusion_matrix': evaluation['confusion_matrix']
        }
        
        print(f"Accuracy: {evaluation['accuracy']:.4f}")
        print(f"Recall: {evaluation['recall']:.4f}")
    
    return results

def main():
    """
    Fonction principale qui orchestre tout le processus avec k-fold validation.
    
    Algorithme:
    1. Configuration et préparation:
       - Parse les arguments
       - Crée les dossiers nécessaires
       - Charge et prétraite les données
    
    2. Pour chaque fold:
       - Divise les données (train/val/test)
       - Si mode 'nbayes' ou 'all':
         * Exécute expériences Naive Bayes
       - Si mode 'ann' ou 'all':
         * Exécute expériences ANN
       - Si mode 'annspe' ou 'all':
         * Exécute expériences ANN-SPE
    
    3. Finalisation:
       - Calcule les moyennes sur tous les folds
       - Sauvegarde les résultats détaillés
    
    Utilisation:
        python main.py --mode [ann|nbayes|annspe|all] --output_dir results --n_folds 5
    """
    # Configuration initiale
    args = setup_argparse()
    ensure_directories(args.output_dir)
    
    # Chemins fixes pour les données
    train_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'
    
    print("Démarrage du processus de classification...")
    
    # Préparation des données avec n_folds folds (5 par défaut)
    preprocessor = DataPreprocessor(n_splits=args.n_folds)
    
    # Chargement et prétraitement des données d'entraînement
    print("Chargement et prétraitement des données d'entraînement...")
    train_df = preprocessor.load_data(train_path)
    train_df = preprocessor.preprocess(train_df, is_train=True)
    
    # Création des folds
    preprocessor.create_folds()
    
    # Chargement et prétraitement des données de test
    print("Chargement et prétraitement des données de test...")
    test_df = preprocessor.load_data(test_path)
    test_df = preprocessor.preprocess(test_df, is_train=False)
    
    # Dictionnaires pour stocker les résultats
    results = {
        'nb_results': {'fold_results': [], 'test_results': {}, 'mean_results': {}},
        'ann_results': {'fold_results': [], 'test_results': {}, 'mean_results': {}},
        'ann_spe_results': {'fold_results': [], 'test_results': {}, 'mean_results': {}}
    }
    
    # Pour chaque fold
    for fold_idx in range(preprocessor.n_splits):
        print(f"\n=== Traitement du fold {fold_idx + 1}/{preprocessor.n_splits} ===")
        
        # Récupération des données du fold
        fold_data = preprocessor.get_fold(fold_idx)
        
        # Expériences Naive Bayes
        if args.mode in ['nbayes', 'all']:
            print("\nDémarrage des expériences Naive Bayes...")
            fold_nb_results = run_naive_bayes_experiments(
                fold_data['X_train'], fold_data['X_val'], fold_data['X_test'],
                fold_data['y_train'], fold_data['y_val'], fold_data['y_test']
            )
            results['nb_results']['fold_results'].append({
                'fold_idx': fold_idx,
                'results': fold_nb_results
            })
        
        # Expériences ANN
        if args.mode in ['ann', 'all']:
            print("\nDémarrage des expériences ANN...")
            fold_ann_results = run_ann_experiments(
                fold_data['X_train'], fold_data['X_val'], fold_data['X_test'],
                fold_data['y_train'], fold_data['y_val'], fold_data['y_test']
            )
            results['ann_results']['fold_results'].append({
                'fold_idx': fold_idx,
                'results': fold_ann_results
            })
        
        # Expériences ANN-SPE
        if args.mode in ['annspe', 'all']:
            print("\nDémarrage des expériences ANN-SPE...")
            fold_ann_spe_results = run_ann_spe_experiments(
                fold_data['X_train'], fold_data['X_val'], fold_data['X_test'],
                fold_data['y_train'], fold_data['y_val'], fold_data['y_test']
            )
            results['ann_spe_results']['fold_results'].append({
                'fold_idx': fold_idx,
                'results': fold_ann_spe_results
            })
    
    # Calcul des moyennes sur tous les folds
    print("\nCalcul des moyennes sur tous les folds...")
    
    if args.mode in ['nbayes', 'all']:
        utils.calculate_nb_means(results['nb_results'])
        utils.save_detailed_metrics(results['nb_results'], 'Naive Bayes', args.output_dir)
    
    if args.mode in ['ann', 'all']:
        utils.calculate_ann_means(results['ann_results'])
        utils.save_detailed_metrics(results['ann_results'], 'ANN', args.output_dir)
    
    if args.mode in ['annspe', 'all']:
        utils.calculate_ann_spe_means(results['ann_spe_results'])
        utils.save_detailed_metrics(results['ann_spe_results'], 'ANN-SPE', args.output_dir)
    
    print(f"\nTraitement terminé. Résultats sauvegardés dans {args.output_dir}/metrics/")

if __name__ == "__main__":
    main()

    