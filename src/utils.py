import numpy as np
from sklearn.model_selection import train_test_split
import os


def save_metrics(metrics, model_name, file_path):
    """
    Sauvegarde les métriques de base dans un fichier texte.
    
    Algorithme:
    1. Ouvre le fichier en mode append ('a')
    2. Écrit le nom du modèle
    3. Pour chaque métrique:
       - Écrit la paire clé-valeur sur une nouvelle ligne
    
    Args:
        metrics (dict): Dictionnaire des métriques à sauvegarder
        model_name (str): Nom du modèle évalué
        file_path (str): Chemin du fichier de sortie
    """
    with open(file_path, 'a') as f:
        f.write(f"\nRésultats pour {model_name}:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def save_detailed_metrics(results_dict, model_name, output_dir):
    """
    Sauvegarde les métriques détaillées dans une structure de dossiers organisée.
    
    Structure des dossiers:
    output_dir/
    └── metrics/
        └── model_name/
            ├── best_results.txt    # Meilleurs résultats globaux
            ├── mean_results.txt    # Moyennes par configuration
            ├── detailed_results.txt # Résultats détaillés par fold
            └── confusion_matrices.txt # Matrices de confusion par fold
    
    Algorithme:
    1. Création de la structure de dossiers
    2. Sauvegarde des meilleurs résultats:
       - Trouve la meilleure configuration (accuracy max)
       - Écrit les métriques détaillées
    3. Sauvegarde des résultats moyens:
       - Pour chaque configuration, écrit les moyennes
    4. Sauvegarde des résultats par fold:
       - Pour chaque fold, écrit les résultats détaillés
    5. Sauvegarde des matrices de confusion:
       - Pour chaque fold et configuration
       - Calcule les métriques par classe
    
    Args:
        results_dict (dict): Dictionnaire contenant tous les résultats
        model_name (str): Nom du modèle (ex: 'naive_bayes', 'ann')
        output_dir (str): Dossier racine pour la sauvegarde
    """
    # Création des dossiers
    model_dir = os.path.join(output_dir, 'metrics', model_name.lower().replace(' ', '_'))
    os.makedirs(model_dir, exist_ok=True)
    
    # Sauvegarde des meilleurs résultats
    with open(os.path.join(model_dir, 'best_results.txt'), 'w') as f:
        f.write(f"=== Meilleurs résultats pour {model_name} ===\n\n")
        best_accuracy = 0
        best_config = None
        
        for config, results in results_dict['mean_results'].items():
            # Adaptation pour gérer les deux formats de clés
            accuracy = results.get('accuracy_mean', results.get('mean_accuracy', 0))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
        
        if best_config:
            best_results = results_dict['mean_results'][best_config]
            f.write(f"Meilleure configuration: {best_config}\n")
            # Adaptation pour gérer les deux formats de clés
            accuracy = best_results.get('mean_accuracy', best_results.get('accuracy_mean', 0))
            accuracy_std = best_results.get('std_accuracy', best_results.get('accuracy_std', 0))
            recall = best_results.get('mean_recall', best_results.get('recall_mean', 0))
            recall_std = best_results.get('std_recall', best_results.get('recall_std', 0))
            
            f.write(f"Accuracy: {accuracy:.4f} (±{accuracy_std:.4f})\n")
            f.write(f"Recall: {recall:.4f} (±{recall_std:.4f})\n")
    
    # Sauvegarde des résultats moyens
    with open(os.path.join(model_dir, 'mean_results.txt'), 'w') as f:
        f.write(f"=== Résultats moyens pour {model_name} ===\n\n")
        for config, results in results_dict['mean_results'].items():
            f.write(f"\nConfiguration: {config}\n")
            # Adaptation pour gérer les deux formats de clés
            accuracy = results.get('mean_accuracy', results.get('accuracy_mean', 0))
            accuracy_std = results.get('std_accuracy', results.get('accuracy_std', 0))
            recall = results.get('mean_recall', results.get('recall_mean', 0))
            recall_std = results.get('std_recall', results.get('recall_std', 0))
            
            f.write(f"Accuracy: {accuracy:.4f} (±{accuracy_std:.4f})\n")
            f.write(f"Recall: {recall:.4f} (±{recall_std:.4f})\n")
    
    # Sauvegarde des résultats détaillés par fold
    with open(os.path.join(model_dir, 'detailed_results.txt'), 'w') as f:
        f.write(f"=== Résultats détaillés par fold pour {model_name} ===\n\n")
        for fold_result in results_dict['fold_results']:
            f.write(f"\nFold {fold_result['fold_idx'] + 1}:\n")
            for config, metrics in fold_result['results'].items():
                f.write(f"\nConfiguration: {config}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                
                # Ajout des top words pour ANN
                if model_name == 'ANN' and 'top_words' in metrics:
                    f.write("\nMots les plus importants:\n")
                    for word, score in metrics['top_words']:
                        f.write(f"{word}: {score:.4f}\n")
    
    # Sauvegarde des matrices de confusion
    with open(os.path.join(model_dir, 'confusion_matrices.txt'), 'w') as f:
        f.write(f"=== Matrices de confusion pour {model_name} ===\n\n")
        for fold_result in results_dict['fold_results']:
            f.write(f"\nFold {fold_result['fold_idx'] + 1}:\n")
            for config, metrics in fold_result['results'].items():
                f.write(f"\nConfiguration: {config}\n")
                f.write("Matrice de confusion:\n")
                cm = metrics['confusion_matrix']
                f.write(np.array2string(cm, separator=', '))
                f.write('\n')
                
                # Calcul des métriques par classe
                n_classes = cm.shape[0]
                for i in range(n_classes):
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    tn = cm.sum() - (tp + fp + fn)
                    
                    accuracy = (tp + tn) / cm.sum()
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    f.write(f"\nClasse {i+1}:\n")
                    f.write(f"Accuracy: {accuracy:.4f}\n")
                    f.write(f"Recall: {recall:.4f}\n")

def calculate_nb_means(results):
    """
    Calcule les moyennes des métriques pour Naive Bayes.
    
    Algorithme:
    1. Pour chaque méthode (standard, laplace, goodturing):
       - Pour chaque n-gram (1, 2, 3):
         * Collecte accuracy et recall de tous les folds
         * Calcule moyenne et écart-type
         * Stocke dans results['mean_results']
    
    Args:
        results (dict): Dictionnaire contenant les résultats par fold
    """
    for method in ['standard', 'laplace', 'goodturing']:
        for n_gram in [1, 2, 3]:
            accuracies = []
            recalls = []
            
            for fold_result in results['fold_results']:
                res = fold_result['results']
                key = f'{n_gram}-gram_{method}'
                accuracies.append(res[key]['accuracy'])
                recalls.append(res[key]['recall'])
            
            results['mean_results'][f'{n_gram}-gram_{method}'] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'recall_mean': np.mean(recalls),
                'recall_std': np.std(recalls)
            }

def calculate_ann_means(results):
    """
    Calcule les moyennes des métriques pour ANN.
    
    Algorithme:
    1. Pour chaque taille de vocabulaire (5, 10, 15 mots):
       - Collecte accuracy et recall de tous les folds
       - Calcule moyenne et écart-type
       - Stocke dans results['mean_results']
    
    Args:
        results (dict): Dictionnaire contenant les résultats par fold
    """
    for n_words in [5, 10, 15]:
        accuracies = []
        recalls = []
        
        for fold_result in results['fold_results']:
            res = fold_result['results']
            key = f'ann_{n_words}_words'
            accuracies.append(res[key]['accuracy'])
            recalls.append(res[key]['recall'])
        
        results['mean_results'][f'ann_{n_words}_words'] = {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls)
        }

def calculate_ann_spe_means(results):
    """
    Calcule les moyennes des métriques pour ANN-SPE.
    
    Algorithme:
    1. Pour la taille de vocabulaire fixe (15 mots):
       - Collecte accuracy et recall de tous les folds
       - Calcule moyenne et écart-type
       - Stocke dans results['mean_results']
    2. Affiche un résumé des résultats
    
    Args:
        results (dict): Dictionnaire contenant les résultats par fold
    """
    mean_results = {}
    
    # Pour chaque taille de mots (maintenant seulement 15)
    for n_words in [15]:  # Modifié pour correspondre à la seule taille utilisée
        accuracies = []
        recalls = []
        
        # Collecte des résultats de chaque fold
        for res in results['fold_results']:
            key = f'ann_spe_{n_words}_words'  # Correspond maintenant à la clé utilisée
            accuracies.append(res['results'][key]['accuracy'])
            recalls.append(res['results'][key]['recall'])
        
        # Calcul des moyennes
        mean_results[f'ann_spe_{n_words}_words'] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls)
        }
    
    # Mise à jour des résultats moyens
    results['mean_results'] = mean_results
    
    # Affichage des résultats
    print("\nRésultats moyens ANN-SPE:")
    for key, metrics in mean_results.items():
        print(f"\n{key}:")
        print(f"Accuracy: {metrics['mean_accuracy']:.4f} (±{metrics['std_accuracy']:.4f})")
        print(f"Recall: {metrics['mean_recall']:.4f} (±{metrics['std_recall']:.4f})")