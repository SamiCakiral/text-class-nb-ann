import numpy as np
from sklearn.model_selection import train_test_split
import os


def save_metrics(metrics, model_name, file_path):
    """Sauvegarde les métriques dans un fichier"""
    with open(file_path, 'a') as f:
        f.write(f"\nRésultats pour {model_name}:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def save_detailed_metrics(results_dict, model_name, output_dir):
    """
    Sauvegarde les métriques détaillées dans des fichiers séparés
    
    Structure:
    - output_dir/
        - naive_bayes/
            - best_results.txt
            - mean_results.txt
            - detailed_results.txt
            - confusion_matrices.txt
        - ann/
            - best_results.txt
            - mean_results.txt
            - detailed_results.txt
            - confusion_matrices.txt
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
            if results['accuracy_mean'] > best_accuracy:
                best_accuracy = results['accuracy_mean']
                best_config = config
        
        if best_config:
            best_results = results_dict['mean_results'][best_config]
            f.write(f"Meilleure configuration: {best_config}\n")
            f.write(f"Accuracy: {best_results['accuracy_mean']:.4f} (±{best_results['accuracy_std']:.4f})\n")
            f.write(f"Recall: {best_results['recall_mean']:.4f} (±{best_results['recall_std']:.4f})\n")
    
    # Sauvegarde des résultats moyens
    with open(os.path.join(model_dir, 'mean_results.txt'), 'w') as f:
        f.write(f"=== Résultats moyens pour {model_name} ===\n\n")
        for config, results in results_dict['mean_results'].items():
            f.write(f"\nConfiguration: {config}\n")
            f.write(f"Accuracy: {results['accuracy_mean']:.4f} (±{results['accuracy_std']:.4f})\n")
            f.write(f"Recall: {results['recall_mean']:.4f} (±{results['recall_std']:.4f})\n")
    
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
    """Calcule les moyennes pour Naive Bayes"""
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
    """Calcule les moyennes pour ANN"""
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