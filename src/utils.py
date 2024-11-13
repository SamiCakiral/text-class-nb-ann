import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def split_data(X, y, test_size=0.2, random_state=42):
    """Divise les données en ensembles d'entraînement et de test"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def plot_confusion_matrix(cm, classes):
    """Affiche la matrice de confusion"""
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies classes')
    plt.title('Matrice de confusion')
    plt.show()

def save_metrics(metrics, model_name, file_path):
    """Sauvegarde les métriques dans un fichier"""
    with open(file_path, 'a') as f:
        f.write(f"\nRésultats pour {model_name}:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")