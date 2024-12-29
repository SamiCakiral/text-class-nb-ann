import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualiser_matrice_confusion(matrice_confusion, classes=None, titre='Confusion Matrix', cmap='Blues'):
    """
    Visualise une matrice de confusion avec une heatmap.
    
    Args:
        matrice_confusion (np.array): Matrice de confusion à visualiser
        classes (list, optional): Liste des noms de classes. Par défaut None.
        titre (str, optional): Titre du graphique. Par défaut 'Confusion Matrix'.
        cmap (str, optional): Colormap à utiliser. Par défaut 'Blues'.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice_confusion, annot=True, fmt='d', cmap=cmap, 
                xticklabels=classes or range(len(matrice_confusion)),
                yticklabels=classes or range(len(matrice_confusion)))
    
    plt.title(titre)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
matrice = np.array([[5308,  198,  283,  211],
 [ 150, 5685,   77,   88],
 [ 254,   94, 5157,  495],
 [ 263,  127,  578, 5032]])
visualiser_matrice_confusion(matrice, 
                              classes=['Class 1', 'Class 2', 'Class 3', 'Class 4'])