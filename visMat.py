import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualiser_matrice_confusion(matrice_confusion, titre='Matrice de Confusion', classes=None, cmap='Blues'):
    """
    Visualise une matrice de confusion avec une heatmap.
    
    Args:
        matrice_confusion (np.array): Matrice de confusion à visualiser
        titre (str): Titre du graphique
        classes (list): Liste des noms de classes
        cmap (str): Palette de couleurs à utiliser
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice_confusion, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes or range(len(matrice_confusion)),
                yticklabels=classes or range(len(matrice_confusion)))
    
    plt.title(titre)
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Exemple d'utilisation
    matrice = np.array([[ 574,  387,  218,  721],
                       [ 263, 1055,  227,  355],
                       [ 362,  457,  441,  640],
                       [ 397,  356,  296,  851]])
    
    visualiser_matrice_confusion(matrice, 
                               classes=['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4'])