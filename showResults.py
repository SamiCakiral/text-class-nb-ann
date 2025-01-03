import json
import numpy as np
import matplotlib.pyplot as plt
from visMat import visualiser_matrice_confusion
import seaborn as sns

def charger_resultats(chemin):
    """
    Charge les résultats à partir d'un fichier texte.
    
    Args:
        chemin (str): Chemin vers le fichier de résultats
        
    Returns:
        tuple: (accuracy, std_dev) ou (None, None) si non trouvé
    """
    try:
        with open(chemin, 'r') as f:
            contenu = f.read()
            
        # Extraction des valeurs numériques
        lignes = contenu.split('\n')
        for ligne in lignes:
            if 'Accuracy:' in ligne:
                # Extraire la partie numérique
                valeurs = ligne.split('Accuracy:')[1].strip()
                # Nettoyer la chaîne
                valeurs = valeurs.replace('(', '').replace(')', '')
                if '±' in valeurs:
                    acc_str, std_str = valeurs.split('±')
                    try:
                        acc = float(acc_str.strip())
                        std = float(std_str.strip())
                        return acc, std
                    except ValueError:
                        continue
        return None, None
        
    except FileNotFoundError:
        return None, None
    except Exception as e:
        return None, None

def extraire_matrice_confusion(chemin):
    """
    Extrait les matrices de confusion du fichier.
    
    Args:
        chemin (str): Chemin vers le fichier contenant les matrices
        
    Returns:
        list: Liste des matrices de confusion
    """
    try:
        matrices = []
        matrice_courante = []
        
        with open(chemin, 'r') as f:
            lignes = f.readlines()
            
        for ligne in lignes:
            if '[[' in ligne or '[' in ligne and len(ligne.strip()) > 2:
                # Nettoyer et convertir la ligne en nombres
                nombres = ligne.strip().replace('[', '').replace(']', '').split(',')
                nombres = [int(n.strip()) for n in nombres if n.strip()]
                if nombres:
                    matrice_courante.append(nombres)
                
            if ']]' in ligne and matrice_courante:
                if all(len(row) == len(matrice_courante[0]) for row in matrice_courante):
                    matrices.append(np.array(matrice_courante))
                matrice_courante = []
                
        return matrices
    except Exception as e:
        return []

def afficher_resultats():
    """Affiche tous les résultats d'analyse."""
    print("\n=== Analyse des résultats de classification ===\n")
    
    # Configuration des sous-plots
    plt.figure(figsize=(15, 10))
    
    # 1. Graphique de comparaison des modèles
    plt.subplot(2, 2, 1)
    modeles = ['naive_bayes', 'ann', 'ann-spe']
    resultats = {}
    
    print("Meilleures performances par modèle:")
    print("-" * 40)
    
    for modele in modeles:
        acc, std = charger_resultats(f'results/metrics/{modele}/best_results.txt')
        if acc is not None:
            resultats[modele] = (acc, std)
            print(f"{modele.upper():<15} : {acc:.4f} (±{std:.4f})")
    
    if resultats:
        x = np.arange(len(resultats))
        modeles = list(resultats.keys())
        accuracies = [resultats[m][0] for m in modeles]
        stds = [resultats[m][1] for m in modeles]
        
        plt.bar(x, accuracies, yerr=stds, capsize=5)
        plt.xticks(x, [m.upper() for m in modeles], rotation=45)
        plt.ylabel('Accuracy')
        plt.title('Comparaison des performances')
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # 2. Matrices de confusion
    for i, modele in enumerate(modeles, start=2):
        plt.subplot(2, 2, i)
        matrices = extraire_matrice_confusion(f'results/metrics/{modele}/confusion_matrices.txt')
        
        if matrices:
            sns.heatmap(matrices[0], annot=True, fmt='d', cmap='Blues',
                       xticklabels=['C1', 'C2', 'C3', 'C4'],
                       yticklabels=['C1', 'C2', 'C3', 'C4'])
            plt.title(f'Matrice de confusion - {modele.upper()}')
            plt.xlabel('Classe prédite')
            plt.ylabel('Classe réelle')
    
    plt.tight_layout()
    plt.show()

def main():
    """Fonction principale."""
    afficher_resultats()

if __name__ == "__main__":
    main() 