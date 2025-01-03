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
    """
    Affiche tous les résultats d'analyse avec comparaison des 4 modèles sur deux figures séparées.
    
    Figure 1: Histogramme comparatif des accuracies
    Figure 2: Matrices de confusion des 4 modèles
    """
    print("\n=== Analyse des résultats de classification ===\n")
    modeles = ['naive_bayes', 'ann', 'ann-spe', 'ann-doc']
    resultats = {}
    
    # Collecte des résultats
    print("Meilleures performances par modèle:")
    print("-" * 50)
    for modele in modeles:
        acc, std = charger_resultats(f'results/metrics/{modele}/best_results.txt')
        if acc is not None:
            resultats[modele] = (acc, std)
            print(f"{modele.upper():<15} : {acc:.4f} (±{std:.4f})")
    
    # Figure 1: Histogramme comparatif - Taille responsive
    plt.figure(figsize=(min(12, len(modeles) * 2.5), 6))
    if resultats:
        x = np.arange(len(resultats))
        modeles = list(resultats.keys())
        accuracies = [resultats[m][0] for m in modeles]
        stds = [resultats[m][1] for m in modeles]
        
        # Personnalisation des barres avec espacement adaptatif
        bar_width = min(0.8, 1.0/len(modeles))
        bars = plt.bar(x, accuracies, width=bar_width, yerr=stds, capsize=5)
        plt.xticks(x, [m.upper() for m in modeles], rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.title('Comparaison des performances des modèles')
        
        # Ajout des valeurs sur les barres avec position adaptative
        max_height = max(accuracies) + max(stds)
        for i, v in enumerate(accuracies):
            plt.text(i, v + stds[i] + max_height * 0.01, 
                    f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()

    
    # Figure 2: Matrices de confusion - Taille et disposition responsives
    n_models = len(modeles)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(7*n_cols, 5*n_rows))
    for i, modele in enumerate(modeles, 1):
        plt.subplot(n_rows, n_cols, i)
        matrices = extraire_matrice_confusion(f'results/metrics/{modele}/confusion_matrices.txt')
        
        if matrices:
            sns.heatmap(matrices[0], annot=True, fmt='d', cmap='Blues',
                       xticklabels=['C1', 'C2', 'C3', 'C4'],
                       yticklabels=['C1', 'C2', 'C3', 'C4'],
                       annot_kws={'size': min(10, 120//n_models)})  # Taille de police adaptative
            plt.title(f'Matrice de confusion - {modele.upper()}')
            plt.xlabel('Classe prédite')
            plt.ylabel('Classe réelle')
    
    plt.suptitle('Matrices de confusion pour tous les modèles', 
                 y=1.02, fontsize=min(16, 160//n_models))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustement des marges
    plt.show()

def main():
    """Fonction principale."""
    afficher_resultats()

if __name__ == "__main__":
    main() 