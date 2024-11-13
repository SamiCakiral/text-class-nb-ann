from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import time
import numpy as np
from tqdm import tqdm

class TextClassifierANN:
    def __init__(self, hidden_layer_size=100):
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size,),
            max_iter=500,
            random_state=42,
            verbose=False,
            early_stopping=False,
            validation_fraction=0.1
        )
        self.loss_history = []
        self.validation_scores = []
        
    def predict_convergence(self, window_size=5):
        """Prédit si le modèle va bientôt converger basé sur la stabilité de la loss"""
        if len(self.loss_history) < window_size:
            return False
            
        recent_losses = self.loss_history[-window_size:]
        loss_change = np.abs(np.diff(recent_losses))
        avg_change = np.mean(loss_change)
        
        # Si le changement moyen est très petit, on prédit la convergence
        return avg_change < 0.001
        
    def train(self, X, y, progress_bar=None):
        """Entraîne le réseau de neurones avec un affichage optimisé"""
        start_time = time.time()
        n_samples = X.shape[0]
        classes = np.unique(y)
        
        # Initialisation du modèle avec un premier passage
        self.model.partial_fit(X, y, classes=classes)
        
        if progress_bar:
            progress_bar.reset(total=100)  # On utilise 100 itérations maximum
        
        # Boucle d'entraînement
        for epoch in range(100):  # Limite à 100 epochs
            # Entraînement sur un batch
            self.model.partial_fit(X, y, classes=classes)
            
            # Calcul de la loss et mise à jour de l'historique
            if hasattr(self.model, 'loss_'):
                loss = self.model.loss_
                self.loss_history.append(loss)
                
                # Mise à jour de la barre de progression
                if progress_bar and epoch % 2 == 0:  # Update tous les 2 epochs
                    elapsed = time.time() - start_time
                    remaining = "convergence proche" if self.predict_convergence() else "en cours"
                    progress_bar.set_description(
                        f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                        f"Temps: {elapsed:.1f}s | État: {remaining}"
                    )
                    progress_bar.update(2)
                
                # Vérification de la convergence
                if self.predict_convergence() and epoch > 20:
                    if progress_bar:
                        progress_bar.set_description(f"Convergence détectée après {epoch} epochs")
                    break
                    
                # Early stopping manuel basé sur la loss
                if len(self.loss_history) > 10:
                    if abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-4:
                        if progress_bar:
                            progress_bar.set_description(
                                f"Early stopping après {epoch} epochs"
                            )
                        break
        
        if progress_bar:
            progress_bar.update(progress_bar.total - progress_bar.n)  # Compléter la barre
            progress_bar.set_description(f"Terminé en {time.time() - start_time:.1f}s")
        
    def predict(self, X):
        """Fait des prédictions"""
        return self.model.predict(X)
        
    def evaluate(self, X, y_true, pbar=None):
        """Évalue le modèle et retourne les métriques"""
        start_time = time.time()
        y_pred = self.predict(X)
        if pbar:
            pbar.set_description(f"Évaluation ({time.time() - start_time:.2f}s)")
            pbar.update(len(y_true))
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }