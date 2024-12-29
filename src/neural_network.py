import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import numpy as np
import time

class TextClassifierNN(nn.Module):
    def __init__(self, input_size, n_classes=4, hidden_size=100):
        super(TextClassifierNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, n_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        """Méthode forward requise pour tous les modèles PyTorch"""
        return self.network(x)

class TextClassifierANN:
    def __init__(self, hidden_layer_size=100):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else 
                                 "cpu")
        print(f"Utilisation de l'accélérateur: {self.device}")
        self.hidden_layer_size = hidden_layer_size
        self.model = None
        self.loss_history = []
        
    def _to_tensor(self, X, y=None):
        """Convertit les données en tensors PyTorch"""
        X_tensor = torch.FloatTensor(X.toarray() if hasattr(X, 'toarray') else X).to(self.device)
        if y is not None:
            y_tensor = torch.LongTensor(y.astype(int)).to(self.device)
            return X_tensor, y_tensor
        return X_tensor
        
    def train(self, X, y, progress_bar=None):
        """Entraîne le réseau de neurones avec PyTorch"""
        # Initialisation du modèle
        input_size = X.shape[1]
        self.model = TextClassifierNN(input_size, self.hidden_layer_size).to(self.device)
        
        # Paramètres d'entraînement
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Conversion des données
        X_tensor, y_tensor = self._to_tensor(X, y)
        
        # Paramètres de batch
        batch_size = 512
        n_batches = int(np.ceil(X.shape[0] / batch_size))
        
        # Entraînement
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        if progress_bar:
            progress_bar.reset(total=100)
        
        start_time = time.time()
        
        for epoch in range(100):  # max 100 epochs
            total_loss = 0
            self.model.train()
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X.shape[0])
                
                batch_X = X_tensor[start_idx:end_idx]
                batch_y = y_tensor[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            self.loss_history.append(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if progress_bar and epoch % 2 == 0:
                elapsed = time.time() - start_time
                progress_bar.set_description(
                    f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                    f"Temps: {elapsed:.1f}s"
                )
                progress_bar.update(2)
            
            if patience_counter >= patience:
                if progress_bar:
                    progress_bar.set_description(f"Early stopping après {epoch} epochs")
                break
        
        if progress_bar:
            progress_bar.update(progress_bar.total - progress_bar.n)
            progress_bar.set_description(f"Terminé en {time.time() - start_time:.1f}s")
    
    def predict(self, X):
        """Fait des prédictions"""
        self.model.eval()
        X_tensor = self._to_tensor(X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
    
    def evaluate(self, X, y_true):
        """Évalue le modèle et retourne les métriques"""
        y_pred = self.predict(X)
                    
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }