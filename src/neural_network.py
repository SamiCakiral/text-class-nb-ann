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

class EnhancedTextClassifierNN(nn.Module):
    def __init__(self, input_size, n_classes=4, hidden_size=100, tfidf_dim=None, stats_dim=None):
        super().__init__()
        
        self.tfidf_dim = tfidf_dim
        self.stats_dim = stats_dim
        
        if tfidf_dim and stats_dim:
            # Branche TF-IDF
            self.tfidf_layers = nn.Sequential(
                nn.Linear(tfidf_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2)
            )
            
            # Branche pour les features statistiques (12 features)
            self.stats_layers = nn.Sequential(
                nn.Linear(stats_dim, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # Couche supplémentaire pour mieux traiter les features complexes
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.LayerNorm(hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # Couche de combinaison
            combined_size = (hidden_size // 2) + (hidden_size // 4)
            self.combine_layer = nn.Sequential(
                nn.Linear(combined_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        else:
            # Version sans séparation des features
            self.input_layer = nn.Sequential(
                nn.Linear(input_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        
        # Couches de classification plus profondes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 4, n_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        if self.tfidf_dim and self.stats_dim:
            # Séparation et traitement des features
            tfidf_features = x[:, :self.tfidf_dim]
            stats_features = x[:, self.tfidf_dim:]
            
            tfidf_out = self.tfidf_layers(tfidf_features)
            stats_out = self.stats_layers(stats_features)
            
            # Combinaison
            combined = torch.cat((tfidf_out, stats_out), dim=1)
            x = self.combine_layer(combined)
        else:
            x = self.input_layer(x)
        
        return self.classifier(x)

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
        
    def train(self, X, y, X_val=None, y_val=None, progress_bar=None):
        """Entraîne le réseau de neurones avec PyTorch et validation"""
        # Initialisation du modèle
        input_size = X.shape[1]
        self.model = TextClassifierNN(input_size, hidden_size=self.hidden_layer_size).to(self.device)
        
        # Paramètres d'entraînement
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Conversion des données
        X_tensor, y_tensor = self._to_tensor(X, y)
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._to_tensor(X_val, y_val)
        
        # Paramètres de batch
        batch_size = 512
        n_batches = int(np.ceil(X.shape[0] / batch_size))
        
        # Entraînement
        best_loss = float('inf')
        best_model = None
        patience = 5
        patience_counter = 0
        
        if progress_bar:
            progress_bar.reset(total=100)
        
        start_time = time.time()
        
        for epoch in range(100):  # max 100 epochs
            # Phase d'entraînement
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
            
            train_loss = total_loss / n_batches
            
            # Phase de validation
            val_loss = None
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                # Early stopping sur la perte de validation
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                # Si pas de données de validation, utiliser la perte d'entraînement
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_model = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # Enregistrement des pertes
            self.loss_history.append({
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            if progress_bar and epoch % 2 == 0:
                elapsed = time.time() - start_time
                status = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    status += f" | Val Loss: {val_loss:.4f}"
                status += f" | Temps: {elapsed:.1f}s"
                progress_bar.set_description(status)
                progress_bar.update(2)
            
            if patience_counter >= patience:
                if progress_bar:
                    progress_bar.set_description(f"Early stopping après {epoch} epochs")
                break
        
        # Restauration du meilleur modèle
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        if progress_bar:
            progress_bar.update(progress_bar.total - progress_bar.n)
    
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

class EnhancedTextClassifierANN(TextClassifierANN):
    def __init__(self, hidden_layer_size=100, input_size=None, tfidf_dim=None, stats_dim=None):
        super().__init__(hidden_layer_size)
        self.input_size = input_size
        self.tfidf_dim = tfidf_dim
        self.stats_dim = stats_dim
        self.scheduler = None
    
    def train(self, X, y, X_val=None, y_val=None, progress_bar=None):
        """Version améliorée de l'entraînement avec apprentissage adaptatif"""
        # Vérification des données
        print(f"Stats des features d'entrée:")
        print(f"Train - Mean: {X.mean():.3f}, Std: {X.std():.3f}")
        if X_val is not None:
            print(f"Val - Mean: {X_val.mean():.3f}, Std: {X_val.std():.3f}")
        
        input_size = self.input_size or X.shape[1]
        self.model = EnhancedTextClassifierNN(
            input_size=input_size,
            hidden_size=self.hidden_layer_size,
            tfidf_dim=self.tfidf_dim,
            stats_dim=self.stats_dim
        ).to(self.device)
        
        # Optimisation avec learning rate adaptatif
        criterion = nn.NLLLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Learning rate scheduler avec ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Le reste de la méthode train reste identique
        X_tensor, y_tensor = self._to_tensor(X, y)
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._to_tensor(X_val, y_val)
        
        batch_size = 512
        n_batches = int(np.ceil(X.shape[0] / batch_size))
        
        best_loss = float('inf')
        best_model = None
        patience = 10  # Early stopping patience
        patience_counter = 0
        
        if progress_bar:
            progress_bar.reset(total=100)
        
        start_time = time.time()
        
        for epoch in range(2000):  # Augmentation à 2000 epochs
            # Phase d'entraînement
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            train_loss = total_loss / n_batches
            
            # Phase de validation
            val_loss = None
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                # Ajustement du learning rate basé sur la perte de validation
                self.scheduler.step(val_loss)
                
                # Early stopping sur la perte de validation
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                # Si pas de données de validation, utiliser la perte d'entraînement
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_model = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                self.scheduler.step(train_loss)
            
            # Enregistrement des pertes
            self.loss_history.append({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            if progress_bar and epoch % 20 == 0:  # Mise à jour moins fréquente de la barre de progression
                elapsed = time.time() - start_time
                status = f"Epoch {epoch:4d}/{2000} | Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    status += f" | Val Loss: {val_loss:.4f}"
                status += f" | LR: {optimizer.param_groups[0]['lr']:.2e} | Temps: {elapsed:.1f}s"
                progress_bar.set_description(status)
                progress_bar.update(min(20, progress_bar.total - progress_bar.n))
            
            if patience_counter >= patience:
                if progress_bar:
                    progress_bar.set_description(f"Early stopping après {epoch + 1} epochs")
                break
        
        # Restauration du meilleur modèle
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        if progress_bar:
            progress_bar.update(progress_bar.total - progress_bar.n)