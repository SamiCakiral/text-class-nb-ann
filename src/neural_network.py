import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import numpy as np
import time

class TextClassifierNN(nn.Module):
    """
    Réseau de neurones simple pour la classification de texte.
    
    Architecture:
        1. Couche d'entrée (BoW) -> Couche cachée (hidden_size)
        2. ReLU comme fonction d'activation
        3. Dropout (0.2) pour éviter le surapprentissage
        4. Couche cachée -> Couche de sortie (n_classes)
        5. LogSoftmax pour obtenir des probabilités normalisées

    Args:
        input_size (int): Taille du vocabulaire (dimension du BoW)
        n_classes (int): Nombre de classes à prédire (default: 4)
        hidden_size (int): Taille de la couche cachée (default: 100)
    """

    def __init__(self, input_size, n_classes=4, hidden_size=100):
        super(TextClassifierNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, n_classes),
            nn.LogSoftmax(dim=1)
        ) # network de classification de la forme : input (BoW) -> hidden_size (100) -> output (4 classes)
    
    def forward(self, x):
        """Méthode forward requise pour tous les modèles PyTorch"""
        return self.network(x)


class TextClassifierANN:
    """
    Wrapper pour l'entraînement et l'évaluation du réseau de neurones.
    
    Attributs:
        device (torch.device): Dispositif de calcul (GPU/CPU)
        hidden_layer_size (int): Taille de la couche cachée
        model (TextClassifierNN): Instance du modèle
        loss_history (list): Historique des pertes pendant l'entraînement
    """

    def __init__(self, hidden_layer_size=100):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else 
                                 "cpu")
        print(f"Utilisation de l'accélérateur: {self.device}")
        self.hidden_layer_size = hidden_layer_size
        self.model = None
        self.loss_history = []
        
    def _to_tensor(self, X, y=None):
        """
        Convertit les données en tenseurs PyTorch et les déplace sur le bon dispositif.

        Args:
            X (array/sparse matrix): Données d'entrée
            y (array, optional): Labels

        Returns:
            torch.Tensor: Données converties en tenseurs
        """
        X_tensor = torch.FloatTensor(X.toarray() if hasattr(X, 'toarray') else X).to(self.device) # conversion des données en tensor
        if y is not None:
            y_tensor = torch.LongTensor(y.astype(int)).to(self.device) # conversion des labels en tensor
            return X_tensor, y_tensor
        return X_tensor
        
    def train(self, X, y, X_val=None, y_val=None, progress_bar=None):
        """
        Entraîne le réseau de neurones avec early stopping.

        Algorithme:
        1. Initialisation du modèle et des optimiseurs
        2. Pour chaque epoch:
           - Phase d'entraînement par batch
           - Calcul de la perte sur l'ensemble de validation
           - Early stopping si pas d'amélioration
           - Mise à jour de la barre de progression

        Paramètres d'entraînement:
            - Optimiseur: Adam avec lr=0.001
            - Batch size: 8192
            - Early stopping: patience de 10 epochs
            - Critère: NLLLoss

        Args:
            X (array): Données d'entraînement
            y (array): Labels d'entraînement
            X_val (array, optional): Données de validation
            y_val (array, optional): Labels de validation
            progress_bar (tqdm, optional): Barre de progression
        """
        # Initialisation du modèle
        input_size = X.shape[1]
        self.model = TextClassifierNN(input_size, hidden_size=self.hidden_layer_size).to(self.device) # initialisation du modèle
        
        # Paramètres d'entraînement
        criterion = nn.NLLLoss() # initialisation de la fonction de perte (Negative Log Likelihood Loss)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001) # initialisation de l'optimiseur (Adam)
        
        # Conversion des données
        X_tensor, y_tensor = self._to_tensor(X, y)
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._to_tensor(X_val, y_val)
        
        # Paramètres de batch
        batch_size = 2048*4 # taille des batches
        n_batches = int(np.ceil(X.shape[0] / batch_size)) # nombre de batches   
        
        # Entraînement
        best_loss = float('inf')
        best_model = None
        patience = 10 # nombre de patience
        patience_counter = 0 # compteur de patience
        
        if progress_bar:
            progress_bar.reset(total=100)
        
        start_time = time.time()
        
        for epoch in range(1000):  
            # Phase d'entraînement
            total_loss = 0
            self.model.train()
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X.shape[0])
                
                batch_X = X_tensor[start_idx:end_idx]
                batch_y = y_tensor[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X) # forward pass    
                loss = criterion(outputs, batch_y) # calcul de la perte
                loss.backward() # backpropagation
                optimizer.step() # mise à jour des paramètres
                
                total_loss += loss.item()
            
            train_loss = total_loss / n_batches # calcul de la perte moyenne
            
            # Phase de validation
            val_loss = None
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor) # forward pass
                    val_loss = criterion(val_outputs, y_val_tensor).item() # calcul de la perte
                
                # Early stopping sur la perte de validation
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = self.model.state_dict().copy()
                    patience_counter = 0 # reset du compteur de patience
                else:
                    patience_counter += 1 # incrémentation du compteur de patience
            else:
                # Si pas de données de validation, utiliser la perte d'entraînement
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_model = self.model.state_dict().copy()
                    patience_counter = 0 # reset du compteur de patience
                else:
                    patience_counter += 1 # incrémentation du compteur de patience
            
            # Enregistrement des pertes
            self.loss_history.append({
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            if progress_bar and epoch % 1 == 0:
                elapsed = time.time() - start_time
                status = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}" # affichage de la perte d'entraînement
                if val_loss is not None:
                    status += f" | Val Loss: {val_loss:.4f}" # affichage de la perte de validation
                status += f" | Temps: {elapsed:.1f}s" # affichage du temps écoulé
                progress_bar.set_description(status)
                progress_bar.update(1)
            
            if patience_counter >= patience:
                if progress_bar:
                    progress_bar.set_description(f"Early stopping après {epoch} epochs") # affichage du nombre d'epochs avant l'early stopping
                break
        
        # Restauration du meilleur modèle
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        if progress_bar:
            progress_bar.update(progress_bar.total - progress_bar.n) # mise à jour de la barre de progression
    
    def predict(self, X):
        """Fait des prédictions"""
        self.model.eval()
        X_tensor = self._to_tensor(X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy() # retourne les prédictions
    
    def evaluate(self, X, y_true):
        """Évalue le modèle et retourne les métriques"""
        y_pred = self.predict(X) # prédictions
                    
        return {
            'accuracy': accuracy_score(y_true, y_pred), # calcul de l'accuracy
            'recall': recall_score(y_true, y_pred, average='weighted'), # calcul du recall
            'confusion_matrix': confusion_matrix(y_true, y_pred) # calcul de la matrice de confusion
        }

class EnhancedTextClassifierNN(nn.Module):
    """
    Version améliorée du réseau de neurones avec architecture plus complexe.
    
    Architecture:
    1. Si tfidf_dim et stats_dim sont fournis:
       - Branche TF-IDF: input -> hidden -> hidden/2
       - Branche statistique: input -> hidden/4
       - Combinaison des branches
    2. Sinon:
       - Architecture simple avec normalisation et dropout
    
    Args:
        input_size (int): Taille totale des features
        n_classes (int): Nombre de classes (default: 4)
        hidden_size (int): Taille de base des couches cachées (default: 64)
        tfidf_dim (int, optional): Dimension des features TF-IDF
        stats_dim (int, optional): Dimension des features statistiques
    """

    def __init__(self, input_size, n_classes=4, hidden_size=64, tfidf_dim=None, stats_dim=None):
        super().__init__()
        
        self.tfidf_dim = tfidf_dim
        self.stats_dim = stats_dim
        
        if tfidf_dim and stats_dim:
            # Branche TF-IDF
            self.tfidf_layers = nn.Sequential(
                nn.Linear(tfidf_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2)
            )
            
            # Branche statistique (6 features)
            self.stats_layers = nn.Sequential(
                nn.Linear(stats_dim, hidden_size // 4),
                nn.LayerNorm(hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Combinaison
            combined_size = (hidden_size // 2) + (hidden_size // 4)
            self.combine_layer = nn.Sequential(
                nn.Linear(combined_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        else:
            # Version simple sans séparation
            self.input_layer = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        
        # Classifier simplifié
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),  # 32 -> 16
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 4, n_classes),  # 16 -> 4
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        if self.tfidf_dim and self.stats_dim:
            # Séparation des features
            tfidf_features = x[:, :self.tfidf_dim]
            stats_features = x[:, self.tfidf_dim:]
            
            # Traitement séparé
            tfidf_out = self.tfidf_layers(tfidf_features)
            stats_out = self.stats_layers(stats_features)
            
            # Combinaison
            combined = torch.cat((tfidf_out, stats_out), dim=1)
            x = self.combine_layer(combined)
        else:
            x = self.input_layer(x)
        
        return self.classifier(x)

class EnhancedTextClassifierANN(TextClassifierANN):
    """
    Version améliorée du wrapper avec fonctionnalités avancées.
    
    Améliorations:
    1. Learning rate adaptatif avec ReduceLROnPlateau
    2. Gradient clipping pour stabilité
    3. Optimiseur AdamW avec weight decay
    4. Traitement par batch pour grandes données
    5. Monitoring détaillé des métriques
    
    Args:
        hidden_layer_size (int): Taille de la couche cachée (default: 100)
        input_size (int, optional): Taille totale des features
        tfidf_dim (int, optional): Dimension des features TF-IDF
        stats_dim (int, optional): Dimension des features statistiques
    """

    def __init__(self, hidden_layer_size=100, input_size=None, tfidf_dim=None, stats_dim=None):
        super().__init__(hidden_layer_size)
        self.input_size = input_size
        self.tfidf_dim = tfidf_dim
        self.stats_dim = stats_dim
        self.scheduler = None
    
    def train(self, X, y, X_val=None, y_val=None, progress_bar=None):
        """
        Version améliorée de l'entraînement.

        Améliorations:
        1. Learning rate adaptatif:
           - Réduction sur plateau (factor=0.5)
           - Patience de 5 epochs
           - LR min = 1e-6
        2. Optimisation:
           - AdamW avec weight decay 0.01
           - Gradient clipping (max_norm=1.0)
        3. Monitoring:
           - Suivi du learning rate
           - Statistiques des features
           - Early stopping amélioré (patience=15)

        Args:
            X (array): Données d'entraînement
            y (array): Labels d'entraînement
            X_val (array, optional): Données de validation
            y_val (array, optional): Labels de validation
            progress_bar (tqdm, optional): Barre de progression
        """
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
        optimizer = optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.01)
        
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
        
        batch_size = 2048*4
        n_batches = int(np.ceil(X.shape[0] / batch_size))
        
        best_loss = float('inf')
        best_model = None
        patience = 15  # Early stopping patience
        patience_counter = 0
        epochTotal = 1000
        if progress_bar:
            progress_bar.reset(total=epochTotal)
        
        start_time = time.time()
        
        for epoch in range(epochTotal):  # Augmentation à 2000 epochs
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
            
            if progress_bar and epoch % 1 == 0:  # Mise à jour moins fréquente de la barre de progression
                elapsed = time.time() - start_time
                status = f"Epoch {epoch:4d}/{2000} | Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    status += f" | Val Loss: {val_loss:.4f}"
                status += f" | LR: {optimizer.param_groups[0]['lr']:.2e} | Temps: {elapsed:.1f}s"
                progress_bar.set_description(status)
                progress_bar.update(min(1, progress_bar.total - progress_bar.n))
            
            if patience_counter >= patience:
                if progress_bar:
                    progress_bar.set_description(f"Early stopping après {epoch + 1} epochs")
                break
        
        # Restauration du meilleur modèle
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        if progress_bar:
            progress_bar.update(progress_bar.total - progress_bar.n)
    
    def predict(self, X):
        """
        Prédiction par batch avec normalisation.

        Algorithme:
        1. Découpage des données en batches
        2. Prédiction sur chaque batch
        3. Agrégation des résultats

        Args:
            X (array): Données à prédire

        Returns:
            array: Prédictions pour chaque exemple
        """
        self.model.eval()
        X_tensor = self._to_tensor(X)
        
        predictions = []
        batch_size = 2048*4  # Même batch size que l'entraînement
        
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                batch_X = X_tensor[i:i + batch_size]
                outputs = self.model(batch_X)
                _, batch_pred = torch.max(outputs, 1)
                predictions.extend(batch_pred.cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate(self, X, y_true):
        """
        Évaluation complète du modèle avec diagnostics.

        Métriques calculées:
        - Accuracy: Précision globale
        - Recall pondéré: Moyenne pondérée du recall par classe
        - Matrice de confusion: Distribution des prédictions
        - Diagnostics: Classes uniques et dimensions

        Args:
            X (array): Données à évaluer
            y_true (array): Labels réels

        Returns:
            dict: Dictionnaire contenant toutes les métriques
        """
        y_pred = self.predict(X)
        
        # Vérification et normalisation des dimensions
        y_true = np.array(y_true)
        if len(y_true.shape) > 1:
            y_true = y_true.ravel()
        if len(y_pred.shape) > 1:
            y_pred = y_pred.ravel()
            
        # Vérification des classes
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        print(f"Classes uniques dans y_true: {unique_true}")
        print(f"Classes uniques dans y_pred: {unique_pred}")
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'predictions': y_pred,
            'true_labels': y_true
        }