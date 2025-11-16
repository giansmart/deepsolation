"""
Deep Neural Network for Structural Damage Identification - Experimento 4
Enfoque con Características Estadísticas Agregadas
===================================================

Implementa una red neuronal para clasificación de daños estructurales
usando características estadísticas agregadas por dispositivo.

DIFERENCIA CLAVE vs Exp1-3:
- Entrada: Vector de características estadísticas (no matrices FFT)
- Arquitectura: Fully Connected (no convolucional)
- Enfoque metodológicamente correcto: Una observación = Un dispositivo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from pathlib import Path

# Configurar matplotlib para no mostrar ventanas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class Exp4DamageNet(nn.Module):
    """
    Deep Neural Network para Experimento 4
    Arquitectura Fully Connected para características estadísticas agregadas
    """
    
    def __init__(self, input_size, num_classes, dropout_rate=0.3):
        """
        Inicializar modelo DNN para características agregadas
        
        Args:
            input_size: Número de características estadísticas de entrada
            num_classes: Número de niveles de daño
            dropout_rate: Tasa de dropout para regularización
        """
        super(Exp4DamageNet, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Arquitectura progresiva para características estadísticas
        # Capa 1: Entrada completa
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Capa 2: Reducción progresiva
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Capa 3: Concentración de características
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Capa 4: Abstracción final
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Capa de salida
        self.output = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """Forward pass del modelo"""
        # Capa 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Capa 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Capa 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Capa 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        # Salida
        x = self.output(x)
        return x
    
    def get_model_info(self):
        """Obtener información del modelo"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'Fully Connected DNN',
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers': [
                f'Input: {self.input_size}',
                f'FC1: {self.input_size} → 512',
                'FC2: 512 → 256',
                'FC3: 256 → 128', 
                'FC4: 128 → 64',
                f'Output: 64 → {self.num_classes}'
            ]
        }

class Exp4Trainer:
    """
    Entrenador especializado para el modelo Exp4
    Adaptado para características estadísticas agregadas
    """
    
    def __init__(self, model, device='auto', class_weights=None):
        """
        Inicializar entrenador
        
        Args:
            model: Instancia del modelo Exp4DamageNet
            device: Dispositivo de cómputo ('auto', 'cpu', 'cuda')
            class_weights: Pesos para balancear clases
        """
        self.model = model
        self.device = get_optimal_device() if device == 'auto' else device
        self.model.to(self.device)
        
        # Configurar función de pérdida con pesos de clase
        if class_weights is not None:
            weights_tensor = torch.FloatTensor(class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Inicializar histories
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
        self.best_val_accuracy = 0
        self.best_model_state = None
        
    def train_epoch(self, train_loader, optimizer):
        """Entrenar una época"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validar una época"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(self, train_loader, val_loader, epochs, learning_rate=0.001, patience=10, verbose=True):
        """
        Entrenar el modelo con early stopping
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: Número máximo de épocas
            learning_rate: Tasa de aprendizaje
            patience: Paciencia para early stopping
            verbose: Mostrar progreso
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Entrenamiento
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # Validación
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Guardar historia
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            
            # Early stopping y best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            if verbose and epoch % 10 == 0:
                print(f"Época {epoch:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping en época {epoch}")
                break
        
        # Cargar el mejor modelo
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.train_history, self.val_history
    
    def predict(self, data_loader):
        """Realizar predicciones"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        
        return np.array(predictions), np.array(true_labels)
    
    def save_model(self, filepath):
        """Guardar modelo entrenado"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_info': self.model.get_model_info(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_accuracy': self.best_val_accuracy
        }, filepath)
    
    def load_model(self, filepath):
        """Cargar modelo entrenado"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        return checkpoint['model_info']

def get_optimal_device():
    """Determinar el mejor dispositivo disponible"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ Usando CPU")
    return device

def create_data_loaders(X, y, batch_size=32, test_size=0.2, random_state=42):
    """
    Crear data loaders para entrenamiento y validación
    
    Args:
        X: Características (numpy array)
        y: Labels (numpy array)
        batch_size: Tamaño de batch
        test_size: Proporción para validación
        random_state: Semilla aleatoria
        
    Returns:
        train_loader, val_loader
    """
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalización de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convertir a tensores
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Crear datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Crear data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler

def plot_training_curves(train_history, val_history, save_path=None):
    """Visualizar curvas de entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Colores consistentes con otros experimentos
    train_color = '#FF9999'  # Color de thesis
    val_color = '#90EE90'    # Color de thesis
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_history['loss'], color=train_color, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_history['loss'], color=val_color, label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=12, weight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_history['accuracy'], color=train_color, label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_history['accuracy'], color=val_color, label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=12, weight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Curvas de entrenamiento guardadas: {save_path}")
    
    plt.close()  # Cerrar en lugar de mostrar
    return fig


if __name__ == "__main__":
    # Ejemplo de uso del modelo Exp4
    print("=== Experimento 4: Modelo para Características Agregadas ===")
    
    # Crear modelo de ejemplo
    input_size = 303  # Número de características del preprocesamiento
    num_classes = 3   # N1, N2, N3
    
    model = Exp4DamageNet(input_size, num_classes)
    print(f"✓ Modelo creado con {input_size} características de entrada")
    
    # Información del modelo
    info = model.get_model_info()
    print(f"✓ Arquitectura: {info['architecture']}")
    print(f"✓ Parámetros totales: {info['total_parameters']:,}")
    print(f"✓ Parámetros entrenables: {info['trainable_parameters']:,}")