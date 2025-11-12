"""
Deep Convolutional Neural Network for Structural Damage Identification
PyTorch implementation based on Yu et al. (2018) architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class DCNNDamageNet(nn.Module):
    """
    DCNN for Structural Damage Identification
    Based on Yu et al. (2018) architecture
    """
    
    def __init__(self, input_shape, num_classes, dropout_rate=0.3):
        """
        Initialize DCNN model following Yu et al. architecture
        
        Args:
            input_shape: (n_frequency_components, n_sensors) tuple
            num_classes: Number of damage levels
            dropout_rate: Dropout rate for regularization
        """
        super(DCNNDamageNet, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Input dimensions
        freq_components, n_sensors = input_shape
        
        # First Convolutional Layer - Large kernel to capture periodicities
        kernel_size_1 = min(100, freq_components // 4)  # Adaptive large kernel
        self.conv1 = nn.Conv1d(in_channels=n_sensors, out_channels=128, 
                              kernel_size=kernel_size_1, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second Convolutional Layer - Medium kernel
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, 
                              kernel_size=30, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third Convolutional Layer - Small kernel
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, 
                              kernel_size=10, padding='same')
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Calculate the size after convolutions for fully connected layer
        self._calculate_fc_input_size(freq_components)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.output = nn.Linear(512, num_classes)
        
    def _calculate_fc_input_size(self, input_length):
        """Calculate the input size for the first fully connected layer"""
        # Simulate forward pass through conv layers
        x = torch.randn(1, self.input_shape[1], input_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        self.fc_input_size = x.numel()
        
    def forward(self, x):
        """Forward pass"""
        # Input shape: (batch_size, freq_components, n_sensors)
        # PyTorch Conv1d expects: (batch_size, channels, length)
        # So we need to transpose from (batch, freq, sensors) to (batch, sensors, freq)
        x = x.transpose(1, 2)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # First FC layer
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        
        # Second FC layer
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout_fc2(x)
        
        # Output layer with softmax
        x = self.output(x)
        return F.log_softmax(x, dim=1)


class DCNNTrainer:
    """
    Training and evaluation utilities for DCNN
    """
    
    def __init__(self, model, device=None):
        # Auto-detect best device for Mac M2 Max
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')  # Mac M2 Max GPU
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        
        self.model = model
        self.device = device
        self.model.to(device)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / total
        return val_loss, accuracy
    
    def train_model(self, train_loader, val_loader=None, epochs=100, 
                   learning_rate=0.0035, patience=15):
        """
        Train the model
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of training epochs
            learning_rate: Learning rate (Yu et al. optimal: 0.0035)
            patience: Early stopping patience
        """
        
        # Optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()  # Use with log_softmax
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate_epoch(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, '
                      f'Train Acc: {train_acc:5.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:5.2f}%')
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
                    
            else:
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, '
                      f'Train Acc: {train_acc:5.2f}%')
        
        # Load best model if validation was used
        if val_loader is not None:
            self.model.load_state_dict(torch.load('best_model.pth'))
            print("Best model loaded")
    
    def evaluate_model(self, test_loader, class_names=None):
        """
        Evaluate model on test set
        """
        self.model.eval()
        y_true = []
        y_pred = []
        y_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Convert log probabilities to probabilities
                probs = torch.exp(output)
                pred = output.argmax(dim=1)
                
                y_true.extend(target.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(self.model.num_classes)]
            
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, 
                                     output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probs': y_probs
        }
        
        print(f"\n=== Model Evaluation ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        return results
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'bo-', label='Training Loss')
        if self.history['val_loss']:
            ax1.plot(epochs, self.history['val_loss'], 'ro-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'bo-', label='Training Accuracy')
        if self.history['val_acc']:
            ax2.plot(epochs, self.history['val_acc'], 'ro-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def prepare_data_for_pytorch(processed_signals, csv_data, test_size=0.2, val_size=0.2):
    """
    Prepare data for PyTorch DCNN training
    
    Args:
        processed_signals: Output from SignalPreprocessor.process_all_signals()
        csv_data: DataFrame with damage level information
        test_size: Test set size
        val_size: Validation set size (from training set)
        
    Returns:
        DataLoaders for train, validation, and test sets
    """
    
    X_list = []
    y_list = []
    specimen_info = []
    
    # Create mapping from isolator name to damage level using nivel_damage.csv
    damage_mapping = {}
    if 'ID' in csv_data.columns and 'Ndano' in csv_data.columns:
        # Use nivel_damage.csv format
        for _, row in csv_data.iterrows():
            damage_mapping[row['ID']] = row['Ndano']
    else:
        # Fallback to old format if columns don't match
        for i in range(15):  # Only first 15 unique isolators
            if i < len(csv_data):
                specimen_id = f"A{i+1}"
                damage_mapping[specimen_id] = csv_data.iloc[i]['Nivel_Dano']
                # Also map experiment variations (A1-2, A1-3, etc.) to same damage level  
                damage_mapping[f"{specimen_id}-2"] = csv_data.iloc[i]['Nivel_Dano']
                damage_mapping[f"{specimen_id}-3"] = csv_data.iloc[i]['Nivel_Dano']
    
    print(f"Created damage mapping for {len([k for k in damage_mapping.keys() if '-' not in k])} unique isolators")
    
    # Collect padding information for export
    padding_info = {
        'global_max_freq': 0,
        'shapes_before_padding': {},
        'shapes_after_padding': {}
    }
    
    # Find the global maximum frequency components across all signals
    global_max_freq = 0
    for specimen, sensors in processed_signals.items():
        if specimen in damage_mapping and 'S1' in sensors and 'S2' in sensors:
            s1_shape = sensors['S1']['features'].shape[0]
            s2_shape = sensors['S2']['features'].shape[0]
            global_max_freq = max(global_max_freq, s1_shape, s2_shape)
            # Store original shapes for padding info
            padding_info['shapes_before_padding'][specimen] = {
                'S1': sensors['S1']['features'].shape,
                'S2': sensors['S2']['features'].shape
            }
    
    padding_info['global_max_freq'] = global_max_freq
    print(f"Global max frequency components: {global_max_freq}")
    
    for specimen, sensors in processed_signals.items():
        if specimen in damage_mapping:
            damage_level = damage_mapping[specimen]
            
            # Combine S1 and S2 features if both available
            if 'S1' in sensors and 'S2' in sensors:
                s1_features = sensors['S1']['features']
                s2_features = sensors['S2']['features']
                
                # Pad both signals to global maximum length
                s1_padded = np.pad(s1_features, ((0, global_max_freq - s1_features.shape[0]), (0, 0)), 
                                 mode='constant', constant_values=0)
                s2_padded = np.pad(s2_features, ((0, global_max_freq - s2_features.shape[0]), (0, 0)), 
                                 mode='constant', constant_values=0)
                
                # Stack along sensor dimension: (freq_components, 6) -> 3 axes * 2 sensors
                combined_features = np.concatenate([s1_padded, s2_padded], axis=1)
                
                # Store padded shape for export info
                padding_info['shapes_after_padding'][specimen] = combined_features.shape
                
                X_list.append(combined_features)
                y_list.append(damage_level)
                specimen_info.append({
                    'specimen': specimen,
                    'damage_level': damage_level,
                    'shape': combined_features.shape
                })
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=test_size, random_state=42, 
        stratify=y_encoded
    )
    
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42,
            stratify=y_train.numpy()
        )
    else:
        X_val, y_val = None, None
    
    # Create DataLoaders
    batch_size = 50  # Yu et al. optimal batch size
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )
    
    val_loader = None
    if X_val is not None:
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False
        )
    
    return train_loader, val_loader, test_loader, label_encoder, padding_info


def get_optimal_device():
    """Get the best available device for Mac M2 Max"""
    if torch.backends.mps.is_available():
        return torch.device('mps'), "Mac Chip M(MPS)"
    elif torch.cuda.is_available():
        return torch.device('cuda'), f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    else:
        return torch.device('cpu'), "CPU"


if __name__ == "__main__":
    print("PyTorch DCNN for Structural Damage Identification")
    print("Based on Yu et al. (2018) architecture")
    print("Optimized for Mac M2 Max with MPS acceleration\n")
    
    # Example usage
    input_shape = (1000, 3)  # Example: 1000 frequency components, 3 sensors
    num_classes = 5  # Example: 5 damage levels
    
    # Create model
    model = DCNNDamageNet(input_shape, num_classes)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Get optimal device
    device, device_name = get_optimal_device()
    print(f"Using device: {device} ({device_name})")
    
    # Create trainer with auto-device detection
    trainer = DCNNTrainer(model)  # Will auto-detect MPS
    print(f"Trainer initialized with device: {trainer.device}")