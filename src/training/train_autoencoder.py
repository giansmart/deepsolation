"""
Script de entrenamiento para el autoencoder (ETAPA 1).

Pipeline completo:
1. Carga de datos sincronizados
2. Split train/val (85/15)
3. Entrenamiento del autoencoder con early stopping
4. Checkpointing del mejor modelo
5. Visualización de resultados

Autor: Giancarlo Poémape Lozano
Fecha: 2026-02-07
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Imports del proyecto
from src.loaders import create_dataloader, SynchronizedSignalsDataset
from src.models import create_autoencoder
from src.utils.training_utils import save_experiment_log

# Configuración de seaborn para plots bonitos
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EarlyStopping:
    """
    Early stopping para detener entrenamiento cuando validation loss no mejora.

    Args:
        patience: Número de epochs sin mejora antes de detener
        min_delta: Mejora mínima considerada como progreso
        verbose: Si True, imprime mensajes
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Actualiza contador de early stopping.

        Args:
            val_loss: Validation loss del epoch actual

        Returns:
            True si debe detener el entrenamiento, False en caso contrario
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"   EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> float:
    """
    Entrena el modelo por un epoch.

    Args:
        model: Autoencoder
        dataloader: DataLoader de entrenamiento
        criterion: Función de pérdida (MSE)
        optimizer: Optimizador (Adam)
        device: Dispositivo ('cpu', 'cuda', 'mps')

    Returns:
        Promedio de pérdida en el epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_signals, _ in dataloader:
        # Mover datos al device
        batch_signals = batch_signals.to(device)

        # Forward pass
        reconstruction, _ = model(batch_signals)

        # Calcular pérdida
        loss = criterion(reconstruction, batch_signals)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Acumular pérdida
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Valida el modelo en el conjunto de validación.

    Args:
        model: Autoencoder
        dataloader: DataLoader de validación
        criterion: Función de pérdida (MSE)
        device: Dispositivo

    Returns:
        Promedio de pérdida de validación
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_signals, _ in dataloader:
            batch_signals = batch_signals.to(device)

            # Forward pass
            reconstruction, _ = model(batch_signals)

            # Calcular pérdida
            loss = criterion(reconstruction, batch_signals)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    filepath: Path
) -> None:
    """
    Guarda checkpoint del modelo.

    Args:
        model: Modelo a guardar
        optimizer: Optimizador
        epoch: Epoch actual
        train_loss: Pérdida de entrenamiento
        val_loss: Pérdida de validación
        filepath: Ruta donde guardar el checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, filepath)


def plot_training_history(history: Dict, output_dir: Path) -> None:
    """
    Genera plots de la historia de entrenamiento con seaborn.

    Args:
        history: Diccionario con 'train_loss' y 'val_loss' por epoch
        output_dir: Directorio donde guardar los plots
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot de pérdida train vs val
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(
        x=list(epochs),
        y=history['train_loss'],
        label='Train Loss',
        marker='o',
        linewidth=2,
        ax=ax
    )
    sns.lineplot(
        x=list(epochs),
        y=history['val_loss'],
        label='Validation Loss',
        marker='s',
        linewidth=2,
        ax=ax
    )

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE Loss', fontsize=12, fontweight='bold')
    ax.set_title('Autoencoder Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Plot guardado: {output_dir / 'training_history.png'}")


def visualize_reconstructions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    output_dir: Path,
    n_samples: int = 3
) -> None:
    """
    Visualiza reconstrucciones del autoencoder para validación cualitativa.

    Args:
        model: Autoencoder entrenado
        dataloader: DataLoader de validación
        device: Dispositivo
        output_dir: Directorio donde guardar los plots
        n_samples: Número de muestras a visualizar
    """
    model.eval()

    # Obtener un batch
    batch_signals, _ = next(iter(dataloader))
    batch_signals = batch_signals.to(device)

    with torch.no_grad():
        reconstructions, _ = model(batch_signals)

    # Mover a CPU para plotting
    batch_signals = batch_signals.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    # Visualizar primeras n_samples
    for i in range(min(n_samples, batch_signals.shape[0])):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Reconstrucción del Autoencoder - Muestra {i+1}', fontsize=14, fontweight='bold')

        channel_names = ['S2_NS', 'S2_EW', 'S2_UD', 'S1_NS', 'S1_EW', 'S1_UD']

        for ch in range(6):
            row = ch // 3
            col = ch % 3

            original = batch_signals[i, ch, :]
            reconstructed = reconstructions[i, ch, :]

            # Plot original vs reconstruida (primeras 1000 muestras para claridad)
            axes[row, col].plot(original[:1000], label='Original', alpha=0.7, linewidth=1)
            axes[row, col].plot(reconstructed[:1000], label='Reconstrucción', alpha=0.7, linewidth=1)
            axes[row, col].set_title(channel_names[ch], fontweight='bold')
            axes[row, col].set_xlabel('Muestras')
            axes[row, col].set_ylabel('Amplitud')
            axes[row, col].legend(fontsize=8)
            axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'reconstruction_sample_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"   ✓ Reconstrucciones guardadas: {n_samples} muestras")


def train_autoencoder(
    sync_dir: str,
    labels_csv: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    val_split: float = 0.15,
    latent_dim: int = 512,
    normalize: bool = False,
    window_size: int = None,
    overlap: float = 0.5,
    device: str = 'auto'
) -> Dict:
    """
    Pipeline completo de entrenamiento del autoencoder.

    Args:
        sync_dir: Directorio de señales sincronizadas
        labels_csv: CSV con etiquetas
        output_dir: Directorio de salida para checkpoints y logs
        epochs: Número máximo de epochs
        batch_size: Tamaño de batch
        lr: Learning rate inicial
        weight_decay: L2 regularization
        patience: Paciencia para early stopping
        val_split: Proporción de datos para validación
        window_size: Tamaño de ventana en muestras (None = señal completa)
        overlap: Fracción de overlap entre ventanas (0.0 a 0.9)
        device: Dispositivo ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        Diccionario con historia de entrenamiento
    """
    # Crear directorios de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = output_path / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    # Configurar device
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print("=" * 70)
    print("ENTRENAMIENTO DEL AUTOENCODER (ETAPA 1)")
    print("=" * 70)
    print(f"\n📍 Configuración:")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Weight decay: {weight_decay}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Validation split: {val_split * 100:.0f}%")
    print(f"   Latent dim: {latent_dim}")
    print(f"   Normalize: {normalize}")
    print(f"   Window size: {window_size or 'full signal'}")
    print(f"   Overlap: {overlap}\n")

    # 1. Cargar dataset completo
    print("1. Cargando datos...")
    full_dataset = SynchronizedSignalsDataset(
        sync_dir=sync_dir,
        labels_csv=labels_csv,
        return_metadata=False,
        normalize=normalize,
        window_size=window_size,
        overlap=overlap
    )
    total_samples = len(full_dataset)
    target_length = full_dataset.signal_length

    print(f"   ✓ Mediciones: {full_dataset.n_measurements}")
    print(f"   ✓ Aisladores únicos: {full_dataset.n_unique_isolators}")
    if window_size is not None:
        print(f"   ✓ Ventanas por señal: {full_dataset.windows_per_signal}")
        print(f"   ✓ Total muestras (ventanas): {total_samples} ({full_dataset.windows_per_signal}×)")
    else:
        print(f"   ✓ Total muestras: {total_samples}")
    print(f"   ✓ Signal length: {target_length}")

    # 2. Split train/val por medición (evita data leakage entre ventanas)
    n_meas = full_dataset.n_measurements
    meas_indices = list(range(n_meas))
    rng = np.random.RandomState(42)
    rng.shuffle(meas_indices)

    n_val_meas = max(1, int(val_split * n_meas))
    val_meas_set = set(meas_indices[:n_val_meas])
    train_meas_set = set(meas_indices[n_val_meas:])

    # Mapear mediciones a índices de ventanas (o de muestras sin windowing)
    wps = full_dataset.windows_per_signal
    train_indices = []
    val_indices = []
    for meas_idx in range(n_meas):
        window_idxs = list(range(meas_idx * wps, (meas_idx + 1) * wps))
        if meas_idx in val_meas_set:
            val_indices.extend(window_idxs)
        else:
            train_indices.extend(window_idxs)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    train_size = len(train_indices)
    val_size = len(val_indices)

    print(f"   ✓ Train: {len(train_meas_set)} mediciones → {train_size} muestras")
    print(f"   ✓ Val:   {len(val_meas_set)} mediciones → {val_size} muestras")

    # 3. Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"   ✓ Train batches: {len(train_loader)}")
    print(f"   ✓ Val batches:   {len(val_loader)}\n")

    # 4. Crear modelo
    print("2. Creando autoencoder...")
    model = create_autoencoder(latent_dim=latent_dim, target_length=target_length, device=device)
    n_params = model.count_parameters()
    print(f"   ✓ Autoencoder creado: {n_params:,} parámetros\n")

    # 5. Configurar optimización
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # 6. Training loop
    print("3. Iniciando entrenamiento...")
    print("=" * 70)

    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    best_val_epoch = 0
    start_time = datetime.now()

    for epoch in range(1, epochs + 1):
        # Entrenar
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validar
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Guardar historia
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Log de progreso
        print(f"Epoch [{epoch}/{epochs}]  "
              f"Train Loss: {train_loss:.6f}  "
              f"Val Loss: {val_loss:.6f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                checkpoints_dir / 'autoencoder_best.pth'
            )
            print(f"   ✓ Mejor modelo guardado (val_loss: {val_loss:.6f})")

        # Guardar último modelo
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss,
            checkpoints_dir / 'autoencoder_last.pth'
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping
        if early_stopping(val_loss):
            print(f"\n⚠️  Early stopping activado en epoch {epoch}")
            break

        print()

    # Tiempo total
    elapsed_time = datetime.now() - start_time
    print("=" * 70)
    print(f"✅ Entrenamiento completado en {elapsed_time}")
    print(f"   Mejor val_loss: {best_val_loss:.6f}")
    print("=" * 70 + "\n")

    # 7. Guardar historia de entrenamiento
    print("4. Guardando resultados...")
    history_file = output_path / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   ✓ Historia guardada: {history_file}")

    # 8. Generar plots
    print("\n5. Generando visualizaciones...")
    plot_training_history(history, output_path)

    # 9. Visualizar reconstrucciones
    print("\n6. Generando reconstrucciones de validación...")
    model.load_state_dict(
        torch.load(checkpoints_dir / 'autoencoder_best.pth')['model_state_dict']
    )
    visualize_reconstructions(model, val_loader, device, output_path, n_samples=3)

    # 10. Guardar experiment log
    print("\n7. Guardando experiment log...")
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]

    save_experiment_log(
        output_path=output_path,
        config={
            'epochs_max': epochs,
            'epochs_ran': len(history['train_loss']),
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'patience': patience,
            'val_split': val_split,
            'latent_dim': model.latent_dim,
            'device': device,
            'normalize': normalize,
            'window_size': window_size,
            'overlap': overlap,
            'target_length': target_length
        },
        dataset_info={
            'total_samples': total_samples,
            'n_measurements': full_dataset.n_measurements,
            'n_unique_isolators': full_dataset.n_unique_isolators,
            'windows_per_signal': full_dataset.windows_per_signal,
            'train_samples': train_size,
            'val_samples': val_size,
            'train_measurements': len(train_meas_set),
            'val_measurements': len(val_meas_set),
            'split_strategy': 'by_measurement'
        },
        metrics={
            'best_val_loss': best_val_loss,
            'best_val_epoch': best_val_epoch,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'gap_ratio': round(final_val_loss / final_train_loss, 2) if final_train_loss > 0 else float('inf'),
            'total_params': n_params,
            'training_time': str(elapsed_time)
        }
    )

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO")
    print("=" * 70)
    print(f"\n📁 Resultados guardados en: {output_path}")
    print(f"   - Checkpoints: checkpoints/")
    print(f"   - Plots: *.png")
    print(f"   - History: training_history.json")
    print(f"   - Log: experiment_log.json\n")

    return history


def main():
    """Función principal con argumentos CLI."""
    parser = argparse.ArgumentParser(
        description='Entrena el autoencoder para aprendizaje no supervisado (ETAPA 1)'
    )

    parser.add_argument(
        '--sync-dir',
        type=str,
        default='data/processed/synchronized/',
        help='Directorio de señales sincronizadas'
    )
    parser.add_argument(
        '--labels-csv',
        type=str,
        default='data/nivel_damage.csv',
        help='CSV con etiquetas'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/',
        help='Directorio de salida para checkpoints y logs'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Número máximo de epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamaño de batch'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate inicial'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='L2 regularization'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Paciencia para early stopping'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Proporción de datos para validación'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=512,
        help='Dimensión del espacio latente (default: 512)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalizar señales por canal (z-score)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=None,
        help='Tamaño de ventana en muestras (None = señal completa). Ej: 10000'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help='Fracción de overlap entre ventanas (default: 0.5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Dispositivo de entrenamiento'
    )

    args = parser.parse_args()

    try:
        train_autoencoder(
            sync_dir=args.sync_dir,
            labels_csv=args.labels_csv,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            val_split=args.val_split,
            latent_dim=args.latent_dim,
            normalize=args.normalize,
            window_size=args.window_size,
            overlap=args.overlap,
            device=args.device
        )
        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
