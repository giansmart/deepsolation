"""
Data loader para el autoencoder (ETAPA 1).

Carga señales sincronizadas S2-S1 desde data/processed/synchronized/
para entrenamiento no supervisado del autoencoder.

Autor: Giancarlo Poémape Lozano
Fecha: 2026-02-07
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_valid_isolators(
    sync_dir: str,
    labels_csv: str
) -> List[Tuple[str, str, str]]:
    """
    Obtiene lista de aisladores válidos (con señales sincronizadas).

    Args:
        sync_dir: Directorio de señales sincronizadas
        labels_csv: CSV con etiquetas (edificio, pasada, specimen_id, ...)

    Returns:
        Lista de tuplas (edificio, pasada, specimen_id) que tienen
        archivos S2_synchronized.npy y S1_synchronized.npy válidos
    """
    sync_path = Path(sync_dir)
    labels_df = pd.read_csv(labels_csv)
    labels_df.columns = ['edificio', 'pasada', 'specimen_id', 'tipo', 'nivel_dano']

    valid_isolators = []

    for _, row in labels_df.iterrows():
        edificio = row['edificio']
        pasada = row['pasada']
        aislador_id = row['specimen_id']

        # Verificar que existan archivos sincronizados
        isolator_dir = sync_path / edificio / pasada / aislador_id
        s2_file = isolator_dir / 'S2_synchronized.npy'
        s1_file = isolator_dir / 'S1_synchronized.npy'
        metadata_file = isolator_dir / 'metadata.json'

        if s2_file.exists() and s1_file.exists() and metadata_file.exists():
            valid_isolators.append((edificio, pasada, aislador_id))

    return valid_isolators


class SynchronizedSignalsDataset(Dataset):
    """
    Dataset de señales sincronizadas para entrenamiento del autoencoder.

    Carga pares S2-S1 sincronizados y los concatena en un único array (60000, 6).

    Args:
        sync_dir: Directorio raíz de señales sincronizadas
        labels_csv: CSV con etiquetas
        transform: Transformación opcional (para augmentación)
        return_metadata: Si True, retorna también metadata del aislador

    Shape del output:
        signal: (6, 60000) → [S2_NS, S2_EW, S2_UD, S1_NS, S1_EW, S1_UD]
        metadata: Dict con información del aislador (opcional)
    """

    def __init__(
        self,
        sync_dir: str,
        labels_csv: str,
        transform: Optional[callable] = None,
        return_metadata: bool = False
    ):
        self.sync_dir = Path(sync_dir)
        self.transform = transform
        self.return_metadata = return_metadata

        # Obtener lista de aisladores válidos
        self.isolators = get_valid_isolators(sync_dir, labels_csv)

        if len(self.isolators) == 0:
            raise ValueError(
                f"No se encontraron señales sincronizadas en {sync_dir}. "
                "Ejecuta run_synchronization_pipeline.py primero."
            )

    def __len__(self) -> int:
        """Retorna número de mediciones válidas."""
        return len(self.isolators)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Carga una medición (par S2-S1 sincronizado).

        Args:
            idx: Índice del aislador en la lista

        Returns:
            signal: Tensor (6, 60000) con señales concatenadas
            metadata: Dict con info del aislador (si return_metadata=True)
        """
        edificio, pasada, aislador_id = self.isolators[idx]
        isolator_dir = self.sync_dir / edificio / pasada / aislador_id

        # Cargar señales sincronizadas
        S2 = np.load(isolator_dir / 'S2_synchronized.npy')  # (60000, 3)
        S1 = np.load(isolator_dir / 'S1_synchronized.npy')  # (60000, 3)

        # Concatenar: [S2_NS, S2_EW, S2_UD, S1_NS, S1_EW, S1_UD]
        signal = np.concatenate([S2, S1], axis=1)  # (60000, 6)

        # Transpose para Conv1D: (timesteps, channels) → (channels, timesteps)
        signal = signal.T  # (6, 60000)

        # Aplicar transformación si existe
        if self.transform is not None:
            signal = self.transform(signal)

        # Convertir a tensor PyTorch (float32)
        signal_tensor = torch.from_numpy(signal.astype(np.float32))

        # Cargar metadata si se solicita
        if self.return_metadata:
            with open(isolator_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            metadata.update({
                'edificio': edificio,
                'pasada': pasada,
                'specimen_id': aislador_id
            })

            return signal_tensor, metadata

        return signal_tensor, {}


def create_dataloader(
    sync_dir: str,
    labels_csv: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[callable] = None
) -> torch.utils.data.DataLoader:
    """
    Crea un DataLoader de PyTorch para el autoencoder.

    Args:
        sync_dir: Directorio de señales sincronizadas
        labels_csv: CSV con etiquetas
        batch_size: Tamaño de batch
        shuffle: Si True, mezcla datos en cada epoch
        num_workers: Número de workers para carga paralela
        transform: Transformación opcional (augmentación)

    Returns:
        DataLoader listo para entrenamiento

    Example:
        >>> dataloader = create_dataloader(
        ...     sync_dir='data/processed/synchronized/',
        ...     labels_csv='data/nivel_damage.csv',
        ...     batch_size=16,
        ...     shuffle=True
        ... )
        >>> for batch, _ in dataloader:
        ...     print(batch.shape)  # (16, 6, 60000)
    """
    dataset = SynchronizedSignalsDataset(
        sync_dir=sync_dir,
        labels_csv=labels_csv,
        transform=transform,
        return_metadata=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Acelera transferencia CPU->GPU en Mac M2
    )

    return dataloader


if __name__ == '__main__':
    """
    Script de prueba para validar el data loader.

    Uso:
        python -m deepsolation.src.loaders.autoencoder_loader
    """
    print("=" * 70)
    print("TEST: Data Loader para Autoencoder")
    print("=" * 70)

    # Parámetros
    sync_dir = 'deepsolation/data/processed/synchronized/'
    labels_csv = 'deepsolation/data/nivel_damage.csv'

    try:
        # Crear dataset
        print("\n1. Creando dataset...")
        dataset = SynchronizedSignalsDataset(
            sync_dir=sync_dir,
            labels_csv=labels_csv,
            return_metadata=True
        )
        print(f"   ✓ Dataset creado: {len(dataset)} mediciones válidas")

        # Verificar primera muestra
        print("\n2. Verificando primera muestra...")
        signal, metadata = dataset[0]
        print(f"   ✓ Shape de señal: {signal.shape}")
        print(f"   ✓ Tipo: {signal.dtype}")
        print(f"   ✓ Aislador: {metadata['edificio']}/{metadata['pasada']}/{metadata['specimen_id']}")
        print(f"   ✓ Offset aplicado: {metadata['offset_applied']}s")

        # Crear dataloader
        print("\n3. Creando DataLoader...")
        dataloader = create_dataloader(
            sync_dir=sync_dir,
            labels_csv=labels_csv,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )
        print(f"   ✓ DataLoader creado: {len(dataloader)} batches de tamaño 8")

        # Probar un batch
        print("\n4. Probando un batch...")
        batch, _ = next(iter(dataloader))
        print(f"   ✓ Shape del batch: {batch.shape}")
        print(f"   ✓ Min valor: {batch.min():.4f}")
        print(f"   ✓ Max valor: {batch.max():.4f}")
        print(f"   ✓ Mean valor: {batch.mean():.4f}")

        print("\n" + "=" * 70)
        print("✅ TEST EXITOSO: Data loader funcionando correctamente")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
