"""
Data loader para el autoencoder (ETAPA 1).

Carga señales sincronizadas S2-S1 desde data/processed/synchronized/
para entrenamiento no supervisado del autoencoder.

Soporta segmentación con ventanas deslizantes (overlap) para
data augmentation, inspirado en Feng et al. (2025).

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


def get_valid_measurements(
    sync_dir: str,
    labels_csv: str
) -> List[Tuple[str, str, str]]:
    """
    Obtiene lista de mediciones válidas (con señales sincronizadas).

    Cada medición es una combinación única de (edificio, pasada, specimen_id).
    Un mismo aislador físico puede tener múltiples mediciones (pasadas).

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

    valid_measurements = []

    for _, row in labels_df.iterrows():
        edificio = row['edificio']
        pasada = row['pasada']
        aislador_id = row['specimen_id']

        # Verificar que existan archivos sincronizados
        measurement_dir = sync_path / edificio / pasada / aislador_id
        s2_file = measurement_dir / 'S2_synchronized.npy'
        s1_file = measurement_dir / 'S1_synchronized.npy'
        metadata_file = measurement_dir / 'metadata.json'

        if s2_file.exists() and s1_file.exists() and metadata_file.exists():
            valid_measurements.append((edificio, pasada, aislador_id))

    return valid_measurements


class SynchronizedSignalsDataset(Dataset):
    """
    Dataset de señales sincronizadas para entrenamiento del autoencoder.

    Carga pares S2-S1 sincronizados y los concatena en un único array.
    Soporta segmentación con ventanas deslizantes para data augmentation.

    Args:
        sync_dir: Directorio raíz de señales sincronizadas
        labels_csv: CSV con etiquetas
        transform: Transformación opcional (para augmentación)
        return_metadata: Si True, retorna también metadata del aislador
        normalize: Si True, aplica z-score por canal (media=0, std=1)
        window_size: Tamaño de ventana en muestras. None = señal completa
        overlap: Fracción de overlap entre ventanas (0.0 a 0.9). Default: 0.5
        representation: Representación de la señal. 'raw' = forma de onda,
            'fft' = magnitud del espectro FFT con log1p. Default: 'raw'

    Shape del output:
        signal: (6, signal_length) donde signal_length depende de representation:
            - 'raw': window_size o señal completa
            - 'fft': window_size // 2 + 1 (bins de frecuencia)
        metadata: Dict con información del aislador (opcional)
    """

    def __init__(
        self,
        sync_dir: str,
        labels_csv: str,
        transform: Optional[callable] = None,
        return_metadata: bool = False,
        normalize: bool = False,
        window_size: Optional[int] = None,
        overlap: float = 0.5,
        representation: str = 'raw'
    ):
        if representation not in ('raw', 'fft'):
            raise ValueError(f"representation debe ser 'raw' o 'fft', recibido: '{representation}'")

        self.sync_dir = Path(sync_dir)
        self.transform = transform
        self.return_metadata = return_metadata
        self.normalize = normalize
        self.representation = representation
        self._window_size = window_size
        self.overlap = overlap

        # Obtener lista de mediciones válidas (edificio, pasada, specimen_id)
        self.measurements = get_valid_measurements(sync_dir, labels_csv)

        if len(self.measurements) == 0:
            raise ValueError(
                f"No se encontraron señales sincronizadas en {sync_dir}. "
                "Ejecuta run_synchronization_pipeline.py primero."
            )

        # Detectar largo de señal desde el primer archivo
        first_dir = self.sync_dir / self.measurements[0][0] / self.measurements[0][1] / self.measurements[0][2]
        sample = np.load(first_dir / 'S2_synchronized.npy')
        self._raw_signal_length = sample.shape[0]

        # Pre-computar índices de ventanas si se usa windowing
        self.window_indices: Optional[List[Tuple[int, int]]] = None
        if self._window_size is not None:
            self._build_window_indices()

    def _build_window_indices(self) -> None:
        """
        Pre-computa pares (isolator_idx, start_sample) para todas las ventanas.

        Genera ventanas deslizantes con el overlap configurado sobre cada señal.
        """
        stride = int(self._window_size * (1 - self.overlap))
        self.window_indices = []

        for meas_idx in range(len(self.measurements)):
            start = 0
            while start + self._window_size <= self._raw_signal_length:
                self.window_indices.append((meas_idx, start))
                start += stride

    @property
    def signal_length(self) -> int:
        """Largo de señal que produce este dataset, considerando representación."""
        raw_len = self._window_size if self._window_size is not None else self._raw_signal_length
        if self.representation == 'fft':
            return raw_len // 2 + 1
        return raw_len

    @property
    def n_measurements(self) -> int:
        """Número de mediciones (señales originales) en el dataset."""
        return len(self.measurements)

    @property
    def n_unique_isolators(self) -> int:
        """Número de aisladores físicos únicos (por edificio + specimen_id)."""
        unique = {(ed, sid) for ed, _, sid in self.measurements}
        return len(unique)

    @property
    def windows_per_signal(self) -> int:
        """Número de ventanas extraídas por señal (1 si no hay windowing)."""
        if self.window_indices is None:
            return 1
        return len(self.window_indices) // len(self.measurements)

    def __len__(self) -> int:
        """Retorna número total de muestras (ventanas o señales completas)."""
        if self.window_indices is not None:
            return len(self.window_indices)
        return len(self.measurements)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Carga una ventana o señal completa (par S2-S1 sincronizado).

        Args:
            idx: Índice de la ventana (o del aislador si no hay windowing)

        Returns:
            signal: Tensor (6, signal_length) con señales concatenadas
            metadata: Dict con info del aislador (si return_metadata=True)
        """
        # Determinar qué medición y qué segmento cargar
        if self.window_indices is not None:
            meas_idx, start = self.window_indices[idx]
        else:
            meas_idx = idx
            start = None

        edificio, pasada, aislador_id = self.measurements[meas_idx]
        measurement_dir = self.sync_dir / edificio / pasada / aislador_id

        # Cargar señales sincronizadas
        S2 = np.load(measurement_dir / 'S2_synchronized.npy')  # (signal_length, 3)
        S1 = np.load(measurement_dir / 'S1_synchronized.npy')  # (signal_length, 3)

        # Concatenar: [S2_NS, S2_EW, S2_UD, S1_NS, S1_EW, S1_UD]
        signal = np.concatenate([S2, S1], axis=1)  # (signal_length, 6)

        # Extraer ventana si hay windowing
        if start is not None:
            signal = signal[start:start + self._window_size, :]

        # Transformar a dominio frecuencial si se usa FFT
        if self.representation == 'fft':
            # rfft por canal: (N, 6) → (N//2+1, 6) complejo → magnitud + log1p
            signal = np.log1p(np.abs(np.fft.rfft(signal, axis=0)))

        # Normalizar por canal: z-score (media=0, std=1)
        if self.normalize:
            mean = signal.mean(axis=0, keepdims=True)   # (1, 6)
            std = signal.std(axis=0, keepdims=True)      # (1, 6)
            std[std == 0] = 1.0  # Evitar división por cero
            signal = (signal - mean) / std

        # Transpose para Conv1D: (timesteps, channels) → (channels, timesteps)
        signal = signal.T  # (6, signal_length)

        # Aplicar transformación si existe
        if self.transform is not None:
            signal = self.transform(signal)

        # Convertir a tensor PyTorch (float32)
        signal_tensor = torch.from_numpy(signal.astype(np.float32))

        # Cargar metadata si se solicita
        if self.return_metadata:
            with open(measurement_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            metadata.update({
                'edificio': edificio,
                'pasada': pasada,
                'specimen_id': aislador_id
            })

            if start is not None:
                metadata['window_start'] = start
                metadata['window_size'] = self._window_size

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
        # --- Test 1: Sin windowing (comportamiento original) ---
        print("\n1. Dataset SIN windowing (señal completa)...")
        dataset_full = SynchronizedSignalsDataset(
            sync_dir=sync_dir,
            labels_csv=labels_csv,
            return_metadata=True
        )
        print(f"   ✓ Muestras: {len(dataset_full)}")
        print(f"   ✓ Signal length: {dataset_full.signal_length}")
        print(f"   ✓ Windows per signal: {dataset_full.windows_per_signal}")

        signal, metadata = dataset_full[0]
        print(f"   ✓ Shape: {signal.shape}")
        print(f"   ✓ Aislador: {metadata['edificio']}/{metadata['pasada']}/{metadata['specimen_id']}")

        # --- Test 2: Con windowing ---
        print("\n2. Dataset CON windowing (window_size=10000, overlap=0.5)...")
        dataset_windowed = SynchronizedSignalsDataset(
            sync_dir=sync_dir,
            labels_csv=labels_csv,
            window_size=10000,
            overlap=0.5,
            return_metadata=True
        )
        print(f"   ✓ Mediciones: {dataset_windowed.n_measurements}")
        print(f"   ✓ Aisladores únicos: {dataset_windowed.n_unique_isolators}")
        print(f"   ✓ Ventanas por señal: {dataset_windowed.windows_per_signal}")
        print(f"   ✓ Total muestras: {len(dataset_windowed)}")
        print(f"   ✓ Signal length: {dataset_windowed.signal_length}")
        print(f"   ✓ Multiplicación: {len(dataset_windowed) / dataset_windowed.n_measurements:.1f}×")

        signal_w, metadata_w = dataset_windowed[0]
        print(f"   ✓ Shape ventana: {signal_w.shape}")

        # Verificar última ventana de la primera señal
        last_window_idx = dataset_windowed.windows_per_signal - 1
        signal_last, meta_last = dataset_windowed[last_window_idx]
        print(f"   ✓ Última ventana: start={meta_last['window_start']}, shape={signal_last.shape}")

        # --- Test 3: DataLoader con windowing ---
        print("\n3. DataLoader con windowing...")
        windowed_loader = torch.utils.data.DataLoader(
            SynchronizedSignalsDataset(
                sync_dir=sync_dir,
                labels_csv=labels_csv,
                window_size=10000,
                overlap=0.5
            ),
            batch_size=32,
            shuffle=True
        )
        batch, _ = next(iter(windowed_loader))
        print(f"   ✓ Batch shape: {batch.shape}")
        print(f"   ✓ Batches totales: {len(windowed_loader)}")

        print("\n" + "=" * 70)
        print("✅ TEST EXITOSO: Data loader con windowing funcionando")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
