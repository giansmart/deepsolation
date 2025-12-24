"""
Utilidades para análisis de clustering con pares de señales sincronizadas (S2, S1).

Este módulo implementa funciones para:
1. Cargar pares de señales (S2, S1) sincronizadas por espécimen
2. Aplicar FFT a cada par
3. Extraer características espectrales individuales y relacionales

Fecha: 2025-12-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.fft import fft, fftfreq


def standardize_signal_length(
    signal: np.ndarray,
    target_length: int = 60000
) -> np.ndarray:
    """
    Estandariza la longitud de una señal mediante truncamiento o zero-padding.

    Esta función garantiza que todas las señales tengan la misma longitud,
    lo cual es crítico para:
    - Consistencia en la resolución de frecuencia del FFT
    - Comparabilidad entre diferentes especímenes

    Args:
        signal: Señal de entrada con shape (n_samples, n_axes)
        target_length: Longitud objetivo (por defecto 60,000 muestras = 10 min @ 100Hz)

    Returns:
        Señal estandarizada con shape (target_length, n_axes)

    Examples:
        >>> signal = np.random.rand(72000, 3)  # Señal más larga
        >>> standardized = standardize_signal_length(signal, 60000)
        >>> standardized.shape
        (60000, 3)
    """
    current_length = signal.shape[0]

    if current_length > target_length:
        # Truncar si es más largo
        return signal[:target_length, :]
    elif current_length < target_length:
        # Zero-pad si es más corto
        pad_length = target_length - current_length
        # Pad solo en el eje de tiempo (axis=0), no en el eje de componentes
        return np.pad(signal, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
    else:
        # Ya tiene la longitud correcta
        return signal


def load_paired_signals_for_clustering(
    signals_dir: str,
    labels_csv: str,
    base_specimens_only: bool = True,
    target_length: int = 60000,
    verbose: bool = True
) -> List[Dict]:
    """
    Carga pares de señales (S2, S1) sincronizadas para análisis de clustering.

    **Concepto clave**: Cada par (S2, S1) representa UNA medición de un aislador.
    - S2: Sensor en sótano 2 (base del aislador) - Excitación
    - S1: Sensor en sótano 1 (sobre el aislador) - Respuesta
    - El daño se manifiesta en la RELACIÓN entre S2 y S1, no en señales individuales

    Args:
        signals_dir: Directorio raíz con carpetas de especímenes (ej: "data/Signals_Raw/")
        labels_csv: Ruta al archivo CSV con mapeo ID → TIPO → Ndano
        base_specimens_only: Si True, carga solo especímenes base (A1, A2, ... sin -2, -3)
        target_length: Longitud objetivo para estandarización (muestras)
        verbose: Si True, imprime progreso y estadísticas

    Returns:
        Lista de diccionarios con estructura:
        [
            {
                'specimen_id': 'A1',
                'signal_S2': np.array(shape=(target_length, 3)),  # [N_S, E_W, U_D]
                'signal_S1': np.array(shape=(target_length, 3)),
                'nivel_dano': 'N1',
                'tipo': 'B'
            },
            ...
        ]

    Raises:
        FileNotFoundError: Si no se encuentra signals_dir o labels_csv
        ValueError: Si un par de archivos no tiene el mismo número de muestras (después de cargar)

    Examples:
        >>> paired_data = load_paired_signals_for_clustering(
        ...     signals_dir="../data/Signals_Raw/",
        ...     labels_csv="../data/nivel_damage.csv",
        ...     base_specimens_only=True
        ... )
        >>> len(paired_data)  # ~14 especímenes base
        14
    """
    # Verificar que existan los directorios/archivos
    signals_path = Path(signals_dir)
    labels_path = Path(labels_csv)

    if not signals_path.exists():
        raise FileNotFoundError(f"Directorio de señales no encontrado: {signals_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Archivo de etiquetas no encontrado: {labels_csv}")

    # 1. Leer archivo de etiquetas y crear mapeo
    if verbose:
        print("📋 PASO 1: Cargando etiquetas...")

    labels_df = pd.read_csv(labels_path)
    # Crear diccionario: ID → (TIPO, Ndano)
    labels_map = {}
    for _, row in labels_df.iterrows():
        labels_map[row['ID']] = {
            'tipo': row['TIPO'],
            'nivel_dano': row['Ndano']
        }

    if verbose:
        print(f"   ✓ Cargadas etiquetas para {len(labels_map)} especímenes")

    # 2. Listar carpetas en Signals_Raw/
    if verbose:
        print(f"\n📂 PASO 2: Escaneando directorio {signals_dir}...")

    specimen_dirs = [d for d in signals_path.iterdir() if d.is_dir()]

    # 3. Filtrar solo especímenes base si se solicita
    if base_specimens_only:
        # Especímenes base no tienen guiones en el ID (A1, A2, no A1-2)
        specimen_dirs = [d for d in specimen_dirs if '-' not in d.name]
        if verbose:
            print(f"   ✓ Filtrado a {len(specimen_dirs)} especímenes base (sin variantes -2, -3)")
    else:
        if verbose:
            print(f"   ✓ Encontrados {len(specimen_dirs)} especímenes totales")

    # 4. Cargar pares de señales
    if verbose:
        print(f"\n🔄 PASO 3: Cargando pares (S2, S1)...")

    paired_data = []
    skipped_specimens = []

    for specimen_dir in sorted(specimen_dirs):
        specimen_id = specimen_dir.name

        # Verificar que el espécimen tenga etiquetas
        if specimen_id not in labels_map:
            if verbose:
                print(f"   ⚠️  {specimen_id}: Sin etiquetas en CSV, omitido")
            skipped_specimens.append(specimen_id)
            continue

        # Rutas a los archivos de señales
        s1_file = specimen_dir / "completo_S1.txt"
        s2_file = specimen_dir / "completo_S2.txt"

        # Verificar existencia de ambos archivos
        if not s1_file.exists() or not s2_file.exists():
            if verbose:
                print(f"   ⚠️  {specimen_id}: Archivos S1 o S2 faltantes, omitido")
            skipped_specimens.append(specimen_id)
            continue

        try:
            # Cargar señales usando pandas
            # Formato: Fecha Hora N_S E_W U_D (columnas separadas por espacio, skip header)
            df_s2 = pd.read_csv(s2_file, delimiter=' ', skiprows=1)
            df_s1 = pd.read_csv(s1_file, delimiter=' ', skiprows=1)

            # Extraer columnas [2, 3, 4] = [N_S, E_W, U_D]
            signal_s2 = df_s2.iloc[:, [2, 3, 4]].values
            signal_s1 = df_s1.iloc[:, [2, 3, 4]].values

            # Verificar que ambas señales tengan el mismo número de muestras ANTES de estandarizar
            if signal_s2.shape[0] != signal_s1.shape[0]:
                if verbose:
                    print(f"   ⚠️  {specimen_id}: S2 ({signal_s2.shape[0]}) y S1 ({signal_s1.shape[0]}) tienen diferente longitud, omitido")
                skipped_specimens.append(specimen_id)
                continue

            # Estandarizar longitud a target_length
            signal_s2_std = standardize_signal_length(signal_s2, target_length)
            signal_s1_std = standardize_signal_length(signal_s1, target_length)

            # Crear diccionario de par
            pair = {
                'specimen_id': specimen_id,
                'signal_S2': signal_s2_std,
                'signal_S1': signal_s1_std,
                'nivel_dano': labels_map[specimen_id]['nivel_dano'],
                'tipo': labels_map[specimen_id]['tipo'],
                'original_length': signal_s2.shape[0]  # Guardar longitud original para referencia
            }

            paired_data.append(pair)

            if verbose:
                action = "truncada" if signal_s2.shape[0] > target_length else ("padded" if signal_s2.shape[0] < target_length else "sin cambios")
                print(f"   ✓ {specimen_id}: {signal_s2.shape[0]:,} muestras → {target_length:,} ({action}) | {labels_map[specimen_id]['nivel_dano']} | Tipo {labels_map[specimen_id]['tipo']}")

        except Exception as e:
            if verbose:
                print(f"   ❌ {specimen_id}: Error al cargar - {str(e)}")
            skipped_specimens.append(specimen_id)
            continue

    # 5. Resumen final
    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN DE CARGA:")
        print(f"{'='*60}")
        print(f"   • Pares cargados exitosamente: {len(paired_data)}")
        print(f"   • Especímenes omitidos: {len(skipped_specimens)}")
        if skipped_specimens:
            print(f"     Omitidos: {', '.join(skipped_specimens)}")

        # Distribución por nivel de daño
        nivel_counts = {}
        for pair in paired_data:
            nivel = pair['nivel_dano']
            nivel_counts[nivel] = nivel_counts.get(nivel, 0) + 1

        print(f"\n   📈 Distribución por nivel de daño:")
        for nivel in sorted(nivel_counts.keys()):
            count = nivel_counts[nivel]
            pct = (count / len(paired_data)) * 100
            print(f"      {nivel}: {count} pares ({pct:.1f}%)")

        print(f"\n   ⚖️  Ratio de desbalance: {max(nivel_counts.values()) / min(nivel_counts.values()):.2f}:1")
        print(f"{'='*60}\n")

    return paired_data


if __name__ == "__main__":
    # Código de ejemplo/prueba
    print("🧪 Probando carga de pares de señales...")

    # Rutas de ejemplo (ajustar según tu estructura)
    SIGNALS_DIR = "../../data/Signals_Raw/"
    LABELS_CSV = "../../data/nivel_damage.csv"

    try:
        paired_data = load_paired_signals_for_clustering(
            signals_dir=SIGNALS_DIR,
            labels_csv=LABELS_CSV,
            base_specimens_only=True,
            verbose=True
        )

        print(f"\n✅ Prueba exitosa! Cargados {len(paired_data)} pares.")
        print(f"\nEjemplo del primer par:")
        first_pair = paired_data[0]
        print(f"   ID: {first_pair['specimen_id']}")
        print(f"   Nivel de daño: {first_pair['nivel_dano']}")
        print(f"   Shape S2: {first_pair['signal_S2'].shape}")
        print(f"   Shape S1: {first_pair['signal_S1'].shape}")

    except FileNotFoundError as e:
        print(f"⚠️  Archivo no encontrado: {e}")
        print(f"   Ajusta las rutas SIGNALS_DIR y LABELS_CSV en el código")
