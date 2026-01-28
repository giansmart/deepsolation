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
from typing import List, Dict, Tuple, Optional, Any
from scipy.fft import fft, fftfreq
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)


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


def load_paired_signals(
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
                print(f"   ⚠️  {specimen_id}: No encontrado en etiquetas, asignando 'Sin_etiqueta'")
            nivel_dano = 'Sin_etiqueta'
            tipo = 'Unknown'
        else:
            nivel_dano = labels_map[specimen_id]['nivel_dano']
            tipo = labels_map[specimen_id]['tipo']

        # Buscar archivos que empiecen con "completo_S1" y "completo_S2"
        s1_files = list(specimen_dir.glob("completo_S1*.txt"))
        s2_files = list(specimen_dir.glob("completo_S2*.txt"))

        # Verificar existencia de ambos archivos
        if not s1_files or not s2_files:
            if verbose:
                print(f"   ⚠️  {specimen_id}: Archivos S1 o S2 faltantes, omitido")
            skipped_specimens.append(specimen_id)
            continue

        # Tomar el primer archivo encontrado
        s1_file = s1_files[0]
        s2_file = s2_files[0]

        try:
            # Cargar señales usando pandas
            # Formato: Fecha Hora N_S E_W U_D (separado por espacios)
            df_s2 = pd.read_csv(s2_file, sep=r'\s+', skiprows=1)
            df_s1 = pd.read_csv(s1_file, sep=r'\s+', skiprows=1)

            # Extraer columnas [2, 3, 4] = [N_S, E_W, U_D]
            signal_s2 = df_s2.iloc[:, [2, 3, 4]].values
            signal_s1 = df_s1.iloc[:, [2, 3, 4]].values

            # Estandarizar longitud a target_length (maneja diferentes longitudes automáticamente)
            signal_s2_std = standardize_signal_length(signal_s2, target_length)
            signal_s1_std = standardize_signal_length(signal_s1, target_length)

            # Crear diccionario de par
            pair = {
                'specimen_id': specimen_id,
                'signal_S2': signal_s2_std,
                'signal_S1': signal_s1_std,
                'nivel_dano': nivel_dano,
                'tipo': tipo,
                'original_length': signal_s2.shape[0]  # Guardar longitud original para referencia
            }

            paired_data.append(pair)

            if verbose:
                action = "truncada" if signal_s2.shape[0] > target_length else ("padded" if signal_s2.shape[0] < target_length else "sin cambios")
                print(f"   ✓ {specimen_id}: {signal_s2.shape[0]:,} muestras → {target_length:,} ({action}) | {nivel_dano} | Tipo {tipo}")

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


def apply_fft_to_pairs(
    paired_data: List[Dict],
    sampling_rate: int = 100
) -> List[Dict]:
    """
    Aplica FFT a cada par (S2, S1) para análisis en el dominio de frecuencia.

    Esta función transforma las señales del dominio temporal al dominio de frecuencia
    usando la Fast Fourier Transform (FFT). El análisis espectral es crucial porque:
    - El daño estructural se manifiesta como cambios en frecuencias características
    - La relación S1/S2 en frecuencia representa la función de transferencia H(f)
    - Permite identificar resonancias y bandas de energía específicas

    Args:
        paired_data: Lista de diccionarios con pares (S2, S1) del paso anterior
                     Cada dict debe contener 'signal_S2' y 'signal_S1'
        sampling_rate: Frecuencia de muestreo en Hz (default: 100 Hz)

    Returns:
        Lista de diccionarios enriquecida con FFT:
        [
            {
                'specimen_id': 'A1',
                'signal_S2': np.array(60000, 3),
                'signal_S1': np.array(60000, 3),
                'fft_S2': {
                    'freqs': np.array(30001,),         # Frecuencias [0, Nyquist]
                    'magnitudes': np.array(30001, 3),  # |FFT| por eje
                    'power_spectrum': np.array(30001, 3)  # PSD por eje
                },
                'fft_S1': {...},
                'nivel_dano': 'N1',
                'tipo': 'B',
                ...
            },
            ...
        ]

    Notes:
        - Solo se retorna la mitad positiva del espectro (0 a Nyquist = 50 Hz)
        - Power spectrum = magnitudes^2, normalizado por N
        - Cada eje (N_S, E_W, U_D) se procesa independientemente

    Examples:
        >>> fft_data = apply_fft_to_pairs(paired_data, sampling_rate=100)
        >>> first_pair = fft_data[0]
        >>> freqs = first_pair['fft_S2']['freqs']
        >>> print(f"Rango de frecuencias: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
        Rango de frecuencias: 0.00 - 50.00 Hz
    """
    if verbose := True:
        print("🔄 PASO 2: Aplicando FFT a cada par (S2, S1)...")
        print(f"{'='*60}")

    fft_enriched_data = []

    for pair in paired_data:
        specimen_id = pair['specimen_id']
        signal_s2 = pair['signal_S2']  # Shape: (60000, 3)
        signal_s1 = pair['signal_S1']

        n_samples = signal_s2.shape[0]

        # Calcular frecuencias (solo mitad positiva: 0 a Nyquist)
        freqs = fftfreq(n_samples, d=1/sampling_rate)
        positive_freq_mask = freqs >= 0
        freqs_positive = freqs[positive_freq_mask]

        # Función helper para calcular FFT de una señal multicanal
        def compute_fft_multiaxis(signal: np.ndarray) -> Dict:
            """
            Calcula FFT para señal con 3 ejes (N_S, E_W, U_D).

            Args:
                signal: Array con shape (n_samples, 3)

            Returns:
                Dict con 'freqs', 'magnitudes', 'power_spectrum'
            """
            # Inicializar arrays para almacenar resultados
            n_freqs = len(freqs_positive)
            magnitudes = np.zeros((n_freqs, 3))
            power_spectrum = np.zeros((n_freqs, 3))

            # Calcular FFT para cada eje
            for axis_idx in range(3):  # 0=N_S, 1=E_W, 2=U_D
                signal_axis = signal[:, axis_idx]

                # Aplicar FFT
                fft_result = fft(signal_axis)

                # Extraer solo frecuencias positivas
                fft_positive = fft_result[positive_freq_mask]

                # Calcular magnitud
                magnitude = np.abs(fft_positive)

                # Calcular power spectrum (PSD)
                # Normalizado por N para conservar energía
                power = (magnitude ** 2) / n_samples

                magnitudes[:, axis_idx] = magnitude
                power_spectrum[:, axis_idx] = power

            return {
                'freqs': freqs_positive,
                'magnitudes': magnitudes,
                'power_spectrum': power_spectrum
            }

        # Aplicar FFT a S2 y S1
        fft_s2 = compute_fft_multiaxis(signal_s2)
        fft_s1 = compute_fft_multiaxis(signal_s1)

        # Crear diccionario enriquecido
        enriched_pair = {
            **pair,  # Mantener todos los datos originales
            'fft_S2': fft_s2,
            'fft_S1': fft_s1
        }

        fft_enriched_data.append(enriched_pair)

        if verbose:
            n_freqs = len(freqs_positive)
            freq_resolution = freqs_positive[1] - freqs_positive[0]
            print(f"   ✓ {specimen_id}: FFT aplicado")
            print(f"      • Frecuencias: {n_freqs} bins | Resolución: {freq_resolution:.4f} Hz")
            print(f"      • Rango: {freqs_positive[0]:.2f} - {freqs_positive[-1]:.2f} Hz")

    if verbose:
        print(f"{'='*60}")
        print(f"✅ FFT aplicado exitosamente a {len(fft_enriched_data)} pares\n")

    return fft_enriched_data


def extract_simple_spectral_features(
    fft_data: List[Dict],
    freq_range: Tuple[float, float] = (0, 20),
    include_transfer_features: bool = False
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Extrae características espectrales de cada par (S2, S1) para clustering.

    Args:
        fft_data: Lista de diccionarios con FFT aplicado
        freq_range: Tupla (min_freq, max_freq) para el análisis. Default: (0, 20) Hz
        include_transfer_features: Si True, agrega features de H(f) y Δ(f)

    Returns:
        Tupla (features_matrix, features_df, feature_names)
        - Sin transfer features: 18 características
        - Con transfer features: 36 características (18 base + 18 transfer)
    """
    if verbose := True:
        print("🔄 Extrayendo características espectrales...")
        print(f"{'='*60}")
        print(f"   Rango de frecuencias: {freq_range[0]} - {freq_range[1]} Hz")
        print(f"   Include transfer features: {include_transfer_features}")

    # Nombres de características
    axis_names = ['NS', 'EW', 'UD']
    feature_names = []

    # Frecuencias dominantes
    for sensor in ['S2', 'S1']:
        for axis in axis_names:
            feature_names.append(f'freq_dom_{sensor}_{axis}')

    # Magnitudes de pico
    for sensor in ['S2', 'S1']:
        for axis in axis_names:
            feature_names.append(f'mag_peak_{sensor}_{axis}')

    # Energía total
    for sensor in ['S2', 'S1']:
        for axis in axis_names:
            feature_names.append(f'energy_{sensor}_{axis}')

    # Features de transferencia (ratio y delta) si se solicitan
    if include_transfer_features:
        for axis in axis_names:
            feature_names.append(f'ratio_mean_{axis}')   # S1/S2 mean
            feature_names.append(f'ratio_std_{axis}')    # S1/S2 std
            feature_names.append(f'ratio_max_{axis}')    # S1/S2 max
        for axis in axis_names:
            feature_names.append(f'delta_mean_{axis}')   # S1-S2 mean
            feature_names.append(f'delta_std_{axis}')    # S1-S2 std
            feature_names.append(f'delta_energy_{axis}') # S1-S2 energy

    # Extraer features de cada par
    features_list = []
    metadata_list = []

    for pair in fft_data:
        specimen_id = pair['specimen_id']
        nivel_dano = pair['nivel_dano']

        # Obtener frecuencias y aplicar máscara de rango
        freqs = pair['fft_S2']['freqs']
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs_filtered = freqs[freq_mask]

        # Inicializar vector de features para este espécimen
        specimen_features = []

        # Extraer features para S2 y S1
        for sensor_key in ['fft_S2', 'fft_S1']:
            fft_info = pair[sensor_key]
            magnitudes = fft_info['magnitudes'][freq_mask, :]  # Shape: (n_freqs_filtered, 3)
            power_spectrum = fft_info['power_spectrum'][freq_mask, :]

            # Por cada eje (N-S, E-W, U-D)
            for axis_idx in range(3):
                mag_axis = magnitudes[:, axis_idx]
                power_axis = power_spectrum[:, axis_idx]

                # 1. Frecuencia dominante
                peak_idx = np.argmax(mag_axis)
                freq_dominante = freqs_filtered[peak_idx]
                specimen_features.append(freq_dominante)

            # Segunda pasada: magnitudes de pico
            for axis_idx in range(3):
                mag_axis = magnitudes[:, axis_idx]
                mag_peak = np.max(mag_axis)
                specimen_features.append(mag_peak)

            # Tercera pasada: energía total
            for axis_idx in range(3):
                power_axis = power_spectrum[:, axis_idx]
                energia_total = np.sum(power_axis)
                specimen_features.append(energia_total)

        # Features de transferencia H(f) y Δ(f)
        if include_transfer_features:
            mag_s2 = pair['fft_S2']['magnitudes'][freq_mask, :]
            mag_s1 = pair['fft_S1']['magnitudes'][freq_mask, :]

            epsilon = 1e-10
            H_f = mag_s1 / (mag_s2 + epsilon)  # Función de transferencia
            delta_f = mag_s1 - mag_s2          # Diferencia espectral

            # H(f) features por eje
            for axis_idx in range(3):
                specimen_features.append(np.mean(H_f[:, axis_idx]))
                specimen_features.append(np.std(H_f[:, axis_idx]))
                specimen_features.append(np.max(H_f[:, axis_idx]))

            # Δ(f) features por eje
            for axis_idx in range(3):
                specimen_features.append(np.mean(delta_f[:, axis_idx]))
                specimen_features.append(np.std(delta_f[:, axis_idx]))
                specimen_features.append(np.sum(np.abs(delta_f[:, axis_idx])))

        features_list.append(specimen_features)
        metadata_list.append({
            'specimen_id': specimen_id,
            'nivel_dano': nivel_dano,
            'tipo': pair['tipo']
        })

        if verbose:
            print(f"   ✓ {specimen_id} ({nivel_dano}): {len(specimen_features)} features extraídas")

    # Convertir a numpy array
    features_matrix = np.array(features_list)

    # Crear DataFrame con features + metadata
    features_df = pd.DataFrame(features_matrix, columns=feature_names)
    for key in ['specimen_id', 'nivel_dano', 'tipo']:
        features_df.insert(0, key, [m[key] for m in metadata_list])

    if verbose:
        print(f"{'='*60}")
        print(f"✅ Features extraídas exitosamente")
        print(f"   • Matriz de features: {features_matrix.shape}")
        print(f"   • {features_matrix.shape[0]} especímenes × {features_matrix.shape[1]} características")
        print(f"   • Rango de valores por feature:")
        print(f"      - Min: {features_matrix.min():.6f}")
        print(f"      - Max: {features_matrix.max():.6f}")
        print(f"      - Mean: {features_matrix.mean():.6f}\n")

    return features_matrix, features_df, feature_names


def evaluate_clustering(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    labels_real: np.ndarray,
    kmeans_model: Optional[Any] = None,
    experiment_name: str = "Clustering"
) -> Dict[str, float]:
    """
    Calcula y muestra métricas de clustering (intrínsecas y extrínsecas).

    Args:
        features: Matriz de features normalizadas
        cluster_labels: Labels asignados por el algoritmo de clustering
        labels_real: Labels reales (N1, N2, N3, Sin_etiqueta)
        kmeans_model: Modelo KMeans (opcional, para obtener inertia)
        experiment_name: Nombre del experimento para identificar en el output

    Returns:
        Dict con todas las métricas calculadas
    """
    print("=" * 60)
    print(f"📊 MÉTRICAS: {experiment_name}")
    print("=" * 60)

    metrics = {}

    # --- Métricas INTRÍNSECAS ---
    print("\n🔹 Intrínsecas:")

    if kmeans_model is not None:
        metrics['inertia'] = kmeans_model.inertia_
        print(f"   Inertia:           {metrics['inertia']:>10.2f}")

    metrics['silhouette'] = silhouette_score(features, cluster_labels)
    metrics['davies_bouldin'] = davies_bouldin_score(features, cluster_labels)
    metrics['calinski'] = calinski_harabasz_score(features, cluster_labels)

    print(f"   Silhouette:        {metrics['silhouette']:>10.4f}  [-1, 1] ↑")
    print(f"   Davies-Bouldin:    {metrics['davies_bouldin']:>10.4f}  [0, ∞) ↓")
    print(f"   Calinski-Harabasz: {metrics['calinski']:>10.2f}  [0, ∞) ↑")

    # --- Métricas EXTRÍNSECAS ---
    label_mapping = {'N1': 0, 'N2': 1, 'N3': 2}
    mask = labels_real != 'Sin_etiqueta'
    n_labeled = mask.sum()

    print(f"\n🔹 Extrínsecas ({n_labeled}/{len(labels_real)} con etiqueta):")

    if n_labeled > 0:
        labels_true = np.array([label_mapping[l] for l in labels_real[mask]])
        labels_pred = cluster_labels[mask]

        metrics['ari'] = adjusted_rand_score(labels_true, labels_pred)
        metrics['nmi'] = normalized_mutual_info_score(labels_true, labels_pred)
        metrics['homogeneity'] = homogeneity_score(labels_true, labels_pred)
        metrics['completeness'] = completeness_score(labels_true, labels_pred)
        metrics['v_measure'] = v_measure_score(labels_true, labels_pred)

        print(f"   ARI:               {metrics['ari']:>10.4f}  [-1, 1] ↑")
        print(f"   NMI:               {metrics['nmi']:>10.4f}  [0, 1] ↑")
        print(f"   Homogeneity:       {metrics['homogeneity']:>10.4f}  [0, 1] ↑")
        print(f"   Completeness:      {metrics['completeness']:>10.4f}  [0, 1] ↑")
        print(f"   V-Measure:         {metrics['v_measure']:>10.4f}  [0, 1] ↑")
    else:
        print("   ⚠️ No hay especímenes con etiqueta")

    print("=" * 60)

    return metrics
