"""
Utilidades para análisis de wavelets en pares de señales (S2, S1).

Fecha: 2026-01-20
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import pywt


def apply_wavelet_to_pairs(
    paired_data: List[Dict],
    wavelet: str = 'db4',
    level: int = 5
) -> List[Dict]:
    """
    Aplica Discrete Wavelet Transform (DWT) a cada par (S2, S1).

    La DWT descompone la señal en coeficientes de aproximación (baja frecuencia)
    y detalle (alta frecuencia) en múltiples niveles.

    Args:
        paired_data: Lista de diccionarios con pares (S2, S1)
        wavelet: Familia de wavelet ('db4', 'sym5', 'coif3', etc.)
        level: Niveles de descomposición (default: 5)

    Returns:
        Lista de diccionarios enriquecida con coeficientes wavelet
    """
    print(f"🔄 Aplicando DWT ({wavelet}, nivel={level}) a cada par...")

    wavelet_enriched_data = []

    for pair in paired_data:
        signal_s2 = pair['signal_S2']
        signal_s1 = pair['signal_S1']

        def compute_dwt_multiaxis(signal: np.ndarray) -> Dict:
            """Calcula DWT para señal con 3 ejes."""
            coeffs_by_axis = []
            for axis_idx in range(signal.shape[1]):
                axis_signal = signal[:, axis_idx]
                coeffs = pywt.wavedec(axis_signal, wavelet, level=level)
                coeffs_by_axis.append(coeffs)

            return {
                'coeffs': coeffs_by_axis,
                'wavelet': wavelet,
                'level': level
            }

        enriched_pair = {
            **pair,
            'dwt_S2': compute_dwt_multiaxis(signal_s2),
            'dwt_S1': compute_dwt_multiaxis(signal_s1)
        }
        wavelet_enriched_data.append(enriched_pair)

    print(f"✅ DWT aplicado a {len(wavelet_enriched_data)} pares\n")
    return wavelet_enriched_data


def extract_wavelet_features(
    wavelet_data: List[Dict],
    include_transfer_features: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Extrae características estadísticas de los coeficientes wavelet.

    Args:
        wavelet_data: Lista de pares con DWT aplicado
        include_transfer_features: Si True, agrega features ratio y delta (S1/S2, S1-S2)
        verbose: Mostrar progreso

    Returns:
        Tuple con (matriz_features, dataframe_features, nombres_features)
    """
    if verbose:
        print("🔄 Extrayendo características de coeficientes wavelet...")
        print(f"   Include transfer features: {include_transfer_features}")

    features_list = []
    metadata_list = []
    feature_names = None

    for pair in wavelet_data:
        specimen_id = pair['specimen_id']
        nivel_dano = pair['nivel_dano']
        tipo = pair['tipo']

        dwt_s2 = pair['dwt_S2']
        dwt_s1 = pair['dwt_S1']
        n_levels = dwt_s2['level']

        features = {}
        axis_names = ['NS', 'EW', 'UD']

        # Features base: estadísticas de S2 y S1
        for signal_name, dwt_data in [('S2', dwt_s2), ('S1', dwt_s1)]:
            for axis_idx, axis_name in enumerate(axis_names):
                coeffs = dwt_data['coeffs'][axis_idx]

                for level_idx, coeff in enumerate(coeffs):
                    level_name = 'cA' if level_idx == 0 else f'cD{n_levels - level_idx + 1}'
                    prefix = f'{signal_name}_{axis_name}_{level_name}'

                    features[f'{prefix}_energy'] = np.sum(coeff ** 2)
                    features[f'{prefix}_mean_abs'] = np.mean(np.abs(coeff))
                    features[f'{prefix}_std'] = np.std(coeff)

        # Features de transferencia: ratio y delta entre S1 y S2
        if include_transfer_features:
            epsilon = 1e-10
            for axis_idx, axis_name in enumerate(axis_names):
                coeffs_s2 = dwt_s2['coeffs'][axis_idx]
                coeffs_s1 = dwt_s1['coeffs'][axis_idx]

                for level_idx in range(len(coeffs_s2)):
                    level_name = 'cA' if level_idx == 0 else f'cD{n_levels - level_idx + 1}'
                    prefix = f'transfer_{axis_name}_{level_name}'

                    energy_s2 = np.sum(coeffs_s2[level_idx] ** 2)
                    energy_s1 = np.sum(coeffs_s1[level_idx] ** 2)
                    mean_abs_s2 = np.mean(np.abs(coeffs_s2[level_idx]))
                    mean_abs_s1 = np.mean(np.abs(coeffs_s1[level_idx]))

                    # Ratio features
                    features[f'{prefix}_ratio_energy'] = energy_s1 / (energy_s2 + epsilon)
                    features[f'{prefix}_ratio_mean_abs'] = mean_abs_s1 / (mean_abs_s2 + epsilon)

                    # Delta features
                    features[f'{prefix}_delta_energy'] = energy_s1 - energy_s2

        if feature_names is None:
            feature_names = list(features.keys())

        features_list.append(list(features.values()))
        metadata_list.append({
            'specimen_id': specimen_id,
            'nivel_dano': nivel_dano,
            'tipo': tipo
        })

        if verbose:
            print(f"   ✓ {specimen_id} ({nivel_dano}): {len(features)} features")

    features_matrix = np.array(features_list)
    features_df = pd.DataFrame(features_matrix, columns=feature_names)
    for key in ['specimen_id', 'nivel_dano', 'tipo']:
        features_df.insert(0, key, [m[key] for m in metadata_list])

    if verbose:
        print(f"✅ Features wavelet extraídas: {features_matrix.shape}")

    return features_matrix, features_df, feature_names
