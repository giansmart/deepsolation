"""
Aplicación de correcciones de sincronización temporal entre sensores S1 y S2.

Este módulo implementa la corrección determinista de desincronización mediante
shift de índices de array, preservando los datos originales y generando
metadata de trazabilidad.

Autor: Giancarlo Poémape Lozano
Fecha: 2026-02-07
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .detect_timestamp_offsets import read_first_timestamp


SAMPLING_RATE = 100  # Hz
TARGET_LENGTH = 60000  # muestras (10 minutos a 100 Hz)
MIN_SIGNAL_LENGTH = 58000  # muestras mínimas requeridas después de sincronización (9.67 min)
MAX_OFFSET_SECONDS = 3600  # offset máximo tolerable (1 hora)


def standardize_signal_length(signal: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    """
    Estandariza longitud de señal a target_length.

    Args:
        signal: Array de forma (n_samples, 3) con ejes [N-S, E-W, U-D]
        target_length: Longitud objetivo en muestras

    Returns:
        Array de forma (target_length, 3)
    """
    current_length = signal.shape[0]

    if current_length > target_length:
        # Truncar
        return signal[:target_length, :]
    elif current_length < target_length:
        # Padding con ceros
        pad_length = target_length - current_length
        return np.pad(signal, ((0, pad_length), (0, 0)), mode='constant')
    else:
        # Ya tiene la longitud correcta
        return signal


def apply_shift_correction(
    S2: np.ndarray,
    S1: np.ndarray,
    offset_seconds: float,
    timestamp_s2_start: datetime,
    sampling_rate: int = SAMPLING_RATE,
    min_length: int = MIN_SIGNAL_LENGTH
) -> Tuple[np.ndarray, np.ndarray, datetime]:
    """
    Aplica corrección de sincronización mediante truncado inteligente.

    Estrategia (SIN PADDING CON CEROS):
    - Si offset > 0: S1 está adelantado → truncar inicio de S1
    - Si offset < 0: S1 está atrasado → truncar inicio de S2 (NO padding)
    - Si offset = 0: No hacer nada

    Args:
        S2: Señal S2 (base), shape (n_samples, 3)
        S1: Señal S1 (superior), shape (n_samples, 3)
        offset_seconds: Offset en segundos (S1 - S2)
        timestamp_s2_start: Timestamp inicial de S2 (para ajustar si se trunca S2)
        sampling_rate: Frecuencia de muestreo en Hz
        min_length: Longitud mínima requerida después de sincronización

    Returns:
        Tuple (S2_corrected, S1_corrected, timestamp_adjusted)
        - S2_corrected, S1_corrected: arrays sincronizados de longitud igual
        - timestamp_adjusted: timestamp inicial ajustado (cambia si se truncó S2)

    Raises:
        ValueError: Si offset extremo, señal resultante muy corta, o datos insuficientes
    """
    # Validación 1: Offset máximo tolerable
    if abs(offset_seconds) > MAX_OFFSET_SECONDS:
        raise ValueError(
            f"Offset extremo: {offset_seconds}s > {MAX_OFFSET_SECONDS}s (máximo tolerable)"
        )

    # Caso trivial: ya sincronizado
    if abs(offset_seconds) < 0.01:  # Tolerancia de 10ms
        min_len = min(S2.shape[0], S1.shape[0])
        S2_trimmed = S2[:min_len, :]
        S1_trimmed = S1[:min_len, :]

        # Validación 2: Longitud mínima
        if min_len < min_length:
            raise ValueError(
                f"Señales demasiado cortas: {min_len} muestras < {min_length} requerido"
            )

        return S2_trimmed, S1_trimmed, timestamp_s2_start

    # Convertir offset a número de muestras
    offset_samples = int(round(abs(offset_seconds) * sampling_rate))

    if offset_seconds > 0:
        # S1 adelantado → truncar inicio de S1
        if offset_samples >= S1.shape[0]:
            raise ValueError(
                f"Offset ({offset_samples} muestras) >= longitud de S1 ({S1.shape[0]} muestras)"
            )

        S1_corrected = S1[offset_samples:, :]
        min_len = min(S2.shape[0], S1_corrected.shape[0])
        S2_trimmed = S2[:min_len, :]
        S1_corrected = S1_corrected[:min_len, :]

        # Timestamp no cambia (S2 empieza donde siempre)
        timestamp_adjusted = timestamp_s2_start

    else:  # offset < 0
        # S1 atrasado → truncar inicio de S2 (NO PADDING CON CEROS)
        if offset_samples >= S2.shape[0]:
            raise ValueError(
                f"Offset ({offset_samples} muestras) >= longitud de S2 ({S2.shape[0]} muestras)"
            )

        S2_trimmed = S2[offset_samples:, :]  # Saltar primeras muestras de S2
        min_len = min(S2_trimmed.shape[0], S1.shape[0])
        S2_trimmed = S2_trimmed[:min_len, :]
        S1_corrected = S1[:min_len, :]

        # Timestamp ajustado: S2 ahora empieza más tarde
        timestamp_adjusted = timestamp_s2_start + timedelta(seconds=abs(offset_seconds))

    # Validación 2: Longitud mínima después de sincronización
    final_length = min_len
    if final_length < min_length:
        raise ValueError(
            f"Señal resultante muy corta: {final_length} muestras < {min_length} requerido "
            f"(offset={offset_seconds}s, S2_orig={S2.shape[0]}, S1_orig={S1.shape[0]})"
        )

    return S2_trimmed, S1_corrected, timestamp_adjusted


def validate_synchronization(S2_sync: np.ndarray, S1_sync: np.ndarray) -> Dict:
    """
    Valida calidad de sincronización usando correlación cruzada básica.

    Args:
        S2_sync: Señal S2 sincronizada, shape (n_samples, 3)
        S1_sync: Señal S1 sincronizada, shape (n_samples, 3)

    Returns:
        Dict con métricas: {lag_samples, lag_seconds, max_correlation, is_valid}
    """
    # Usar solo eje N-S para validación rápida
    s2_ns = S2_sync[:, 0]
    s1_ns = S1_sync[:, 0]

    # Correlación cruzada normalizada
    corr = np.correlate(s2_ns, s1_ns, mode='same')
    corr_normalized = corr / (np.std(s2_ns) * np.std(s1_ns) * len(s2_ns))

    # Encontrar lag en el pico de correlación
    lag_at_max = np.argmax(corr_normalized) - len(corr_normalized) // 2
    max_correlation = corr_normalized[np.argmax(corr_normalized)]

    # Convertir a segundos
    lag_seconds = lag_at_max / SAMPLING_RATE

    # Validar: lag debe ser < 1 segundo
    is_valid = abs(lag_at_max) < SAMPLING_RATE  # < 100 muestras = 1 segundo

    if not is_valid:
        warnings.warn(
            f"Sincronización subóptima: lag residual = {lag_at_max} muestras "
            f"({lag_seconds:.2f}s)"
        )

    return {
        'lag_samples': int(lag_at_max),
        'lag_seconds': float(lag_seconds),
        'max_correlation': float(max_correlation),
        'is_valid': bool(is_valid)  # Convertir numpy.bool_ a bool nativo
    }


def save_signal_as_txt(
    signal: np.ndarray,
    timestamp_start: datetime,
    output_path: Path,
    sampling_rate: int = SAMPLING_RATE
) -> None:
    """
    Guarda señal en formato texto con timestamps reconstruidos.

    Args:
        signal: Array de forma (n_samples, 3) con ejes [N-S, E-W, U-D]
        timestamp_start: Timestamp del primer dato (ya corregido por offset)
        output_path: Ruta del archivo .txt de salida
        sampling_rate: Frecuencia de muestreo en Hz

    Formato de salida:
        Fecha Hora  N_S  E_W  U_D
        YYYY/MM/DD HH:MM:SS.mmm  val1  val2  val3
        ...
    """
    n_samples = signal.shape[0]

    # Generar timestamps para cada muestra
    timestamps = [
        timestamp_start + timedelta(seconds=i / sampling_rate)
        for i in range(n_samples)
    ]

    # Escribir archivo
    with open(output_path, 'w') as f:
        # Encabezado
        f.write("Fecha Hora  N_S  E_W  U_D\n")

        # Datos
        for ts, (ns, ew, ud) in zip(timestamps, signal):
            # Formatear timestamp: "YYYY/MM/DD HH:MM:SS.mmm"
            ts_str = ts.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]  # Truncar a milisegundos
            f.write(f"{ts_str}  {ns:.8f}  {ew:.8f}  {ud:.8f}\n")


def apply_corrections(
    signals_dir: str,
    offsets_csv: str,
    output_dir: str,
    method: str = 'shift_indices'
) -> Dict:
    """
    Aplica correcciones de sincronización a señales S1.

    Args:
        signals_dir: Directorio raíz de señales RAW (ej: "data/Signals_Raw/")
        offsets_csv: CSV con offsets detectados
        output_dir: Directorio de salida (ej: "data/processed/synchronized/")
        method: Método de corrección ('shift_indices' únicamente soportado)

    Returns:
        Dict con estadísticas: {corrected: int, skipped: int, errors: List}

    Raises:
        FileNotFoundError: Si signals_dir o offsets_csv no existen
        ValueError: Si method no es 'shift_indices'
    """
    if method != 'shift_indices':
        raise ValueError(f"Método '{method}' no soportado. Usar 'shift_indices'")

    signals_path = Path(signals_dir)
    offsets_path = Path(offsets_csv)
    output_path = Path(output_dir)

    if not signals_path.exists():
        raise FileNotFoundError(f"Directorio de señales no encontrado: {signals_dir}")
    if not offsets_path.exists():
        raise FileNotFoundError(f"Tabla de offsets no encontrada: {offsets_csv}")

    # Leer tabla de offsets
    print("=" * 70)
    print("🔧 APLICACIÓN DE CORRECCIONES DE SINCRONIZACIÓN")
    print("=" * 70)
    print(f"\n📋 Paso 1: Cargando tabla de offsets desde {offsets_csv}...")

    offsets_df = pd.read_csv(offsets_path)

    print(f"   ✓ Cargados {len(offsets_df)} registros")
    print(f"\n⚙️  Paso 2: Procesando correcciones (método: {method})...\n")

    # Crear directorio de salida
    output_path.mkdir(parents=True, exist_ok=True)

    # Estadísticas
    stats = {
        'corrected': 0,
        'already_synced': 0,
        'errors': [],
        'validation_passed': 0,
        'validation_failed': 0
    }

    for idx, row in offsets_df.iterrows():
        edificio = row['edificio']
        pasada = row['pasada']
        specimen_id = row['specimen_id']
        offset_seconds = row['offset_seconds']
        sync_status = row['sync_status']

        # Rutas
        specimen_dir_raw = signals_path / edificio / pasada / specimen_id
        specimen_dir_out = output_path / edificio / pasada / specimen_id

        # Crear directorio de salida
        specimen_dir_out.mkdir(parents=True, exist_ok=True)

        try:
            # Buscar archivos S1 y S2
            s2_files = list(specimen_dir_raw.glob("completo_S2*.txt"))
            s1_files = list(specimen_dir_raw.glob("completo_S1*.txt"))

            if not s2_files or not s1_files:
                raise FileNotFoundError(f"Archivos S1/S2 no encontrados en {specimen_dir_raw}")

            s2_file = s2_files[0]
            s1_file = s1_files[0]

            # Cargar señales
            df_s2 = pd.read_csv(s2_file, sep=r'\s+', engine='python', skiprows=1)
            df_s1 = pd.read_csv(s1_file, sep=r'\s+', engine='python', skiprows=1)

            # Extraer columnas [N_S, E_W, U_D] (columnas 2, 3, 4)
            S2 = df_s2.iloc[:, [2, 3, 4]].values.astype(np.float32)
            S1 = df_s1.iloc[:, [2, 3, 4]].values.astype(np.float32)

            # Leer timestamp inicial de S2 (para ajuste si se trunca S2)
            ts_s2_start, _ = read_first_timestamp(s2_file)

            # Aplicar corrección
            if sync_status == 'SYNCED':
                # Ya sincronizado, solo igualar longitudes
                S2_corrected, S1_corrected, ts_adjusted = apply_shift_correction(
                    S2, S1, 0.0, ts_s2_start
                )
                stats['already_synced'] += 1
                icon = '✅'
            else:
                # Aplicar corrección por shift/truncado
                S2_corrected, S1_corrected, ts_adjusted = apply_shift_correction(
                    S2, S1, offset_seconds, ts_s2_start
                )
                stats['corrected'] += 1
                icon = '🔧'

            # Re-estandarizar a TARGET_LENGTH
            S2_final = standardize_signal_length(S2_corrected, TARGET_LENGTH)
            S1_final = standardize_signal_length(S1_corrected, TARGET_LENGTH)

            # Validar sincronización
            validation = validate_synchronization(S2_final, S1_final)
            if validation['is_valid']:
                stats['validation_passed'] += 1
                val_icon = '✓'
            else:
                stats['validation_failed'] += 1
                val_icon = '⚠'

            # Usar timestamp ajustado (ya calculado en apply_shift_correction)
            # Ambas señales empiezan en el mismo instante físico después de corrección
            ts_synchronized = ts_adjusted

            # Guardar señales sincronizadas (.npy para entrenamiento)
            np.save(specimen_dir_out / 'S2_synchronized.npy', S2_final)
            np.save(specimen_dir_out / 'S1_synchronized.npy', S1_final)

            # Guardar señales sincronizadas (.txt para revisión manual)
            save_signal_as_txt(S2_final, ts_synchronized, specimen_dir_out / 'S2_synchronized.txt')
            save_signal_as_txt(S1_final, ts_synchronized, specimen_dir_out / 'S1_synchronized.txt')

            # Guardar metadata
            metadata = {
                'offset_applied': float(offset_seconds),
                'method': method,
                'original_length_S2': int(S2.shape[0]),
                'original_length_S1': int(S1.shape[0]),
                'final_length': int(TARGET_LENGTH),
                'validation': validation
            }

            with open(specimen_dir_out / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"   {icon} {edificio}/{pasada}/{specimen_id}: "
                  f"offset={offset_seconds:+7.1f}s, validation={val_icon} "
                  f"(lag={validation['lag_seconds']:.3f}s)")

        except Exception as e:
            stats['errors'].append({
                'specimen_id': f"{edificio}/{pasada}/{specimen_id}",
                'error': str(e)
            })
            print(f"   ❌ {edificio}/{pasada}/{specimen_id}: Error - {str(e)}")

    # Resumen final
    print(f"\n{'='*70}")
    print(f"📊 RESUMEN DE CORRECCIONES")
    print(f"{'='*70}")
    print(f"   Total mediciones procesadas:    {len(offsets_df)}")
    print(f"   Señales corregidas:             {stats['corrected']}")
    print(f"   Ya sincronizadas (sin cambios): {stats['already_synced']}")
    print(f"   Errores:                        {len(stats['errors'])}\n")

    print(f"   🔍 Validación post-corrección:")
    print(f"      Validaciones exitosas:       {stats['validation_passed']}")
    print(f"      Validaciones fallidas:       {stats['validation_failed']}")

    if stats['validation_failed'] > 0:
        pct_failed = (stats['validation_failed'] / (stats['validation_passed'] + stats['validation_failed'])) * 100
        print(f"      ⚠️  {pct_failed:.1f}% con sincronización subóptima")

    print(f"{'='*70}\n")

    return stats


def main():
    """
    Función principal para ejecución standalone.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Aplica correcciones de sincronización a señales S1/S2'
    )
    parser.add_argument(
        '--signals-dir',
        default='data/Signals_Raw/',
        help='Directorio raíz de señales (default: data/Signals_Raw/)'
    )
    parser.add_argument(
        '--offsets-csv',
        default='data/processed/timestamp_offsets.csv',
        help='Tabla de offsets (default: data/processed/timestamp_offsets.csv)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed/synchronized/',
        help='Directorio de salida (default: data/processed/synchronized/)'
    )
    parser.add_argument(
        '--method',
        default='shift_indices',
        choices=['shift_indices'],
        help='Método de corrección (default: shift_indices)'
    )

    args = parser.parse_args()

    # Ejecutar correcciones
    stats = apply_corrections(
        signals_dir=args.signals_dir,
        offsets_csv=args.offsets_csv,
        output_dir=args.output_dir,
        method=args.method
    )

    print(f"✅ Señales sincronizadas guardadas en: {args.output_dir}")
    print(f"   {stats['corrected']} señales corregidas")
    print(f"   {stats['already_synced']} ya sincronizadas\n")


if __name__ == '__main__':
    main()
