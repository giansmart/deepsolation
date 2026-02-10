"""
Detección automática de offsets temporales entre sensores S1 y S2.

Este módulo implementa la detección determinista de desincronización temporal
entre pares de sensores (S2-base, S1-superior) mediante comparación directa
de timestamps.

Autor: Giancarlo Poémape Lozano
Fecha: 2026-02-07
"""

from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parsea timestamp en formato 'YYYY/MM/DD HH:MM:SS.mmm'.

    Args:
        timestamp_str: String con formato 'YYYY/MM/DD HH:MM:SS.mmm'

    Returns:
        datetime object

    Examples:
        >>> parse_timestamp('2025/09/17 08:24:20.000')
        datetime.datetime(2025, 9, 17, 8, 24, 20)
    """
    return datetime.strptime(timestamp_str, '%Y/%m/%d %H:%M:%S.%f')


def read_first_timestamp(file_path: Path) -> Tuple[datetime, str]:
    """
    Lee el timestamp de la primera línea de datos (segunda línea del archivo).

    Args:
        file_path: Ruta al archivo completo_S*.txt

    Returns:
        Tuple (datetime, timestamp_str)

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el formato del timestamp es inválido
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    with open(file_path, 'r') as f:
        header = f.readline()  # Saltarse encabezado "Fecha Hora  N_S  E_W  U_D"
        first_data_line = f.readline().strip()

    # Formato: "YYYY/MM/DD HH:MM:SS.mmm  val1  val2  val3"
    parts = first_data_line.split()
    if len(parts) < 2:
        raise ValueError(f"Formato inválido en {file_path}: {first_data_line}")

    timestamp_str = f"{parts[0]} {parts[1]}"
    timestamp = parse_timestamp(timestamp_str)

    return timestamp, timestamp_str


def detect_offsets(signals_dir: str, labels_csv: str) -> pd.DataFrame:
    """
    Detecta offsets temporales entre S1 y S2 para todas las mediciones.

    **Método determinista**: Compara timestamps de la primera muestra de cada
    archivo usando Python puro (datetime), sin heurísticas ni ML.

    Args:
        signals_dir: Directorio raíz de señales (ej: "data/Signals_Raw/")
        labels_csv: CSV con columnas: edificio, pasada, specimen_id, tipo, nivel_damage

    Returns:
        DataFrame con columnas:
        - edificio, pasada, specimen_id
        - timestamp_S2_start, timestamp_S1_start
        - offset_seconds (S1 - S2)
        - sync_status: 'SYNCED' (<1s) | 'MINOR_OFFSET' (1-60s) | 'MAJOR_OFFSET' (>60s)

    Raises:
        FileNotFoundError: Si signals_dir o labels_csv no existen
    """
    signals_path = Path(signals_dir)
    labels_path = Path(labels_csv)

    if not signals_path.exists():
        raise FileNotFoundError(f"Directorio de señales no encontrado: {signals_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Archivo de etiquetas no encontrado: {labels_csv}")

    # Leer etiquetas
    print("=" * 70)
    print("📊 DETECCIÓN DE OFFSETS TEMPORALES S1-S2")
    print("=" * 70)
    print(f"\n📋 Cargando etiquetas desde {labels_csv}...")

    labels_df = pd.read_csv(labels_path)
    labels_df.columns = ['edificio', 'pasada', 'specimen_id', 'tipo', 'nivel_dano']

    print(f"   ✓ Cargadas {len(labels_df)} mediciones")
    print(f"\n🔍 Analizando pares S1/S2...\n")

    # Contenedor de resultados
    results = []
    skipped = []
    errors = []

    for idx, row in labels_df.iterrows():
        edificio = row['edificio']
        pasada = row['pasada']
        specimen_id = row['specimen_id']

        # Construir ruta al directorio del espécimen
        specimen_dir = signals_path / edificio / pasada / specimen_id

        if not specimen_dir.exists():
            skipped.append(f"{edificio}/{pasada}/{specimen_id}")
            print(f"   ⚠️  {edificio}/{pasada}/{specimen_id}: Directorio no encontrado")
            continue

        # Buscar archivos S1 y S2
        s2_files = list(specimen_dir.glob("completo_S2*.txt"))
        s1_files = list(specimen_dir.glob("completo_S1*.txt"))

        if not s2_files or not s1_files:
            skipped.append(specimen_id)
            print(f"   ⚠️  {specimen_id}: Archivos S1 o S2 faltantes")
            continue

        # Tomar primer archivo encontrado
        s2_file = s2_files[0]
        s1_file = s1_files[0]

        try:
            # Leer timestamps
            ts_s2, ts_s2_str = read_first_timestamp(s2_file)
            ts_s1, ts_s1_str = read_first_timestamp(s1_file)

            # Calcular offset (S1 - S2)
            offset_seconds = (ts_s1 - ts_s2).total_seconds()

            # Clasificar por magnitud
            if abs(offset_seconds) < 1:
                sync_status = 'SYNCED'
                icon = '✅'
            elif abs(offset_seconds) <= 60:
                sync_status = 'MINOR_OFFSET'
                icon = '⚠️ '
            else:
                sync_status = 'MAJOR_OFFSET'
                icon = '🔴'

            results.append({
                'edificio': edificio,
                'pasada': pasada,
                'specimen_id': specimen_id,
                'timestamp_S2_start': ts_s2_str,
                'timestamp_S1_start': ts_s1_str,
                'offset_seconds': offset_seconds,
                'sync_status': sync_status
            })

            print(f"   {icon} {edificio}/{pasada}/{specimen_id}: "
                  f"offset = {offset_seconds:+7.1f}s ({sync_status})")

        except Exception as e:
            errors.append({'specimen_id': specimen_id, 'error': str(e)})
            print(f"   ❌ {specimen_id}: Error - {str(e)}")

    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results)

    # Estadísticas finales
    print(f"\n{'='*70}")
    print(f"📈 RESUMEN DE DETECCIÓN")
    print(f"{'='*70}")
    print(f"   Total mediciones analizadas:  {len(labels_df)}")
    print(f"   Pares procesados exitosamente: {len(results)}")
    print(f"   Omitidos (archivos faltantes): {len(skipped)}")
    print(f"   Errores de procesamiento:      {len(errors)}\n")

    if len(results) > 0:
        # Distribución por estado
        status_counts = results_df['sync_status'].value_counts()
        print(f"   📊 Distribución por estado de sincronización:")
        for status in ['SYNCED', 'MINOR_OFFSET', 'MAJOR_OFFSET']:
            count = status_counts.get(status, 0)
            pct = (count / len(results)) * 100
            print(f"      {status:15s}: {count:3d} ({pct:5.1f}%)")

        # Estadísticas de offsets
        offsets = results_df['offset_seconds'].values
        print(f"\n   📏 Estadísticas de offsets (segundos):")
        print(f"      Mínimo:   {np.min(offsets):+8.1f}s")
        print(f"      Máximo:   {np.max(offsets):+8.1f}s")
        print(f"      Promedio: {np.mean(offsets):+8.1f}s")
        print(f"      Mediana:  {np.median(offsets):+8.1f}s")
        print(f"      Std Dev:  {np.std(offsets):8.1f}s")

        # Análisis por edificio
        print(f"\n   🏢 Análisis por edificio:")
        for edificio in results_df['edificio'].unique():
            edificio_data = results_df[results_df['edificio'] == edificio]
            total = len(edificio_data)
            synced = len(edificio_data[edificio_data['sync_status'] == 'SYNCED'])
            pct_synced = (synced / total) * 100
            print(f"      {edificio}: {synced}/{total} sincronizados ({pct_synced:.1f}%)")

    print(f"{'='*70}\n")

    return results_df


def main():
    """Función principal para ejecución standalone."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Detecta offsets temporales entre sensores S1 y S2'
    )
    parser.add_argument(
        '--signals-dir',
        default='data/Signals_Raw/',
        help='Directorio raíz de señales'
    )
    parser.add_argument(
        '--labels-csv',
        default='data/nivel_damage.csv',
        help='Archivo CSV con etiquetas'
    )
    parser.add_argument(
        '--output',
        default='data/processed/timestamp_offsets.csv',
        help='Archivo CSV de salida'
    )

    args = parser.parse_args()

    # Ejecutar detección
    offsets_df = detect_offsets(
        signals_dir=args.signals_dir,
        labels_csv=args.labels_csv
    )

    # Guardar resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offsets_df.to_csv(output_path, index=False)

    print(f"✅ Tabla de offsets guardada en: {output_path}")
    print(f"   {len(offsets_df)} registros escritos.\n")


if __name__ == '__main__':
    main()
