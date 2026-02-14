"""
Utilidades para carga de datasets de señales raw.
"""

import os
from glob import glob
from typing import Optional

import pandas as pd
import numpy as np


def load_raw_signals_dataset(
    signals_dir: str,
    labels_csv_path: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Carga todos los archivos de señales raw y crea dataset con etiquetas.

    Args:
        signals_dir: Directorio con señales raw (data/Signals_Raw/)
        labels_csv_path: Archivo CSV con etiquetas (nivel_damage.csv)
        verbose: Si True, imprime progreso

    Returns:
        DataFrame con columnas: ['ID', 'sensor', 'N_S', 'E_W', 'U_D', 'tipo', 'nivel_damage']
    """
    # Cargar etiquetas
    labels_df = pd.read_csv(labels_csv_path)
    labels_dict = {
        row['ID']: {'tipo': row['TIPO'], 'nivel_damage': row['Ndano']}
        for _, row in labels_df.iterrows()
    }

    if verbose:
        print(f"✓ Cargadas etiquetas para {len(labels_dict)} especímenes")

    all_data = []
    specimen_dirs = sorted([
        d for d in os.listdir(signals_dir)
        if d.startswith('A') and os.path.isdir(os.path.join(signals_dir, d))
    ])

    if verbose:
        print(f"✓ Encontradas {len(specimen_dirs)} carpetas de especímenes")

    for specimen_id in specimen_dirs:
        if specimen_id not in labels_dict:
            continue

        specimen_path = os.path.join(signals_dir, specimen_id)
        signal_files = glob(os.path.join(specimen_path, "completo_S*.txt"))

        for signal_file in signal_files:
            sensor = 'S1' if 'S1' in os.path.basename(signal_file) else 'S2'

            try:
                # Leer archivo separado por espacios
                signal_data = pd.read_csv(signal_file, sep=r'\s+', skiprows=1)

                # Extraer columnas de aceleración (índices 2, 3, 4)
                signal_data = signal_data.iloc[:, [2, 3, 4]]
                signal_data.columns = ['N_S', 'E_W', 'U_D']

                # Agregar metadatos
                signal_data['ID'] = specimen_id
                signal_data['sensor'] = sensor
                signal_data['tipo'] = labels_dict[specimen_id]['tipo']
                signal_data['nivel_damage'] = labels_dict[specimen_id]['nivel_damage']

                all_data.append(signal_data)

            except Exception as e:
                if verbose:
                    print(f"❌ Error procesando {signal_file}: {e}")

    if not all_data:
        raise ValueError("No se pudieron cargar datos de señales")

    final_dataset = pd.concat(all_data, ignore_index=True)

    if verbose:
        print(f"✓ Dataset creado: {len(final_dataset):,} filas")
        print(f"  Especímenes: {final_dataset['ID'].nunique()}")

    return final_dataset
