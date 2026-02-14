"""
Logger para tracking de entrenamientos de modelos.

Registra hiperparámetros, métricas y metadata de cada entrenamiento
en un archivo JSON dentro del directorio del experimento.

Autor: Giancarlo Poémape Lozano
Fecha: 2026-02-14
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


def save_experiment_log(
    output_path: Path,
    config: Dict,
    dataset_info: Dict,
    metrics: Dict
) -> None:
    """
    Guarda log del experimento en experiment_log.json dentro de results/.

    Args:
        output_path: Directorio del experimento actual (results/)
        config: Hiperparámetros del entrenamiento
        dataset_info: Información del dataset (n_samples, split)
        metrics: Métricas finales del entrenamiento
    """
    experiment_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'output_dir': str(output_path),
        'config': config,
        'dataset': dataset_info,
        'metrics': metrics
    }

    log_file = output_path / 'experiment_log.json'
    with open(log_file, 'w') as f:
        json.dump(experiment_entry, f, indent=2)
    print(f"   ✓ Experiment log: {log_file}")
