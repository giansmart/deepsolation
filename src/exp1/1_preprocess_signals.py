#!/usr/bin/env python3
"""
Script de Preprocesamiento de Señales
=====================================

Genera el dataset procesado siguiendo la metodología Yu et al. (2018):
- FFT transformation de señales raw
- PSD selection (70% energy threshold)
- Exporta CSV con características y labels

Uso:
    python scripts/preprocess_signals.py [--output OUTPUT_PATH]

Salidas:
    - results/preprocessed_dataset.csv: Dataset procesado listo para entrenamiento
    - results/preprocessing_summary.txt: Resumen detallado del preprocesamiento
"""

import argparse
import sys
from pathlib import Path
import time
from datetime import datetime

# Agregar src al path para acceder a utils
sys.path.append(str(Path(__file__).parent.parent))

from signal_preprocessing import SignalPreprocessor

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de señales para DCNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

    # Preprocesamiento básico (recomendado)
    python scripts/preprocess_signals.py
    
    # Especificar archivo de salida
    python scripts/preprocess_signals.py --output results/my_dataset.csv
        """
    )
    
    parser.add_argument(
        "--output", 
        default="src/exp1/results/preprocessed_dataset.csv",
        help="Ruta del archivo de salida (default: src/exp1/results/preprocessed_dataset.csv)"
    )
    parser.add_argument(
        "--sampling-rate", 
        type=int,
        default=100,
        help="Sampling rate en Hz (default: 100)"
    )
    parser.add_argument(
        "--energy-threshold", 
        type=float,
        default=0.7,
        help="Energy threshold para PSD selection (default: 0.7)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Modo silencioso"
    )
    
    args = parser.parse_args()
    
    try:
        print("="*70)
        print("PREPROCESAMIENTO DE SEÑALES PARA DCNN")
        print("="*70)
        print(f"Metodología: Yu et al. (2018) - FFT + PSD selection")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Configuración - rutas relativas desde el directorio raíz del proyecto
        project_root = Path(__file__).parent.parent.parent  # deepsolation/
        signals_dir = project_root / "data/Signals_Raw"
        labels_file = project_root / "data/nivel_damage.csv"
        output_path = project_root / args.output
        
        # Validar entradas
        if not signals_dir.exists():
            raise FileNotFoundError(f"Directorio de señales no encontrado: {signals_dir}")
        if not labels_file.exists():
            raise FileNotFoundError(f"Archivo de labels no encontrado: {labels_file}")
        
        print(f"✓ Directorio de señales: {signals_dir}")
        print(f"✓ Archivo de labels: {labels_file}")
        print(f"✓ Archivo de salida: {output_path}")
        print()
        
        # Crear directorio de salida
        output_path.parent.mkdir(exist_ok=True)
        
        # Inicializar preprocessor
        print("Inicializando preprocessor...")
        preprocessor = SignalPreprocessor(
            sampling_rate=args.sampling_rate,
            energy_threshold=args.energy_threshold
        )
        print(f"✓ Sampling rate: {args.sampling_rate} Hz")
        print(f"✓ Energy threshold: {args.energy_threshold*100}%")
        print()
        
        # Ejecutar preprocesamiento
        print("Procesando señales...")
        start_time = time.time()
        
        dataset_df = preprocessor.create_training_dataset_csv(
            signals_dir=str(signals_dir),
            labels_csv_path=str(labels_file),
            output_csv_path=str(output_path)
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print()
        print("="*70)
        print("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"📊 Dataset generado: {output_path}")
        print(f"📈 Total de observaciones: {len(dataset_df):,}")
        print(f"📋 Columnas: {len(dataset_df.columns)}")
        print(f"⏱️ Tiempo de procesamiento: {processing_time:.1f} segundos")
        print()
        
        # Estadísticas de distribución de clases
        print("📊 DISTRIBUCIÓN DE CLASES:")
        damage_counts = dataset_df['damage_level'].value_counts().sort_index()
        total_samples = len(dataset_df)
        
        for damage_level, count in damage_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   {damage_level}: {count:,} muestras ({percentage:.1f}%)")
        
        print()
        
        # Estadísticas de especímenes
        specimens = dataset_df['specimen'].nunique()
        print(f"🧪 Especímenes procesados: {specimens}")
        summary_path = str(output_path).replace('.csv', '_summary.txt')
        print(f"📁 Archivo resumen: {summary_path}")
        print()
        
        # Información para siguiente paso
        print("🔗 SIGUIENTES PASOS:")
        print("   1. Balancear dataset (opcional):")
        print(f"      python src/exp1/2_balance_data.py --input {output_path}")
        print("   2. Entrenar modelo:")
        print(f"      python src/exp1/3_train_dcnn.py --input {output_path}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())