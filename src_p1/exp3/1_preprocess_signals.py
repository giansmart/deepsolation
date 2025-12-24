#!/usr/bin/env python3
"""
Script de Preprocesamiento de Señales - Experimento 3
=====================================================

Genera el dataset procesado base para posterior balanceamiento con augmentación conservadora.
Sigue la metodología Yu et al. (2018) idéntica a exp2:

Approach de exp3:
- Etiquetas a nivel de specimen-sensor completo  
- Una muestra = matriz completa de frecuencias por sensor
- Dataset base para posterior augmentación conservadora
- Metodología Yu et al. (2018) implementada fielmente

Procesamiento:
- FFT transformation de señales raw
- PSD selection (90% energy threshold optimizado)
- Padding consistente de matrices 
- Export con estructura apropiada para DCNN
- Preparación para augmentación posterior

Uso:
    python3 src/exp3/1_preprocess_signals.py [--output OUTPUT_PATH]

Salidas:
    - results/preprocessed_dataset.csv: Dataset base listo para balanceamiento
    - results/preprocessing_summary.txt: Resumen detallado del preprocesamiento

Pipeline exp3:
    1. 1_preprocess_signals.py → Dataset base (este script)
    2. 2_balance_data.py → Augmentación conservadora 
    3. 3_train_dcnn.py → Entrenamiento con datos balanceados

Diferencias vs exp2:
    - Dataset base idéntico a exp2
    - Preparado para pipeline de augmentación
    - Posteriori balanceamiento conservador

RESULTADO: ~68 muestras base → Balanceamiento posterior a ~204 muestras
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
        description="Preprocesamiento de señales para DCNN - Experimento 3 (base para augmentación)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

    # Preprocesamiento básico (recomendado)
    python src/exp3/1_preprocess_signals.py
    
    # Especificar archivo de salida  
    python src/exp3/1_preprocess_signals.py --output src/exp3/results/my_dataset.csv

Pipeline completo exp3:
    1. python src/exp3/1_preprocess_signals.py
    2. python src/exp3/2_balance_data.py --input src/exp3/results/preprocessed_dataset.csv  
    3. python src/exp3/3_train_dcnn.py --input src/exp3/results/balanced_dataset.csv

NOTA: Este script genera el dataset BASE. El balanceamiento se aplica posteriormente
con augmentación conservadora científicamente justificada.
        """
    )
    
    parser.add_argument(
        "--output", 
        default="src/exp3/results/preprocessed_dataset.csv",
        help="Ruta del archivo de salida (default: src/exp3/results/preprocessed_dataset.csv)"
    )
    
    parser.add_argument(
        "--energy-threshold", 
        type=float,
        default=0.90,
        help="Threshold de energía para selección PSD (default: 0.90 - optimizado según Yu et al. 2018)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar información detallada del proceso"
    )
    
    args = parser.parse_args()
    
    try:
        print("="*70)
        print("PREPROCESAMIENTO DE SEÑALES - EXPERIMENTO 3")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 Metodología: Yu et al. (2018) - Approach por matriz completa")
        print(f"📊 Etiquetas a nivel specimen-sensor completo")
        print(f"🎯 Objetivo: Evitar data leakage con GroupKFold")
        print()
        
        # Determinar archivo de salida
        project_root = Path(__file__).parent.parent.parent  # deepsolation/
        if args.output:
            output_path = project_root / args.output
        else:
            output_path = project_root / "src/exp3/results/preprocessed_dataset.csv"
            
        # Crear directorio de salida si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configurar preprocessor
        print(f"📂 Archivo de salida: {output_path}")
        print(f"🎯 Energy threshold: {args.energy_threshold}")
        print()
        
        # Inicializar preprocessor con threshold optimizado
        preprocessor = SignalPreprocessor(
            sampling_rate=100,
            energy_threshold=args.energy_threshold  # 0.90 por defecto según Yu et al. (2018)
        )
        
        # Procesar señales
        print("🔄 Iniciando preprocesamiento...")
        start_time = time.time()
        
        # Directorio de señales (heredado de estructura del proyecto)
        signals_dir = project_root / "data" / "Signals_Raw"
        
        if not signals_dir.exists():
            raise FileNotFoundError(f"Directorio de señales no encontrado: {signals_dir}")
        
        print(f"📁 Directorio de señales: {signals_dir}")
        
        # Procesar todas las señales
        processed_signals = preprocessor.process_all_signals(
            signals_dir=str(signals_dir),
            output_dir=str(output_path.parent)
        )
        
        processing_time = time.time() - start_time
        print(f"✓ Preprocesamiento completado en {processing_time:.1f} segundos")
        
        # Cargar labels para exportación
        labels_path = project_root / "data" / "nivel_damage.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Archivo de labels no encontrado: {labels_path}")
        
        import pandas as pd
        labels_df = pd.read_csv(labels_path)
        
        # Exportar a CSV
        print("📤 Exportando a CSV...")
        export_start = time.time()
        
        csv_summary = preprocessor.export_to_csv(
            processed_signals, 
            labels_df,
            output_path
        )
        
        export_time = time.time() - export_start
        print(f"✓ Exportación completada en {export_time:.1f} segundos")
        
        # Generar resumen
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        
        # Analizar dataset generado para resumen
        if output_path.exists():
            dataset_df = pd.read_csv(output_path)
            total_samples = len(dataset_df)
            specimens = dataset_df['specimen'].nunique() if 'specimen' in dataset_df.columns else 0
            damage_dist = dataset_df['damage_level'].value_counts().sort_index() if 'damage_level' in dataset_df.columns else {}
        else:
            total_samples = specimens = 0
            damage_dist = {}
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\\n")
            f.write("TRAINING DATASET EXPORT SUMMARY - EXPERIMENTO 3\\n")
            f.write("Generated for DCNN Training Analysis (Base para Augmentación)\\n")
            f.write("=" * 80 + "\\n\\n")
            f.write(f"Export Date: {datetime.now()}\\n")
            f.write(f"Methodology: Yu et al. (2018) - FFT + PSD selection\\n")
            f.write(f"Data Leakage Prevention: Dataset base para augmentación posterior\\n\\n")
            
            f.write("DATASET COMPOSITION:\\n")
            f.write("-" * 30 + "\\n")
            f.write(f"Total specimens: {specimens}\\n")
            f.write(f"Total observations: {total_samples:,}\\n")
            
            if len(damage_dist) > 0:
                f.write("\\nDamage level distribution (NATURAL):\\n")
                for damage_level, count in damage_dist.items():
                    percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                    f.write(f"  {damage_level}: {count:,} observations ({percentage:.1f}%)\\n")
            
            f.write("\\n" + "=" * 50 + "\\n")
            f.write("DISTRIBUCIÓN NATURAL DE CLASES\\n")
            f.write("=" * 50 + "\\n")
            f.write("NOTA: Dataset base para augmentación conservadora posterior\\n")
            f.write("El balanceamiento se aplicará con técnicas conservadoras en 2_balance_data.py\\n\\n")
            
            f.write("VENTAJAS DE ESTE APPROACH EXP3:\\n")
            f.write("- No data leakage entre specimens físicos\\n") 
            f.write("- Evaluación realista con GroupKFold\\n")
            f.write("- Distribución natural como baseline\\n")
            f.write("- Base sólida para augmentación conservadora\\n")
            f.write("- Métricas de performance confiables\\n\\n")
            
            f.write("=" * 80 + "\\n")
            f.write("END OF SUMMARY\\n")
            f.write("=" * 80 + "\\n")
        
        print("\\n" + "="*70)
        print("🎉 PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"📊 Dataset generado: {output_path}")
        print(f"📋 Resumen: {summary_path}")
        print(f"⏱️ Tiempo total: {processing_time + export_time:.1f} segundos")
        print()
        print("🔍 APPROACH EXP3 IMPLEMENTADO:")
        print("   ✅ Dataset base con metodología Yu et al. (2018)")
        print("   ✅ Una muestra = matriz completa por (specimen, sensor)")
        print("   ✅ ~68 muestras base (approach por matriz completa)")
        print("   ✅ Etiquetas a nivel de aislador completo") 
        print("   ✅ Dataset base para augmentación conservadora")
        print("   ✅ Preparado para pipeline: preprocess → balance → train")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error durante el preprocesamiento: {e}")
        return 1

if __name__ == "__main__":
    exit(main())