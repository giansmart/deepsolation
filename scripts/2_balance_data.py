#!/usr/bin/env python3
"""
Script de Balanceo de Dataset con SMOTE
======================================

Balancea el dataset procesado usando SMOTE (Synthetic Minority Oversampling Technique)
para mejorar el rendimiento en clases minoritarias (N2, N3).

Entrada:
    - CSV procesado por 1_preprocess_signals.py
    
Salida:
    - CSV balanceado listo para entrenamiento
    - Reporte de balanceo con estadísticas

Uso:
    python scripts/2_balance_data.py --input results/preprocessed_dataset.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import Counter

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Importar SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

def check_dependencies():
    """Verificar que SMOTE esté disponible"""
    if not SMOTE_AVAILABLE:
        print("❌ Error: imbalanced-learn no está instalado")
        print("Instálalo con: pip install imbalanced-learn")
        return False
    return True

def load_and_validate_dataset(input_path):
    """Cargar y validar dataset CSV"""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {input_path}")
    
    print(f"📂 Cargando dataset: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        
        # Validar columnas requeridas
        required_cols = ['specimen', 'sensor', 'component_NS', 'component_EW', 'component_UD', 'damage_level']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Columnas faltantes en dataset: {missing_cols}")
        
        print(f"✓ Dataset cargado: {len(df):,} observaciones")
        print(f"✓ Especímenes: {df['specimen'].nunique()}")
        
        # Mostrar distribución de clases original
        print("\n📊 DISTRIBUCIÓN ORIGINAL:")
        damage_counts = df['damage_level'].value_counts().sort_index()
        total_samples = len(df)
        
        for damage_level, count in damage_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   {damage_level}: {count:,} ({percentage:.1f}%)")
        
        return df
        
    except Exception as e:
        raise Exception(f"Error cargando dataset: {e}")

def prepare_data_for_smote(df):
    """Preparar datos para SMOTE agrupando por espécimen"""
    print("\n🔧 Preparando datos para SMOTE...")
    
    # Agrupar por espécimen para crear matrices de características
    specimens_data = []
    specimens_labels = []
    specimen_ids = []
    
    for specimen in df['specimen'].unique():
        specimen_data = df[df['specimen'] == specimen]
        damage_level = specimen_data['damage_level'].iloc[0]
        
        # Combinar datos de ambos sensores
        sensors_data = []
        for sensor in ['S1', 'S2']:
            sensor_data = specimen_data[specimen_data['sensor'] == sensor]
            if not sensor_data.empty:
                # Extraer componentes NS, EW, UD
                components = sensor_data[['component_NS', 'component_EW', 'component_UD']].values
                # Aplanar para SMOTE: usar estadísticas agregadas por sensor
                stats = np.array([
                    np.mean(components, axis=0),  # Media
                    np.std(components, axis=0),   # Desviación estándar  
                    np.max(components, axis=0),   # Máximo
                    np.min(components, axis=0),   # Mínimo
                    np.median(components, axis=0) # Mediana
                ]).flatten()
                sensors_data.append(stats)
        
        if len(sensors_data) == 2:  # Solo si tenemos ambos sensores
            # Concatenar estadísticas de ambos sensores
            combined_stats = np.concatenate(sensors_data)
            specimens_data.append(combined_stats)
            specimens_labels.append(damage_level)
            specimen_ids.append(specimen)
    
    if not specimens_data:
        raise ValueError("No se pudieron preparar los datos para SMOTE")
    
    X = np.array(specimens_data)
    y = np.array(specimens_labels)
    
    print(f"✓ Datos preparados: {X.shape[0]} especímenes con {X.shape[1]} características estadísticas")
    print(f"✓ Distribución antes de SMOTE: {dict(Counter(y))}")
    
    return X, y, specimen_ids

def apply_smote(X, y, strategy='auto', k_neighbors=5):
    """Aplicar SMOTE para balancear clases"""
    print(f"\n⚖️ Aplicando SMOTE...")
    print(f"Estrategia: {strategy}")
    
    # Ajustar k_neighbors automáticamente según la clase minoritaria
    class_counts = Counter(y)
    min_samples = min(class_counts.values())
    
    # k_neighbors debe ser menor que el número de muestras en la clase minoritaria
    max_k_neighbors = min_samples - 1
    adjusted_k_neighbors = min(k_neighbors, max_k_neighbors)
    
    if adjusted_k_neighbors < 1:
        # Si no hay suficientes muestras para SMOTE, usar duplicación simple
        print(f"⚠️ Clase minoritaria muy pequeña ({min_samples} muestras)")
        print(f"⚠️ Usando duplicación simple en lugar de SMOTE")
        return apply_simple_duplication(X, y, strategy)
    
    if adjusted_k_neighbors != k_neighbors:
        print(f"⚠️ K-neighbors ajustado de {k_neighbors} a {adjusted_k_neighbors} (clase minoritaria: {min_samples} muestras)")
    
    print(f"K-neighbors: {adjusted_k_neighbors}")
    
    try:
        # Configurar SMOTE
        smote = SMOTE(
            sampling_strategy=strategy,
            k_neighbors=adjusted_k_neighbors,
            random_state=42
        )
        
        # Aplicar SMOTE
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"✓ SMOTE aplicado exitosamente")
        print(f"  Especímenes originales: {len(X)}")
        print(f"  Especímenes balanceados: {len(X_balanced)}")
        print(f"  Distribución balanceada: {dict(Counter(y_balanced))}")
        
        return X_balanced, y_balanced
        
    except Exception as e:
        raise Exception(f"Error aplicando SMOTE: {e}")

def apply_simple_duplication(X, y, strategy):
    """Aplicar duplicación simple cuando SMOTE no es posible"""
    class_counts = Counter(y)
    
    if strategy == 'auto':
        # Balancear todas las clases al nivel de la clase mayoritaria
        target_count = max(class_counts.values())
    elif strategy == 'minority':
        # Balancear solo la clase minoritaria al nivel de la segunda más pequeña
        sorted_counts = sorted(class_counts.values())
        target_count = sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]
    else:  # not_majority
        # Balancear todas excepto la mayoritaria al nivel de la segunda mayor
        sorted_counts = sorted(class_counts.values(), reverse=True)
        target_count = sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]
    
    X_balanced = []
    y_balanced = []
    
    # Procesar cada clase
    unique_classes = np.unique(y)
    for class_label in unique_classes:
        class_mask = y == class_label
        class_X = X[class_mask]
        class_y = y[class_mask]
        
        current_count = len(class_X)
        
        if strategy == 'not_majority' and current_count == max(class_counts.values()):
            # No duplicar la clase mayoritaria
            needed_samples = current_count
        elif strategy == 'minority' and current_count != min(class_counts.values()):
            # Solo duplicar la clase minoritaria
            needed_samples = current_count
        else:
            needed_samples = target_count
        
        # Agregar todas las muestras originales
        X_balanced.extend(class_X)
        y_balanced.extend(class_y)
        
        # Duplicar muestras si es necesario
        if needed_samples > current_count:
            samples_to_add = needed_samples - current_count
            
            # Duplicar de forma cíclica
            for i in range(samples_to_add):
                idx = i % current_count
                # Agregar pequeña variación para evitar duplicados exactos
                sample = class_X[idx].copy()
                noise = np.random.normal(0, np.std(sample) * 0.01, sample.shape)
                sample_with_noise = sample + noise
                
                X_balanced.append(sample_with_noise)
                y_balanced.append(class_label)
    
    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)
    
    print(f"✓ Duplicación simple aplicada")
    print(f"  Especímenes originales: {len(X)}")
    print(f"  Especímenes balanceados: {len(X_balanced)}")
    print(f"  Distribución balanceada: {dict(Counter(y_balanced))}")
    
    return X_balanced, y_balanced

def create_balanced_dataset(df_original, X_balanced, y_balanced, specimen_ids):
    """Crear dataset balanceado expandiendo especímenes sintéticos"""
    print("\n📝 Creando dataset balanceado...")
    
    # Separar datos originales y sintéticos
    n_original = len(specimen_ids)
    
    # Datos originales (conservar tal como están)
    original_mask = list(range(n_original))
    synthetic_mask = list(range(n_original, len(X_balanced)))
    
    balanced_rows = []
    
    # 1. Agregar todos los datos originales
    for i in original_mask:
        specimen_id = specimen_ids[i]
        specimen_data = df_original[df_original['specimen'] == specimen_id]
        
        # Agregar todas las filas del espécimen original
        for _, row in specimen_data.iterrows():
            balanced_rows.append(row.to_dict())
    
    print(f"✓ Agregados {len(balanced_rows)} observaciones originales")
    
    # 2. Generar especímenes sintéticos
    synthetic_count = 0
    for i in synthetic_mask:
        damage_level = y_balanced[i]
        
        # Buscar un espécimen original de la misma clase como plantilla
        template_specimens = df_original[df_original['damage_level'] == damage_level]['specimen'].unique()
        if len(template_specimens) == 0:
            continue
            
        # Usar el primer espécimen como plantilla
        template_specimen = template_specimens[0]
        template_data = df_original[df_original['specimen'] == template_specimen]
        
        # Crear espécimen sintético con ID único
        synthetic_id = f"SYN_{damage_level}_{synthetic_count:03d}"
        synthetic_count += 1
        
        # Agregar pequeña variación sintética a los datos de la plantilla
        variation_factor = 0.05  # 5% de variación
        
        for _, row in template_data.iterrows():
            synthetic_row = row.to_dict()
            synthetic_row['specimen'] = synthetic_id
            
            # Agregar variación pequeña a los componentes
            for component in ['component_NS', 'component_EW', 'component_UD']:
                original_value = synthetic_row[component]
                noise = np.random.normal(0, abs(original_value) * variation_factor)
                synthetic_row[component] = original_value + noise
            
            balanced_rows.append(synthetic_row)
    
    print(f"✓ Generados {synthetic_count} especímenes sintéticos")
    
    # Crear DataFrame balanceado
    df_balanced = pd.DataFrame(balanced_rows)
    
    print(f"✓ Dataset balanceado creado: {len(df_balanced):,} observaciones")
    
    return df_balanced

def save_balanced_dataset(df_balanced, output_path):
    """Guardar dataset balanceado y generar reporte"""
    output_path = Path(output_path)
    
    # Guardar CSV
    df_balanced.to_csv(output_path, index=False)
    print(f"✓ Dataset balanceado guardado: {output_path}")
    
    # Generar reporte
    report_path = str(output_path).replace('.csv', '_balance_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE BALANCEO CON SMOTE\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Archivo de salida: {output_path}\n\n")
        
        f.write("DISTRIBUCIÓN BALANCEADA:\n")
        f.write("-" * 30 + "\n")
        damage_counts = df_balanced['damage_level'].value_counts().sort_index()
        total_samples = len(df_balanced)
        
        for damage_level, count in damage_counts.items():
            percentage = (count / total_samples) * 100
            f.write(f"{damage_level}: {count:,} observaciones ({percentage:.1f}%)\n")
        
        f.write(f"\nTotal de observaciones: {total_samples:,}\n")
        f.write(f"Especímenes únicos: {df_balanced['specimen'].nunique()}\n")
        
        # Estadísticas de especímenes sintéticos
        synthetic_specimens = df_balanced[df_balanced['specimen'].str.startswith('SYN_')]['specimen'].nunique()
        f.write(f"Especímenes sintéticos: {synthetic_specimens}\n")
        
        f.write("\nMETODOLOGÍA:\n")
        f.write("-" * 30 + "\n")
        f.write("1. SMOTE aplicado a nivel de espécimen\n")
        f.write("2. Características estadísticas por sensor\n")
        f.write("3. Generación sintética con 5% de variación\n")
        f.write("4. Conservación de datos originales\n")
    
    print(f"✓ Reporte de balanceo: {report_path}")
    
    return report_path

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Balanceo de dataset con SMOTE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

    # Balanceo estándar
    python scripts/2_balance_data.py --input results/preprocessed_dataset.csv
    
    # Especificar archivo de salida
    python scripts/2_balance_data.py --input results/preprocessed_dataset.csv \\
                                     --output results/balanced_dataset.csv
    
    # Configurar parámetros SMOTE
    python scripts/2_balance_data.py --input results/preprocessed_dataset.csv \\
                                     --k-neighbors 3 --strategy minority

Estrategias disponibles:
    - auto: Balancea automáticamente todas las clases
    - minority: Sobremuestrea solo la clase minoritaria
    - not_majority: Sobremuestrea todas las clases excepto la mayoritaria
        """
    )
    
    parser.add_argument(
        "--input", 
        required=True,
        help="Ruta del dataset procesado (CSV)"
    )
    parser.add_argument(
        "--output", 
        help="Ruta del dataset balanceado (default: auto basado en input)"
    )
    parser.add_argument(
        "--strategy", 
        choices=['auto', 'minority', 'not_majority'],
        default='auto',
        help="Estrategia de SMOTE (default: auto)"
    )
    parser.add_argument(
        "--k-neighbors", 
        type=int,
        default=5,
        help="Número de k-neighbors para SMOTE (default: 5)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Modo silencioso"
    )
    
    args = parser.parse_args()
    
    try:
        print("="*70)
        print("BALANCEO DE DATASET CON SMOTE")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Verificar dependencias
        if not check_dependencies():
            return 1
        
        # Determinar archivo de salida
        if args.output:
            output_path = Path(args.output)
        else:
            input_path = Path(args.input)
            output_path = input_path.parent / f"{input_path.stem}_balanced.csv"
        
        print(f"📂 Input: {args.input}")
        print(f"📁 Output: {output_path}")
        print()
        
        # Cargar dataset
        df = load_and_validate_dataset(args.input)
        
        # Preparar datos para SMOTE
        X, y, specimen_ids = prepare_data_for_smote(df)
        
        # Aplicar SMOTE
        start_time = time.time()
        X_balanced, y_balanced = apply_smote(X, y, strategy=args.strategy, k_neighbors=args.k_neighbors)
        
        # Crear dataset balanceado
        df_balanced = create_balanced_dataset(df, X_balanced, y_balanced, specimen_ids)
        
        # Guardar resultados
        output_path.parent.mkdir(exist_ok=True)
        report_path = save_balanced_dataset(df_balanced, output_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print()
        print("="*70)
        print("BALANCEO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"⏱️ Tiempo de procesamiento: {processing_time:.1f} segundos")
        print(f"📊 Dataset balanceado: {output_path}")
        print(f"📋 Reporte: {report_path}")
        print()
        
        # Mostrar distribución final
        print("📊 DISTRIBUCIÓN FINAL:")
        damage_counts = df_balanced['damage_level'].value_counts().sort_index()
        total_samples = len(df_balanced)
        
        for damage_level, count in damage_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   {damage_level}: {count:,} ({percentage:.1f}%)")
        
        print()
        print("🔗 SIGUIENTE PASO:")
        print("   Entrenar modelo con datos balanceados:")
        print(f"   python scripts/3_train_dcnn.py --input {output_path}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())