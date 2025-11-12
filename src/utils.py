"""
Utilidades para análisis de datasets y conteos
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter

def analyze_dataset_structure(csv_path):
    """
    Analiza la estructura del dataset CSV y proporciona estadísticas
    
    Args:
        csv_path (str): Ruta al archivo CSV
        
    Returns:
        dict: Diccionario con estadísticas del dataset
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Análisis básico
        unique_damage_levels = df['Nivel_Dano'].unique()
        damage_distribution = df['Nivel_Dano'].value_counts()
        
        # Estadísticas
        stats = {
            'shape': df.shape,
            'total_records': len(df),
            'damage_levels': unique_damage_levels,
            'damage_distribution': damage_distribution,
            'damage_percentages': {
                level: (count / len(df)) * 100 
                for level, count in damage_distribution.items()
            }
        }
        
        return stats, df
        
    except FileNotFoundError:
        return None, None

def count_signal_files(signals_dir):
    """
    Cuenta archivos de señales y estima número de aisladores únicos
    
    Args:
        signals_dir (str): Directorio con señales raw
        
    Returns:
        dict: Estadísticas de archivos de señales
    """
    if not os.path.exists(signals_dir):
        return None
    
    # Buscar todos los archivos de señales
    signal_files = []
    isolator_dirs = []
    
    for root, dirs, files in os.walk(signals_dir):
        for file in files:
            if file.startswith('completo_S') and file.endswith('.txt'):
                signal_files.append(os.path.join(root, file))
                # Extraer directorio del aislador (A1, A1-2, etc.)
                isolator_dir = os.path.basename(root)
                if isolator_dir.startswith('A'):
                    isolator_dirs.append(isolator_dir)
    
    # Contar aisladores únicos (A1, A2, etc. sin variaciones -2, -3)
    unique_isolators = set()
    for isolator in isolator_dirs:
        # Extraer base (A1 de A1-2)
        base_isolator = isolator.split('-')[0]
        unique_isolators.add(base_isolator)
    
    # Contar sensores por aislador
    isolator_counter = Counter(isolator_dirs)
    
    stats = {
        'total_signal_files': len(signal_files),
        'total_isolator_experiments': len(set(isolator_dirs)),
        'unique_isolators': len(unique_isolators),
        'unique_isolator_list': sorted(list(unique_isolators)),
        'experiments_per_isolator': dict(isolator_counter),
        'sensors_per_experiment': len([f for f in signal_files if 'S1' in f or 'S2' in f])
    }
    
    return stats

def print_dataset_summary(csv_stats, signal_stats):
    """
    Imprime resumen completo del dataset
    
    Args:
        csv_stats (dict): Estadísticas del CSV
        signal_stats (dict): Estadísticas de señales
    """
    if csv_stats:
        print(f"✓ Dataset loaded: {csv_stats['shape']}")
        print(f"  Damage levels: {csv_stats['damage_levels']}")
        print(f"  Total experimental records: {csv_stats['total_records']}")
        
        if signal_stats:
            print(f"  Unique physical isolators: {signal_stats['unique_isolators']} ({', '.join(signal_stats['unique_isolator_list'][:5])}{'...' if len(signal_stats['unique_isolator_list']) > 5 else ''})")
            print(f"  Total signal files available: {signal_stats['total_signal_files']}")
        
        print(f"  Damage distribution:")
        for level in csv_stats['damage_levels']:
            count = csv_stats['damage_distribution'][level]
            percentage = csv_stats['damage_percentages'][level]
            print(f"    {level}: {count} records ({percentage:.1f}%)")
    else:
        print("✗ Could not load dataset statistics")

def create_signal_analysis_report(signal_file, feature_matrix, metadata, output_path):
    """
    Crea un reporte detallado de la transformación de señales
    
    Args:
        signal_file (str): Ruta del archivo de señal original
        feature_matrix (numpy.array): Matriz de características procesada
        metadata (dict): Metadatos del procesamiento
        output_path (str): Ruta donde guardar el reporte
    """
    import numpy as np
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report_lines = [
        "="*80,
        "SIGNAL TRANSFORMATION ANALYSIS REPORT",
        "Yu et al. (2018) Methodology Implementation",
        "="*80,
        "",
        f"Source File: {signal_file}",
        f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "="*50,
        "1. ORIGINAL SIGNAL CHARACTERISTICS",
        "="*50,
        f"Original samples: ~{metadata.get('original_samples', 'Unknown')}",
        f"Sampling rate: {metadata.get('sampling_rate', 'Unknown')} Hz",
        f"Duration: ~{metadata.get('duration_seconds', 'Unknown')} seconds",
        f"Axes: {metadata.get('num_axes', 3)} (N-S, E-W, U-D)",
        "",
        "="*50,
        "2. FFT TRANSFORMATION",
        "="*50,
        f"FFT components: {metadata.get('fft_components', 'Unknown')}",
        f"Frequency resolution: {metadata.get('freq_resolution', 'Unknown')} Hz",
        f"Nyquist frequency: {metadata.get('nyquist_freq', 'Unknown')} Hz",
        "",
        "="*50,
        "3. POWER SPECTRAL DENSITY (PSD) SELECTION",
        "="*50,
        f"Energy threshold: {metadata.get('energy_threshold', 0.7)*100}%",
        f"Selected components: {feature_matrix.shape[0] if feature_matrix is not None else 'Error'}",
        f"Energy retained: {metadata.get('energy_retained', 'Unknown')*100 if metadata.get('energy_retained') else 'Unknown'}%",
        "",
        "="*50,
        "4. COMPRESSION RESULTS",
        "="*50,
        f"Compression ratio: {metadata.get('compression_ratio', 'Unknown')}",
        f"Data reduction: {(1 - metadata.get('compression_ratio', 0))*100 if metadata.get('compression_ratio') else 'Unknown'}%",
        f"Final matrix shape: {feature_matrix.shape if feature_matrix is not None else 'Error'}",
        "",
        "="*50,
        "5. FREQUENCY COMPONENTS SELECTED",
        "="*50,
    ]
    
    # Agregar información sobre componentes de frecuencia seleccionados
    if 'selected_frequencies' in metadata and metadata['selected_frequencies'] is not None:
        selected_freqs = metadata['selected_frequencies']
        report_lines.extend([
            f"Frequency range: {selected_freqs.min():.2f} - {selected_freqs.max():.2f} Hz",
            f"Most important frequencies (top 10):",
        ])
        
        # Mostrar las 10 frecuencias más importantes (si hay suficientes)
        top_freqs = selected_freqs[:min(10, len(selected_freqs))]
        for i, freq in enumerate(top_freqs):
            report_lines.append(f"  {i+1:2d}. {freq:.2f} Hz")
    
    report_lines.extend([
        "",
        "="*50,
        "6. QUALITY ASSESSMENT",
        "="*50,
        f"Compression quality: {'Excellent' if metadata.get('compression_ratio', 1) < 0.3 else 'Good' if metadata.get('compression_ratio', 1) < 0.5 else 'Moderate'}",
        f"Data reduction: {'High' if metadata.get('compression_ratio', 1) < 0.3 else 'Medium' if metadata.get('compression_ratio', 1) < 0.5 else 'Low'}",
        f"Information retained: {'Sufficient' if metadata.get('energy_retained', 0) >= 0.7 else 'May be insufficient'}",
        "",
        "="*50,
        "7. METHODOLOGY NOTES",
        "="*50,
        "• FFT transforms time-domain signals to frequency-domain",
        "• PSD selection retains components with highest energy content",
        "• 70% energy threshold balances compression vs. information loss",
        "• Selected frequencies become input features for DCNN",
        "• Automatic feature extraction eliminates manual engineering",
        "",
        "="*80,
        "END OF REPORT",
        "="*80
    ])
    
    # Escribir reporte
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Signal analysis report saved: {output_path}")

def export_training_dataset(processed_signals, csv_data, output_path, padding_info=None):
    """
    Exporta dataset completo con features y etiquetas para trazabilidad
    
    Args:
        processed_signals (dict): Señales procesadas por el preprocessor
        csv_data (DataFrame): Dataset con etiquetas
        output_path (str): Ruta donde guardar el dataset
        padding_info (dict): Información sobre padding aplicado
    
    Returns:
        str: Ruta del archivo exportado
    """
    import numpy as np
    import pickle
    from datetime import datetime
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Crear mapeo de etiquetas (igual que en prepare_data_for_pytorch)
    damage_mapping = {}
    for i in range(15):  # Solo primeros 15 aisladores únicos
        if i < len(csv_data):
            specimen_id = f"A{i+1}"
            damage_mapping[specimen_id] = csv_data.iloc[i]['Nivel_Dano']
            # También mapear variaciones experimentales
            damage_mapping[f"{specimen_id}-2"] = csv_data.iloc[i]['Nivel_Dano']
            damage_mapping[f"{specimen_id}-3"] = csv_data.iloc[i]['Nivel_Dano']
    
    # Recolectar datos de entrenamiento
    training_data = []
    
    for specimen, sensors in processed_signals.items():
        if specimen in damage_mapping and 'S1' in sensors and 'S2' in sensors:
            damage_level = damage_mapping[specimen]
            s1_features = sensors['S1']['features']
            s2_features = sensors['S2']['features']
            s1_metadata = sensors['S1']['metadata']
            s2_metadata = sensors['S2']['metadata']
            
            # Crear registro de entrenamiento
            training_record = {
                'specimen_id': specimen,
                'damage_level': damage_level,
                'original_signal_files': {
                    'S1': s1_metadata.get('file_path', 'Unknown'),
                    'S2': s2_metadata.get('file_path', 'Unknown')
                },
                'preprocessing_metadata': {
                    'S1': {
                        'original_samples': s1_metadata.get('original_samples'),
                        'selected_components': s1_features.shape[0],
                        'compression_ratio': s1_metadata.get('compression_ratio'),
                        'energy_retained': s1_metadata.get('energy_retained'),
                        'frequency_range': s1_metadata.get('frequency_range')
                    },
                    'S2': {
                        'original_samples': s2_metadata.get('original_samples'),
                        'selected_components': s2_features.shape[0],
                        'compression_ratio': s2_metadata.get('compression_ratio'),
                        'energy_retained': s2_metadata.get('energy_retained'),
                        'frequency_range': s2_metadata.get('frequency_range')
                    }
                },
                'features': {
                    'S1': s1_features,  # Array de características
                    'S2': s2_features   # Array de características
                },
                'padded_shape': None  # Se llenará después del padding
            }
            
            training_data.append(training_record)
    
    # Crear dataset final con metadata completo
    dataset_export = {
        'metadata': {
            'export_timestamp': datetime.now().isoformat(),
            'total_specimens': len(training_data),
            'unique_damage_levels': list(set([record['damage_level'] for record in training_data])),
            'preprocessing_params': {
                'sampling_rate': 100,
                'energy_threshold': 0.7,
                'methodology': 'Yu et al. (2018) - FFT + PSD selection'
            },
            'padding_info': padding_info,
            'data_source': {
                'csv_file': 'Arreglo_3_actual_clean.csv',
                'signals_directory': 'data/Signals_Raw/',
                'methodology_paper': 'Yu et al. (2018) - Deep Convolutional Neural Network'
            }
        },
        'training_data': training_data,
        'damage_mapping': damage_mapping
    }
    
    # Guardar como pickle para preservar arrays numpy
    with open(output_path, 'wb') as f:
        pickle.dump(dataset_export, f)
    
    # Crear también un resumen en texto legible
    summary_path = output_path.replace('.pkl', '_summary.txt')
    create_dataset_summary(dataset_export, summary_path)
    
    # Crear dataset CSV legible con características aplanadas
    csv_path = output_path.replace('.pkl', '_features.csv')
    create_csv_dataset(dataset_export, csv_path)
    
    # Crear también un CSV compacto (una fila por espécimen)
    compact_csv_path = output_path.replace('.pkl', '_specimens.csv')
    create_compact_csv(dataset_export, compact_csv_path)
    
    return output_path, summary_path, csv_path

def create_dataset_summary(dataset_export, output_path):
    """
    Crea un resumen legible del dataset exportado
    
    Args:
        dataset_export (dict): Dataset exportado
        output_path (str): Ruta para guardar resumen
    """
    metadata = dataset_export['metadata']
    training_data = dataset_export['training_data']
    
    # Calcular estadísticas
    damage_counts = {}
    total_features = 0
    compression_stats = []
    
    for record in training_data:
        # Contar niveles de daño
        damage_level = record['damage_level']
        damage_counts[damage_level] = damage_counts.get(damage_level, 0) + 1
        
        # Estadísticas de compresión
        s1_compression = record['preprocessing_metadata']['S1']['compression_ratio']
        s2_compression = record['preprocessing_metadata']['S2']['compression_ratio']
        compression_stats.extend([s1_compression, s2_compression])
        
        # Contar características totales
        s1_features = record['features']['S1'].shape[0] * record['features']['S1'].shape[1]
        s2_features = record['features']['S2'].shape[0] * record['features']['S2'].shape[1]
        total_features += s1_features + s2_features
    
    summary_lines = [
        "="*80,
        "TRAINING DATASET EXPORT SUMMARY",
        "Generated for DCNN Training Traceability",
        "="*80,
        "",
        f"Export Date: {metadata['export_timestamp']}",
        f"Methodology: {metadata['preprocessing_params']['methodology']}",
        "",
        "="*50,
        "DATASET COMPOSITION",
        "="*50,
        f"Total training specimens: {metadata['total_specimens']}",
        f"Damage level distribution:",
    ]
    
    # Distribución de daño
    for damage_level, count in damage_counts.items():
        percentage = (count / len(training_data)) * 100
        summary_lines.append(f"  {damage_level}: {count} specimens ({percentage:.1f}%)")
    
    summary_lines.extend([
        "",
        "="*50,
        "PREPROCESSING STATISTICS", 
        "="*50,
        f"Sampling rate: {metadata['preprocessing_params']['sampling_rate']} Hz",
        f"Energy threshold: {metadata['preprocessing_params']['energy_threshold']*100}%",
        f"Average compression ratio: {np.mean(compression_stats):.3f}",
        f"Compression range: {np.min(compression_stats):.3f} - {np.max(compression_stats):.3f}",
        f"Total features extracted: {total_features:,}",
        "",
        "="*50,
        "SPECIMEN DETAILS",
        "="*50,
    ])
    
    # Detalles por espécimen
    for record in training_data:
        specimen = record['specimen_id']
        damage = record['damage_level']
        s1_shape = record['features']['S1'].shape
        s2_shape = record['features']['S2'].shape
        s1_compression = record['preprocessing_metadata']['S1']['compression_ratio']
        s2_compression = record['preprocessing_metadata']['S2']['compression_ratio']
        
        summary_lines.append(
            f"{specimen:8} | {damage:2} | S1: {s1_shape} ({s1_compression:.3f}) | S2: {s2_shape} ({s2_compression:.3f})"
        )
    
    summary_lines.extend([
        "",
        "="*50,
        "FILES INCLUDED",
        "="*50,
        "Signal files processed:",
    ])
    
    # Listar archivos fuente
    for record in training_data:
        specimen = record['specimen_id']
        s1_file = os.path.basename(record['original_signal_files']['S1'])
        s2_file = os.path.basename(record['original_signal_files']['S2'])
        summary_lines.append(f"  {specimen}: {s1_file}, {s2_file}")
    
    summary_lines.extend([
        "",
        "="*50,
        "USAGE NOTES",
        "="*50,
        "• This dataset contains processed features ready for DCNN training",
        "• Features are FFT-transformed frequency components (Yu et al. methodology)",
        "• Padding may be applied to equalize matrix dimensions",
        "• Each specimen has S1 and S2 sensor data combined",
        "• Damage levels: N1 (no damage), N2 (moderate), N3 (severe)",
        "",
        "="*80,
        "END OF SUMMARY",
        "="*80
    ])
    
    # Escribir resumen
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

def create_csv_dataset(dataset_export, output_path):
    """
    Crea un dataset CSV legible con características aplanadas
    
    Args:
        dataset_export (dict): Dataset exportado
        output_path (str): Ruta para guardar CSV
    """
    training_data = dataset_export['training_data']
    
    # Lista para almacenar filas del CSV
    csv_rows = []
    
    for record in training_data:
        specimen_id = record['specimen_id']
        damage_level = record['damage_level']
        
        # Obtener características de ambos sensores
        s1_features = record['features']['S1']  # Shape: (freq_components, 3)
        s2_features = record['features']['S2']  # Shape: (freq_components, 3)
        
        # Aplanar características para CSV (cada fila = 1 componente de frecuencia)
        for freq_idx in range(s1_features.shape[0]):
            # Crear fila con metadatos + características
            row = {
                'specimen_id': specimen_id,
                'damage_level': damage_level,
                'frequency_component': freq_idx,
                # Sensor S1 - 3 ejes
                'S1_NS': s1_features[freq_idx, 0],  # Norte-Sur
                'S1_EW': s1_features[freq_idx, 1],  # Este-Oeste
                'S1_UD': s1_features[freq_idx, 2],  # Up-Down
                # Sensor S2 - 3 ejes  
                'S2_NS': s2_features[freq_idx, 0],  # Norte-Sur
                'S2_EW': s2_features[freq_idx, 1],  # Este-Oeste
                'S2_UD': s2_features[freq_idx, 2],  # Up-Down
            }
            
            # Agregar metadatos de preprocessing
            row.update({
                'S1_original_samples': record['preprocessing_metadata']['S1']['original_samples'],
                'S1_compression_ratio': record['preprocessing_metadata']['S1']['compression_ratio'],
                'S1_energy_retained': record['preprocessing_metadata']['S1']['energy_retained'],
                'S2_original_samples': record['preprocessing_metadata']['S2']['original_samples'],
                'S2_compression_ratio': record['preprocessing_metadata']['S2']['compression_ratio'],
                'S2_energy_retained': record['preprocessing_metadata']['S2']['energy_retained'],
                'source_S1_file': os.path.basename(record['original_signal_files']['S1']),
                'source_S2_file': os.path.basename(record['original_signal_files']['S2'])
            })
            
            csv_rows.append(row)
    
    # Crear DataFrame y guardar
    df_csv = pd.DataFrame(csv_rows)
    
    # Reordenar columnas para mejor legibilidad
    column_order = [
        'specimen_id', 'damage_level', 'frequency_component',
        'S1_NS', 'S1_EW', 'S1_UD', 'S2_NS', 'S2_EW', 'S2_UD',
        'S1_original_samples', 'S1_compression_ratio', 'S1_energy_retained',
        'S2_original_samples', 'S2_compression_ratio', 'S2_energy_retained',
        'source_S1_file', 'source_S2_file'
    ]
    
    df_csv = df_csv[column_order]
    
    # Guardar CSV
    df_csv.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"✓ CSV dataset exported: {output_path}")
    print(f"  Total rows: {len(csv_rows):,}")
    print(f"  Columns: {len(column_order)}")
    print(f"  Format: Each row = 1 frequency component with S1+S2 data")

def create_compact_csv(dataset_export, output_path):
    """
    Crea un CSV compacto con una fila por espécimen (metadatos solamente)
    
    Args:
        dataset_export (dict): Dataset exportado
        output_path (str): Ruta para guardar CSV compacto
    """
    training_data = dataset_export['training_data']
    
    # Lista para almacenar filas del CSV compacto
    compact_rows = []
    
    for record in training_data:
        # Crear fila con metadatos únicamente (sin características numéricas)
        row = {
            'specimen_id': record['specimen_id'],
            'damage_level': record['damage_level'],
            # Información S1
            'S1_original_samples': record['preprocessing_metadata']['S1']['original_samples'],
            'S1_selected_components': record['preprocessing_metadata']['S1']['selected_components'],
            'S1_compression_ratio': record['preprocessing_metadata']['S1']['compression_ratio'],
            'S1_energy_retained': record['preprocessing_metadata']['S1']['energy_retained'],
            'S1_freq_range_min': record['preprocessing_metadata']['S1']['frequency_range'][0],
            'S1_freq_range_max': record['preprocessing_metadata']['S1']['frequency_range'][1],
            # Información S2
            'S2_original_samples': record['preprocessing_metadata']['S2']['original_samples'],
            'S2_selected_components': record['preprocessing_metadata']['S2']['selected_components'],
            'S2_compression_ratio': record['preprocessing_metadata']['S2']['compression_ratio'],
            'S2_energy_retained': record['preprocessing_metadata']['S2']['energy_retained'],
            'S2_freq_range_min': record['preprocessing_metadata']['S2']['frequency_range'][0],
            'S2_freq_range_max': record['preprocessing_metadata']['S2']['frequency_range'][1],
            # Archivos fuente
            'source_S1_file': os.path.basename(record['original_signal_files']['S1']),
            'source_S2_file': os.path.basename(record['original_signal_files']['S2']),
            # Información de características finales
            'final_feature_matrix_rows': record['features']['S1'].shape[0],  # Después del padding
            'final_feature_matrix_cols': record['features']['S1'].shape[1] + record['features']['S2'].shape[1],  # S1+S2
        }
        
        compact_rows.append(row)
    
    # Crear DataFrame compacto
    df_compact = pd.DataFrame(compact_rows)
    
    # Guardar CSV
    df_compact.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"✓ Compact CSV exported: {output_path}")
    print(f"  Total specimens: {len(compact_rows)}")
    print(f"  Format: One row per specimen with preprocessing metadata")