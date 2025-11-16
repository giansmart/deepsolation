#!/usr/bin/env python3
"""
Análisis Comparativo de Separabilidad de Clases - Experimentos 2, 3, 4
=====================================================================

Analiza la separabilidad intrínseca de las clases de daño (N1, N2, N3) 
en los diferentes espacios de características generados por cada experimento:
- Exp2: Bins individuales de FFT (baseline metodológico)
- Exp3: Bins FFT con balanceo SMOTE
- Exp4: Características estadísticas agregadas

Métricas implementadas:
1. PCA - Visualización 2D de clusters por experimento
2. Silhouette Score - Calidad de clustering por clase y experimento
3. Davies-Bouldin Index - Separación inter/intra cluster por experimento

Uso:
    python3 src/comparison/analyze_separability.py

Requisitos:
    - exp2/results/preprocessed_dataset.csv
    - exp3/results/balanced_dataset.csv  
    - exp4/results/preprocessed_dataset.csv

Salidas:
    - results/separability_comparison_across_experiments.png
    - results/separability_metrics_comparison.png
    - results/separability_analysis_report.txt

Este análisis revela qué espacio de características ofrece mejor separabilidad.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de matplotlib
plt.style.use('default')
sns.set_palette("husl")

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

def load_experiments_datasets():
    """Cargar datasets de los tres experimentos"""
    print("📂 Cargando datasets de experimentos...")
    
    # Rutas a los datasets de cada experimento
    script_dir = Path(__file__).parent.parent
    
    exp2_path = script_dir / "exp2" / "results" / "preprocessed_dataset.csv"
    exp3_path = script_dir / "exp3" / "results" / "balanced_dataset.csv"
    exp4_path = script_dir / "exp4" / "results" / "preprocessed_dataset.csv"
    
    datasets = {}
    
    # Cargar Exp2 (baseline)
    if exp2_path.exists():
        datasets['exp2'] = pd.read_csv(exp2_path)
        print(f"✓ Exp2 (Baseline FFT): {len(datasets['exp2'])} muestras")
    else:
        print(f"⚠️ Exp2 dataset no encontrado: {exp2_path}")
    
    # Cargar Exp3 (balanceado)
    if exp3_path.exists():
        datasets['exp3'] = pd.read_csv(exp3_path)
        print(f"✓ Exp3 (FFT Balanceado): {len(datasets['exp3'])} muestras")
    else:
        print(f"⚠️ Exp3 dataset no encontrado: {exp3_path}")
    
    # Cargar Exp4 (agregado)
    if exp4_path.exists():
        datasets['exp4'] = pd.read_csv(exp4_path)
        print(f"✓ Exp4 (Features Agregadas): {len(datasets['exp4'])} muestras")
    else:
        print(f"⚠️ Exp4 dataset no encontrado: {exp4_path}")
    
    if not datasets:
        raise FileNotFoundError("No se encontró ningún dataset de experimentos")
    
    # Mostrar distribuciones por experimento
    for exp_name, df in datasets.items():
        print(f"\n📊 Distribución {exp_name.upper()}:")
        counts = df['damage_level'].value_counts().sort_index()
        for damage_level, count in counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {damage_level}: {count} ({percentage:.1f}%)")
    
    return datasets

def extract_features_and_labels(df, experiment_name):
    """Extraer features y labels según el tipo de experimento"""
    print(f"\n🔄 Procesando features de {experiment_name}...")
    
    # Determinar tipo de columnas según experimento
    if experiment_name == 'exp4':
        # Exp4 usa características agregadas - excluir columnas de metadatos
        exclude_cols = ['device_id', 'specimen', 'sensor', 'damage_level', 'isolator_id', 'measurement_file', 'augmented', 'file_path']
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('signal_') and col != 'duration_seconds']
    else:
        # Exp2 y Exp3 usan columnas de frecuencia
        import re
        freq_pattern = re.compile(r'^freq_\d+_(NS|EW|UD)$')
        feature_cols = [col for col in df.columns if freq_pattern.match(col)]
    
    if len(feature_cols) == 0:
        available_cols = [col for col in df.columns if col != 'damage_level'][:10]  # Mostrar primeras 10
        raise ValueError(f"No se encontraron features en {experiment_name}. Columnas disponibles: {available_cols}...")
    
    # Extraer features y labels
    X = df[feature_cols].values
    y = df['damage_level'].values
    
    # Manejar valores NaN/infinitos en exp4
    if experiment_name == 'exp4':
        # Reemplazar infinitos por NaN
        X = np.where(np.isfinite(X), X, np.nan)
        
        # Verificar si hay NaN
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            print(f"   ⚠️ Encontrados {np.sum(nan_mask)} valores NaN. Aplicando imputación...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
    
    # Codificar labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_.tolist()
    
    print(f"✓ Features extraídas: {X.shape[1]:,} características")
    print(f"✓ Clases encontradas: {class_names}")
    
    return X, y_encoded, class_names, feature_cols

def compute_separability_metrics(X, y_encoded, class_names, experiment_name):
    """Calcular métricas de separabilidad"""
    print(f"\n📊 Calculando métricas de separabilidad para {experiment_name}...")
    
    metrics = {}
    
    # 1. Silhouette Score global
    try:
        silhouette_avg = silhouette_score(X, y_encoded)
        metrics['silhouette_global'] = silhouette_avg
        print(f"   📈 Silhouette Score: {silhouette_avg:.4f}")
    except Exception as e:
        print(f"   ⚠️ Error calculando Silhouette Score: {e}")
        metrics['silhouette_global'] = 0.0
    
    # 2. Silhouette Score por clase
    try:
        silhouette_samples_scores = silhouette_samples(X, y_encoded)
        silhouette_per_class = {}
        
        for i, class_name in enumerate(class_names):
            class_mask = y_encoded == i
            if np.sum(class_mask) > 0:
                class_silhouette = np.mean(silhouette_samples_scores[class_mask])
                silhouette_per_class[class_name] = class_silhouette
                print(f"   📈 Silhouette {class_name}: {class_silhouette:.4f}")
        
        metrics['silhouette_per_class'] = silhouette_per_class
    except Exception as e:
        print(f"   ⚠️ Error calculando Silhouette por clase: {e}")
        metrics['silhouette_per_class'] = {}
    
    # 3. Davies-Bouldin Index
    try:
        db_score = davies_bouldin_score(X, y_encoded)
        metrics['davies_bouldin'] = db_score
        print(f"   📈 Davies-Bouldin Index: {db_score:.4f}")
    except Exception as e:
        print(f"   ⚠️ Error calculando Davies-Bouldin: {e}")
        metrics['davies_bouldin'] = np.inf
    
    return metrics

def perform_pca_analysis(X, y_encoded, class_names, experiment_name):
    """Realizar análisis PCA para visualización"""
    print(f"\n🔍 Análisis PCA para {experiment_name}...")
    
    # Estandarizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA a 2 y 3 componentes
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    
    print(f"   📊 Varianza explicada PCA-2D: {pca_2d.explained_variance_ratio_.sum():.4f}")
    print(f"   📊 Varianza explicada PCA-3D: {pca_3d.explained_variance_ratio_.sum():.4f}")
    
    return {
        'X_pca_2d': X_pca_2d,
        'X_pca_3d': X_pca_3d,
        'pca_2d': pca_2d,
        'pca_3d': pca_3d,
        'scaler': scaler
    }

def create_comparison_plots(experiments_data, output_dir):
    """Crear plots comparativos entre experimentos"""
    print(f"\n🎨 Creando plots comparativos...")
    
    # Configurar colores consistentes
    class_colors = {'N1': '#2E4057', 'N2': '#048A81', 'N3': '#54C6EB'}
    exp_colors = {'exp2': '#FF6B6B', 'exp3': '#4ECDC4', 'exp4': '#45B7D1'}
    
    # 1. Plot PCA comparativo - Máximo 2 plots por fila para mejor legibilidad
    n_experiments = len(experiments_data)
    if n_experiments <= 2:
        n_rows, n_cols = 1, n_experiments
        fig_height = 5
    else:
        n_rows = (n_experiments + 1) // 2  # Ceiling division
        n_cols = 2
        fig_height = 5 * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, fig_height))
    if n_experiments == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes  # Already a 1D array
    else:
        axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
    for i, (exp_name, data) in enumerate(experiments_data.items()):
        ax = axes[i]
        
        # Plot por clase
        for j, class_name in enumerate(data['class_names']):
            mask = data['y_encoded'] == j
            if np.sum(mask) > 0:
                ax.scatter(data['pca_data']['X_pca_2d'][mask, 0], 
                          data['pca_data']['X_pca_2d'][mask, 1],
                          c=class_colors[class_name], 
                          label=f'{class_name} (n={np.sum(mask)})',
                          alpha=0.7, s=50)
        
        ax.set_xlabel(f'PC1 ({data["pca_data"]["pca_2d"].explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({data["pca_data"]["pca_2d"].explained_variance_ratio_[1]:.2%})')
        
        exp_title = {
            'exp2': 'Exp2: Bins FFT',
            'exp3': 'Exp3: FFT Balanceado', 
            'exp4': 'Exp4: Features Agregadas'
        }
        ax.set_title(exp_title.get(exp_name, exp_name), fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Ocultar subplot extra si hay número impar de experimentos
    if n_experiments % 2 == 1 and n_rows > 1:
        axes[n_experiments].set_visible(False)
    
    plt.tight_layout()
    pca_path = output_dir / "separability_comparison_across_experiments.png"
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ PCA comparison guardado: {pca_path}")
    
    # 2. Plot métricas comparativo - Aumentar espacio vertical para evitar solapamiento
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 14))
    
    # Silhouette Score global
    exp_names = list(experiments_data.keys())
    silhouette_scores = [experiments_data[exp]['metrics']['silhouette_global'] for exp in exp_names]
    
    bars1 = ax1.bar(exp_names, silhouette_scores, color=[exp_colors[exp] for exp in exp_names], alpha=0.8)
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Separabilidad Global por Experimento', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Añadir valores debajo de las barras
    for bar, score in zip(bars1, silhouette_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, -0.05, 
                f'{score:.3f}', ha='center', va='top', fontweight='bold', fontsize=10)
    
    # Davies-Bouldin Index
    db_scores = [experiments_data[exp]['metrics']['davies_bouldin'] for exp in exp_names]
    bars2 = ax2.bar(exp_names, db_scores, color=[exp_colors[exp] for exp in exp_names], alpha=0.8)
    ax2.set_ylabel('Davies-Bouldin Index')
    ax2.set_title('Índice Davies-Bouldin por Experimento\n(Menor = Mejor)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Añadir valores debajo de las barras
    for bar, score in zip(bars2, db_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, -0.2, 
                f'{score:.3f}', ha='center', va='top', fontweight='bold', fontsize=10)
    
    # Silhouette por clase - N1
    n1_scores = []
    for exp in exp_names:
        score = experiments_data[exp]['metrics'].get('silhouette_per_class', {}).get('N1', 0)
        n1_scores.append(score)
    
    bars3 = ax3.bar(exp_names, n1_scores, color=[exp_colors[exp] for exp in exp_names], alpha=0.8)
    ax3.set_ylabel('Silhouette Score N1')
    ax3.set_title('Separabilidad Clase N1', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Añadir valores debajo de las barras
    for bar, score in zip(bars3, n1_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, -0.02, 
                f'{score:.3f}', ha='center', va='top', fontweight='bold', fontsize=10)
    
    # Silhouette por clase - N3 (más problemática)
    n3_scores = []
    for exp in exp_names:
        score = experiments_data[exp]['metrics'].get('silhouette_per_class', {}).get('N3', 0)
        n3_scores.append(score)
    
    bars4 = ax4.bar(exp_names, n3_scores, color=[exp_colors[exp] for exp in exp_names], alpha=0.8)
    ax4.set_ylabel('Silhouette Score N3')
    ax4.set_title('Separabilidad Clase N3 (Crítica)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Añadir valores debajo de las barras
    for bar, score in zip(bars4, n3_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, -0.015, 
                f'{score:.3f}', ha='center', va='top', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    metrics_path = output_dir / "separability_metrics_comparison.png"
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Metrics comparison guardado: {metrics_path}")

def create_pca_3d_comparison(experiments_data, output_dir):
    """Crear plots PCA 3D comparativos"""
    print(f"\n🎨 Creando plots PCA 3D...")
    
    # Configurar colores consistentes
    class_colors = {'N1': '#2E4057', 'N2': '#048A81', 'N3': '#54C6EB'}
    
    # Layout para 3D plots - máximo 2 por fila
    n_experiments = len(experiments_data)
    if n_experiments <= 2:
        n_rows, n_cols = 1, n_experiments
        fig_height = 6
    else:
        n_rows = (n_experiments + 1) // 2  # Ceiling division
        n_cols = 2
        fig_height = 6 * n_rows
    
    fig = plt.figure(figsize=(14, fig_height))
    
    for i, (exp_name, data) in enumerate(experiments_data.items()):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        
        # Plot por clase
        for j, class_name in enumerate(data['class_names']):
            mask = data['y_encoded'] == j
            if np.sum(mask) > 0:
                ax.scatter(data['pca_data']['X_pca_3d'][mask, 0], 
                          data['pca_data']['X_pca_3d'][mask, 1],
                          data['pca_data']['X_pca_3d'][mask, 2],
                          c=class_colors[class_name], 
                          label=f'{class_name} (n={np.sum(mask)})',
                          alpha=0.7, s=50)
        
        # Configurar ejes
        pca_3d = data['pca_data']['pca_3d']
        ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
        
        exp_titles = {
            'exp2': 'Exp2: Bins FFT',
            'exp3': 'Exp3: FFT Balanceado', 
            'exp4': 'Exp4: Features Agregadas'
        }
        ax.set_title(exp_titles.get(exp_name, exp_name), fontsize=12, fontweight='bold')
        ax.legend()
        
        # Mejorar vista 3D
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    pca_3d_path = output_dir / "separability_pca_3d_comparison.png"
    plt.savefig(pca_3d_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ PCA 3D comparison guardado: {pca_3d_path}")

def save_analysis_report(experiments_data, output_path):
    """Guardar reporte del análisis comparativo"""
    print(f"\n💾 Guardando reporte de análisis...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ANÁLISIS COMPARATIVO DE SEPARABILIDAD - EXPERIMENTOS 2, 3, 4\n")
        f.write("=" * 80 + "\n")
        f.write(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Objetivo: Comparar separabilidad en diferentes espacios de características\n\n")
        
        # Resumen por experimento
        f.write("RESUMEN POR EXPERIMENTO:\n")
        f.write("-" * 30 + "\n")
        
        for exp_name, data in experiments_data.items():
            exp_titles = {
                'exp2': 'EXPERIMENTO 2 - Bins FFT (Baseline)',
                'exp3': 'EXPERIMENTO 3 - FFT + Balanceo SMOTE', 
                'exp4': 'EXPERIMENTO 4 - Características Agregadas'
            }
            
            f.write(f"\n{exp_titles.get(exp_name, exp_name)}:\n")
            f.write(f"   Muestras: {len(data['y_encoded'])}\n")
            f.write(f"   Features: {data['X'].shape[1]:,}\n")
            f.write(f"   Silhouette Global: {data['metrics']['silhouette_global']:.4f}\n")
            f.write(f"   Davies-Bouldin: {data['metrics']['davies_bouldin']:.4f}\n")
            
            f.write(f"   Silhouette por clase:\n")
            for class_name in data['class_names']:
                score = data['metrics'].get('silhouette_per_class', {}).get(class_name, 0)
                f.write(f"     {class_name}: {score:.4f}\n")
        
        # Análisis comparativo
        f.write(f"\nANÁLISIS COMPARATIVO:\n")
        f.write("-" * 30 + "\n")
        
        # Mejor separabilidad global
        best_global = max(experiments_data.items(), 
                         key=lambda x: x[1]['metrics']['silhouette_global'])
        f.write(f"• MEJOR SEPARABILIDAD GLOBAL: {best_global[0].upper()}\n")
        f.write(f"  Silhouette Score: {best_global[1]['metrics']['silhouette_global']:.4f}\n\n")
        
        # Separabilidad N3
        f.write(f"• SEPARABILIDAD N3 (CLASE CRÍTICA):\n")
        for exp_name, data in experiments_data.items():
            n3_score = data['metrics'].get('silhouette_per_class', {}).get('N3', 0)
            f.write(f"  {exp_name.upper()}: {n3_score:.4f}\n")
        
        best_n3 = max(experiments_data.items(), 
                     key=lambda x: x[1]['metrics'].get('silhouette_per_class', {}).get('N3', -1))
        f.write(f"  → MEJOR para N3: {best_n3[0].upper()} ({best_n3[1]['metrics'].get('silhouette_per_class', {}).get('N3', 0):.4f})\n\n")
        
        # Conclusiones
        f.write(f"CONCLUSIONES:\n")
        f.write("-" * 30 + "\n")
        
        f.write(f"1. ESPACIO DE CARACTERÍSTICAS:\n")
        if best_global[0] == 'exp4':
            f.write(f"   • Las características agregadas (Exp4) ofrecen MEJOR separabilidad\n")
            f.write(f"   • La agregación estadística reduce la dimensionalidad preservando información discriminativa\n")
        elif best_global[0] == 'exp2':
            f.write(f"   • El espacio FFT original (Exp2) mantiene mejor separabilidad\n") 
            f.write(f"   • El balanceo puede introducir ruido en el espacio de características\n")
        
        f.write(f"\n2. PROBLEMA DE N3:\n")
        n3_scores = {exp: data['metrics'].get('silhouette_per_class', {}).get('N3', 0) 
                    for exp, data in experiments_data.items()}
        avg_n3 = np.mean(list(n3_scores.values()))
        
        if avg_n3 < 0.3:
            f.write(f"   • N3 tiene SEPARABILIDAD INTRÍNSECA BAJA en todos los espacios\n")
            f.write(f"   • Problema fundamental, no de representación de datos\n")
        else:
            f.write(f"   • N3 muestra separabilidad moderada\n")
            f.write(f"   • Mejoras posibles con feature engineering\n")
        
        f.write(f"\n3. RECOMENDACIÓN:\n")
        f.write(f"   • Usar {best_global[0].upper()} para mejor separabilidad general\n")
        f.write(f"   • Considerar ensemble methods para N3\n")
        f.write(f"   • Evaluar features adicionales si N3 es crítico\n")
        
        f.write(f"\n" + "=" * 80 + "\n")
    
    print(f"   ✓ Reporte guardado: {output_path}")

def main():
    """Función principal"""
    print("=" * 80)
    print("ANÁLISIS COMPARATIVO DE SEPARABILIDAD - EXPERIMENTOS 2, 3, 4")
    print("=" * 80)
    print(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Crear directorio de resultados
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        # 1. Cargar datasets de experimentos
        datasets = load_experiments_datasets()
        
        # 2. Procesar cada experimento
        experiments_data = {}
        
        for exp_name, df in datasets.items():
            print(f"\n{'='*50}")
            print(f"PROCESANDO {exp_name.upper()}")
            print(f"{'='*50}")
            
            # Extraer features y labels
            X, y_encoded, class_names, _ = extract_features_and_labels(df, exp_name)
            
            # Calcular métricas
            metrics = compute_separability_metrics(X, y_encoded, class_names, exp_name)
            
            # PCA analysis
            pca_data = perform_pca_analysis(X, y_encoded, class_names, exp_name)
            
            experiments_data[exp_name] = {
                'X': X,
                'y_encoded': y_encoded,
                'class_names': class_names,
                'metrics': metrics,
                'pca_data': pca_data
            }
        
        # 3. Crear visualizaciones comparativas
        create_comparison_plots(experiments_data, output_dir)
        
        # 4. Crear visualizaciones PCA 3D
        create_pca_3d_comparison(experiments_data, output_dir)
        
        # 5. Guardar reporte
        report_path = output_dir / "separability_analysis_report.txt"
        save_analysis_report(experiments_data, report_path)
        
        # 5. Resumen final
        print("\n" + "=" * 80)
        print("🎉 ANÁLISIS COMPARATIVO COMPLETADO")
        print("=" * 80)
        
        print("📊 RANKING DE SEPARABILIDAD GLOBAL:")
        ranking = sorted(experiments_data.items(), 
                        key=lambda x: x[1]['metrics']['silhouette_global'], reverse=True)
        
        for i, (exp_name, data) in enumerate(ranking, 1):
            score = data['metrics']['silhouette_global']
            print(f"   {i}. {exp_name.upper()}: {score:.4f}")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   📊 PCA 2D Comparison: {output_dir}/separability_comparison_across_experiments.png")
        print(f"   🎯 PCA 3D Comparison: {output_dir}/separability_pca_3d_comparison.png")
        print(f"   📈 Metrics Comparison: {output_dir}/separability_metrics_comparison.png")
        print(f"   📋 Reporte detallado: {report_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())