#!/usr/bin/env python3
"""
Script de Comparación entre Experimentos 2, 3 y 4
================================================

Compara los resultados de los enfoques metodológicamente correctos:
- Experimento 2: GroupKFold con matrices FFT completas
- Experimento 3: GroupKFold con balanceamiento de clases
- Experimento 4: GroupKFold con características estadísticas agregadas

NOTA: El Experimento 1 no se incluye ya que es metodológicamente incorrecto
debido a pseudo-replicación por el enfoque de bins FFT independientes.

Genera visualizaciones comparativas y reporte detallado.

Uso:
    python3 src/comparison/compare_experiments.py
"""

import argparse
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import re

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

def load_experiment_data():
    """Cargar datos de los experimentos metodológicamente correctos (2, 3, 4)"""
    project_root = Path(__file__).parent.parent.parent
    
    # Rutas de los archivos JSON para experimentos válidos
    exp2_path = project_root / "src/exp2/results" 
    exp3_path = project_root / "src/exp3/results"
    exp4_path = project_root / "src/exp4/results"
    
    # Archivos JSON de resumen
    exp2_json = exp2_path / "exp2_groupkfold_experiment_summary.json"
    exp3_json = exp3_path / "exp3_balanced_groupkfold_experiment_summary.json"
    exp4_json = exp4_path / "exp4_aggregated_groupkfold_experiment_summary.json"
    
    # Cargar datos
    experiments = {}
    
    # Cargar Experimento 2 (metodológicamente correcto)
    if exp2_json.exists():
        with open(exp2_json, 'r', encoding='utf-8') as f:
            experiments['exp2'] = json.load(f)
    
    # Cargar Experimento 3 (metodológicamente correcto con balanceo)
    if exp3_json.exists():
        with open(exp3_json, 'r', encoding='utf-8') as f:
            experiments['exp3'] = json.load(f)
    
    # Cargar Experimento 4 (metodológicamente correcto con características agregadas)
    if exp4_json.exists():
        with open(exp4_json, 'r', encoding='utf-8') as f:
            experiments['exp4'] = json.load(f)
    
    return experiments

def extract_comparison_metrics(experiments):
    """Extraer métricas clave para comparación"""
    comparison_data = []
    
    for exp_name, exp_data in experiments.items():
        # Manejar diferentes estructuras de JSON entre experimentos
        if exp_name == 'exp4':
            # Exp4 tiene cross_validation directamente en el root
            cv_metrics = exp_data.get('cross_validation', {})
        else:
            # Exp2 y Exp3 tienen cross_validation dentro de metrics
            metrics = exp_data.get('metrics', {})
            cv_metrics = metrics.get('cross_validation', {})
        
        exp_labels = {
            'exp2': 'Exp2: GroupKFold (CNN)',
            'exp3': 'Exp3: GroupKFold + Balanceo (CNN)',
            'exp4': 'Exp4: GroupKFold + Agregadas (DNN)'
        }
        
        split_methods = {
            'exp2': 'GroupKFold',
            'exp3': 'Stratified GroupKFold',
            'exp4': 'GroupKFold'
        }
        
        data_treatments = {
            'exp2': 'Original',
            'exp3': 'Augmentación Conservadora',
            'exp4': 'Características Agregadas'
        }
        
        if exp_name in exp_labels:
            # Función para obtener valor numérico válido
            def get_valid_metric(metric_dict, key, default=0.0):
                value = metric_dict.get(key, default)
                return float(value) if value is not None and str(value).replace('.','').replace('-','').isdigit() else default
            
            row = {
                'experiment': exp_labels[exp_name],
                'split_method': split_methods[exp_name],
                'data_treatment': data_treatments[exp_name],
                'accuracy': get_valid_metric(cv_metrics, 'accuracy'),
                'f1_macro': get_valid_metric(cv_metrics, 'f1_macro'),
                'f1_weighted': get_valid_metric(cv_metrics, 'f1_weighted'),
                'cohen_kappa': get_valid_metric(cv_metrics, 'cohen_kappa'),
                'auc_macro': get_valid_metric(cv_metrics, 'auc_macro'),
                'val_accuracy': get_valid_metric(cv_metrics, 'accuracy'),  # En CV es la misma
                'data_leakage_risk': exp_data.get('data_leakage', {}).get('risk_level', 'LOW')
            }
            
            # Extraer métricas por clase
            per_class = cv_metrics.get('per_class', {})
            for class_name, class_metrics in per_class.items():
                row[f'{class_name}_precision'] = class_metrics.get('precision', 0)
                row[f'{class_name}_recall'] = class_metrics.get('recall', 0)
                row[f'{class_name}_f1'] = class_metrics.get('f1_score', 0)
                row[f'{class_name}_auc'] = class_metrics.get('auc', 0)
            
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_comparison_plots(df, output_dir):
    """Crear gráficos comparativos"""
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    # Definir paleta de colores sobria basada en gráficos de experimentos
    thesis_colors = ['#FF9999', '#D4AF8C', '#90EE90']  # Rosa suave, Dorado, Verde suave
    sns.set_palette(thesis_colors)
    
    # 1. Gráfico de barras de métricas principales
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparación de Enfoques Metodológicamente Correctos', fontsize=16, fontweight='bold')
    
    main_metrics = ['accuracy', 'f1_macro', 'cohen_kappa', 'auc_macro']
    metric_titles = ['Accuracy', 'F1-Score Macro', 'Cohen Kappa', 'AUC Macro']
    
    for idx, (metric, title) in enumerate(zip(main_metrics, metric_titles)):
        ax = axes[idx//2, idx%2]
        
        bars = ax.bar(df['experiment'], df[metric], alpha=0.8, color=thesis_colors)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, df[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'main_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico radar de métricas múltiples
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['Accuracy', 'F1-Macro', 'Cohen Kappa', 'AUC Macro']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Completar el círculo
    
    colors = thesis_colors
    
    for idx, row in df.iterrows():
        values = [row['accuracy'], row['f1_macro'], row['cohen_kappa'], row['auc_macro']]
        values += values[:1]  # Completar el círculo
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['experiment'], color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Perfil Comparativo de Rendimiento\npor Experimento', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.savefig(output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Métricas por clase (solo para las clases consistentes)
    common_classes = []
    for class_prefix in ['N1', 'N2', 'N3', 'N0']:  # Buscar clases comunes
        if f'{class_prefix}_f1' in df.columns:
            common_classes.append(class_prefix)
    
    if common_classes:
        fig, axes = plt.subplots(1, len(common_classes), figsize=(5*len(common_classes), 6))
        if len(common_classes) == 1:
            axes = [axes]
        
        fig.suptitle('Comparación de F1-Score por Clase', fontsize=16, fontweight='bold')
        
        for idx, class_name in enumerate(common_classes):
            f1_col = f'{class_name}_f1'
            if f1_col in df.columns:
                bars = axes[idx].bar(df['experiment'], df[f1_col], alpha=0.8, color=thesis_colors)
                axes[idx].set_title(f'Clase {class_name}', fontweight='bold')
                axes[idx].set_ylabel('F1-Score')
                axes[idx].set_ylim(0, 1.0)
                
                # Añadir valores en las barras
                for bar, value in zip(bars, df[f1_col]):
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'class_f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_comparison_report(df, output_dir):
    """Generar reporte comparativo detallado"""
    
    report_path = output_dir / 'experiment_comparison_report.md'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Reporte Comparativo de Experimentos\n\n")
        f.write("## Deep Learning para Detección de Daño en Aisladores Sísmicos\n\n")
        f.write(f"**Fecha de Generación:** {timestamp}\n\n")
        f.write("---\n\n")
        
        f.write("## Resumen Ejecutivo\n\n")
        f.write("Este reporte compara el rendimiento de tres enfoques metodológicos diferentes:\n\n")
        f.write("1. **Experimento 1**: Enfoque directo con split aleatorio tradicional\n")
        f.write("2. **Experimento 2**: Prevención de data leakage con GroupKFold\n")
        f.write("3. **Experimento 3**: Balanceamiento de clases con augmentación conservadora\n\n")
        
        f.write("## Tabla Comparativa de Métricas Principales\n\n")
        
        # Crear tabla de métricas principales
        main_cols = ['experiment', 'accuracy', 'f1_macro', 'cohen_kappa', 'auc_macro', 'data_leakage_risk']
        main_df = df[main_cols].copy()
        
        f.write("| Experimento | Accuracy | F1-Macro | Cohen Kappa | AUC Macro | Riesgo Data Leakage |\n")
        f.write("|-------------|----------|----------|-------------|-----------|--------------------|\n")
        
        for _, row in main_df.iterrows():
            f.write(f"| {row['experiment']} | {row['accuracy']:.3f} | {row['f1_macro']:.3f} | ")
            f.write(f"{row['cohen_kappa']:.3f} | {row['auc_macro']:.3f} | {row['data_leakage_risk']} |\n")
        
        f.write("\n")
        
        # Análisis de resultados
        f.write("## Análisis de Resultados\n\n")
        
        # Encontrar el mejor por métrica
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        best_f1 = df.loc[df['f1_macro'].idxmax()]
        best_kappa = df.loc[df['cohen_kappa'].idxmax()]
        
        f.write("### Mejores Rendimientos por Métrica\n\n")
        f.write(f"- **Mejor Accuracy**: {best_accuracy['experiment']} ({best_accuracy['accuracy']:.3f})\n")
        f.write(f"- **Mejor F1-Macro**: {best_f1['experiment']} ({best_f1['f1_macro']:.3f})\n")
        f.write(f"- **Mejor Cohen Kappa**: {best_kappa['experiment']} ({best_kappa['cohen_kappa']:.3f})\n\n")
        
        # Análisis por clase (si están disponibles)
        f.write("### Rendimiento por Clase de Daño\n\n")
        for class_name in ['N1', 'N2', 'N3', 'N0']:
            f1_col = f'{class_name}_f1'
            if f1_col in df.columns:
                f.write(f"**Clase {class_name}:**\n")
                for _, row in df.iterrows():
                    f.write(f"- {row['experiment']}: {row[f1_col]:.3f}\n")
                f.write("\n")
        
        f.write("## Observaciones Metodológicas\n\n")
        
        f.write("### Validación y Generalización\n\n")
        f.write("- **Exp1** muestra métricas altas pero con riesgo de sobreestimación debido a data leakage\n")
        f.write("- **Exp2** proporciona una evaluación más realista con GroupKFold anti-leakage\n")
        f.write("- **Exp3** demuestra el impacto del balanceamiento de datos en clases minoritarias\n\n")
        
        f.write("### Consideraciones para N3 (Clase Minoritaria)\n\n")
        n3_f1_col = 'N3_f1'
        if n3_f1_col in df.columns:
            f.write("El rendimiento en la clase N3 es consistentemente bajo, indicando:\n")
            f.write("- Separabilidad intrínseca limitada\n")
            f.write("- Necesidad de técnicas avanzadas de feature engineering\n")
            f.write("- Posible redefinición de criterios de clasificación\n\n")
        
        f.write("## Conclusiones\n\n")
        
        # Conclusiones basadas en los datos
        if best_f1['experiment'] == 'Exp2: Anti-Data Leakage':
            f.write("1. **Exp2 (GroupKFold)** ofrece el mejor balance entre rendimiento y validez metodológica\n")
        elif best_f1['experiment'] == 'Exp3: Balanceamiento':
            f.write("1. **Exp3 (Balanceamiento)** demuestra la efectividad de la augmentación conservadora\n")
        else:
            f.write("1. **Exp1** muestra el mejor rendimiento, pero con limitaciones metodológicas\n")
        
        f.write("2. La prevención de data leakage es fundamental para evaluaciones realistas\n")
        f.write("3. El balanceamiento de clases mejora la detectabilidad de daños severos (N3)\n")
        f.write("4. Se requiere investigación adicional para mejorar la separabilidad de la clase N3\n\n")
        
        f.write("## Recomendaciones\n\n")
        f.write("- **Para aplicación práctica**: Usar metodología de Exp2 o Exp3\n")
        f.write("- **Para investigación futura**: Explorar feature engineering avanzado para N3\n")
        f.write("- **Para validación**: Mantener GroupKFold en todos los experimentos futuros\n")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Comparar resultados de experimentos 1, 2 y 3",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--output",
        default=None,
        help="Directorio de salida (default: src/comparison/results)"
    )
    
    args = parser.parse_args()
    
    try:
        print("="*70)
        print("COMPARACIÓN DE EXPERIMENTOS - DCNN AISLADORES SÍSMICOS")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Configurar directorio de salida
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path(__file__).parent / "results"
        
        output_dir.mkdir(exist_ok=True)
        print(f"📁 Directorio de salida: {output_dir}")
        
        # Cargar datos de experimentos
        print("📊 Cargando datos de experimentos...")
        experiments = load_experiment_data()
        
        if not experiments:
            raise ValueError("No se encontraron datos de experimentos")
        
        print(f"✓ Experimentos cargados: {list(experiments.keys())}")
        
        # Extraer métricas para comparación
        print("🔍 Extrayendo métricas comparativas...")
        comparison_df = extract_comparison_metrics(experiments)
        
        # Guardar tabla comparativa
        csv_path = output_dir / "experiment_comparison_table.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"✓ Tabla guardada: {csv_path}")
        
        # Crear visualizaciones
        print("📈 Generando visualizaciones...")
        create_comparison_plots(comparison_df, output_dir)
        print("✓ Gráficos generados:")
        print("   🎯 main_metrics_comparison.png")
        print("   🎪 radar_comparison.png")
        print("   📊 class_f1_comparison.png")
        
        # Generar reporte
        print("📝 Generando reporte...")
        generate_comparison_report(comparison_df, output_dir)
        print("✓ Reporte generado: experiment_comparison_report.md")
        
        print("\n" + "="*70)
        print("🎉 COMPARACIÓN COMPLETADA EXITOSAMENTE")
        print("="*70)
        print(f"📁 Todos los archivos guardados en: {output_dir}")
        print()
        
        # Mostrar resumen rápido
        print("📊 RESUMEN RÁPIDO:")
        print("-" * 30)
        for _, row in comparison_df.iterrows():
            print(f"{row['experiment']}: Accuracy={row['accuracy']:.3f}, F1={row['f1_macro']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())