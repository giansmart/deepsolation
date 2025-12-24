#!/usr/bin/env python3
"""
Script para ejecutar múltiples iteraciones del Experimento 4
===========================================================

Ejecuta el entrenamiento del exp4 múltiples veces para evaluar la estabilidad
y variabilidad del modelo, generando estadísticas agregadas científicamente válidas.

Uso:
    python3 src/exp4/3_run_multiple_iterations.py --iterations 3

Salidas:
    - results/aggregate_results.json: Resultados agregados con estadísticas
    - results/iteration_*: Resultados individuales de cada iteración
"""

import argparse
import sys
import json
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime
import time

def run_single_iteration(iteration_num, input_dataset):
    """Ejecuta una iteración del entrenamiento"""
    
    print(f"\n{'='*60}")
    print(f"🔄 ITERACIÓN {iteration_num}")
    print(f"{'='*60}")
    
    # Crear directorio para esta iteración
    project_root = Path(__file__).parent.parent.parent
    iter_dir = project_root / f"src/exp4/results/iteration_{iteration_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    # Comando para ejecutar el entrenamiento
    cmd = [
        "python3", 
        str(project_root / "src/exp4/2_train_dcnn.py"),
        "--input", str(input_dataset)
    ]
    
    print(f"📋 Ejecutando: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Ejecutar entrenamiento
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Iteración {iteration_num} completada en {execution_time:.1f}s")
            
            # Copiar resultados a directorio de iteración
            results_dir = project_root / "src/exp4/results"
            
            # Buscar archivos de resultados generados
            json_files = list(results_dir.glob("*experiment_summary.json"))
            if json_files:
                latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
                
                # Cargar resultados
                with open(latest_json, 'r') as f:
                    iteration_results = json.load(f)
                
                # Guardar en directorio de iteración
                iter_json = iter_dir / f"iteration_{iteration_num}_results.json"
                with open(iter_json, 'w') as f:
                    json.dump(iteration_results, f, indent=2, default=str)
                
                # Copiar gráficos importantes
                plots_to_copy = [
                    "*classification_metrics.png",
                    "*confusion_matrix.png", 
                    "*roc_curves.png",
                    "*training_curves.png"
                ]
                
                for pattern in plots_to_copy:
                    matching_files = list(results_dir.glob(pattern))
                    for file_path in matching_files:
                        if file_path.exists():
                            dest_path = iter_dir / f"iteration_{iteration_num}_{file_path.name}"
                            subprocess.run(['cp', str(file_path), str(dest_path)])
                
                print(f"📁 Resultados guardados en: {iter_dir}")
                
                return iteration_results
                
        else:
            print(f"❌ Error en iteración {iteration_num}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Excepción en iteración {iteration_num}: {e}")
        return None

def aggregate_results(all_results):
    """Agregar resultados de múltiples iteraciones"""
    
    print(f"\n📊 Agregando resultados de {len(all_results)} iteraciones...")
    
    # Extraer métricas principales
    accuracies = []
    f1_macros = []
    f1_weighteds = []
    kappas = []
    auc_macros = []
    
    # Métricas por clase
    per_class_metrics = {}
    
    for result in all_results:
        cv = result['cross_validation']
        accuracies.append(cv['accuracy'])
        f1_macros.append(cv['f1_macro'])
        f1_weighteds.append(cv['f1_weighted'])
        kappas.append(cv['cohen_kappa'])
        auc_macros.append(cv.get('auc_macro', 0))
        
        # Agregar métricas por clase
        for class_name, metrics in cv['per_class'].items():
            if class_name not in per_class_metrics:
                per_class_metrics[class_name] = {
                    'precision': [],
                    'recall': [],
                    'f1_score': []
                }
            per_class_metrics[class_name]['precision'].append(metrics['precision'])
            per_class_metrics[class_name]['recall'].append(metrics['recall'])
            per_class_metrics[class_name]['f1_score'].append(metrics['f1_score'])
    
    # Calcular estadísticas agregadas
    aggregate_results = {
        'experiment_name': 'exp4_aggregated_multiple_iterations',
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Statistical Aggregated Features + GroupKFold (Multiple Runs)',
        'approach': 'One observation = One physical device',
        'model_architecture': 'Fully Connected DNN',
        'aggregation_info': {
            'n_iterations': len(all_results),
            'aggregation_method': 'mean_std_across_iterations',
            'note': 'No fixed random seeds - reflects true model variability'
        },
        'aggregate_metrics': {
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'values': accuracies
            },
            'f1_macro': {
                'mean': float(np.mean(f1_macros)),
                'std': float(np.std(f1_macros)),
                'min': float(np.min(f1_macros)),
                'max': float(np.max(f1_macros)),
                'values': f1_macros
            },
            'f1_weighted': {
                'mean': float(np.mean(f1_weighteds)),
                'std': float(np.std(f1_weighteds)),
                'min': float(np.min(f1_weighteds)),
                'max': float(np.max(f1_weighteds)),
                'values': f1_weighteds
            },
            'cohen_kappa': {
                'mean': float(np.mean(kappas)),
                'std': float(np.std(kappas)),
                'min': float(np.min(kappas)),
                'max': float(np.max(kappas)),
                'values': kappas
            },
            'auc_macro': {
                'mean': float(np.mean(auc_macros)),
                'std': float(np.std(auc_macros)),
                'min': float(np.min(auc_macros)),
                'max': float(np.max(auc_macros)),
                'values': auc_macros
            }
        },
        'per_class_aggregate': {}
    }
    
    # Agregar métricas por clase
    for class_name, metrics in per_class_metrics.items():
        aggregate_results['per_class_aggregate'][class_name] = {}
        for metric_name, values in metrics.items():
            aggregate_results['per_class_aggregate'][class_name][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
    
    # Análisis de estabilidad
    cv_accuracy = np.std(accuracies) / np.mean(accuracies)
    cv_f1 = np.std(f1_macros) / np.mean(f1_macros)
    
    stability_level = "HIGH" if cv_accuracy < 0.05 else "MEDIUM" if cv_accuracy < 0.15 else "LOW"
    
    aggregate_results['stability_analysis'] = {
        'coefficient_of_variation': {
            'accuracy': float(cv_accuracy),
            'f1_macro': float(cv_f1)
        },
        'stability_level': stability_level,
        'interpretation': {
            'accuracy_range': f"{np.min(accuracies):.3f} - {np.max(accuracies):.3f}",
            'f1_range': f"{np.min(f1_macros):.3f} - {np.max(f1_macros):.3f}",
            'recommendation': "Include error bars in comparisons" if cv_accuracy > 0.05 else "Results are stable"
        }
    }
    
    # Estadísticas descriptivas para reporte
    aggregate_results['reporting_format'] = {
        'accuracy_report': f"{np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}",
        'f1_macro_report': f"{np.mean(f1_macros):.3f} ± {np.std(f1_macros):.3f}",
        'cohen_kappa_report': f"{np.mean(kappas):.3f} ± {np.std(kappas):.3f}",
        'auc_macro_report': f"{np.mean(auc_macros):.3f} ± {np.std(auc_macros):.3f}"
    }
    
    return aggregate_results

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Ejecutar múltiples iteraciones del Experimento 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

    # Ejecutar 3 iteraciones (recomendado)
    python src/exp4/3_run_multiple_iterations.py --iterations 3
    
    # Ejecutar 5 iteraciones para análisis más robusto
    python src/exp4/3_run_multiple_iterations.py --iterations 5

NOTA: No usa semillas fijas para reflejar la variabilidad real del modelo
        """
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Número de iteraciones a ejecutar (default: 3)"
    )
    
    parser.add_argument(
        "--input",
        default="src/exp4/results/preprocessed_dataset.csv",
        help="Dataset de entrada"
    )
    
    args = parser.parse_args()
    
    try:
        print("="*80)
        print("EXPERIMENTO 4 - MÚLTIPLES ITERACIONES")
        print("Evaluación de Estabilidad y Variabilidad del Modelo")
        print("="*80)
        print(f"🔄 Iteraciones a ejecutar: {args.iterations}")
        print(f"📂 Dataset: {args.input}")
        print(f"🎯 SIN semillas fijas - Variabilidad real")
        print()
        
        # Verificar dataset
        project_root = Path(__file__).parent.parent.parent
        dataset_path = project_root / args.input
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
        
        # Ejecutar iteraciones
        all_results = []
        start_time = time.time()
        
        for i in range(1, args.iterations + 1):
            result = run_single_iteration(i, dataset_path)
            if result:
                all_results.append(result)
            else:
                print(f"⚠️ Iteración {i} falló, continuando...")
        
        total_time = time.time() - start_time
        
        if len(all_results) == 0:
            raise Exception("Todas las iteraciones fallaron")
        
        print(f"\n✅ {len(all_results)}/{args.iterations} iteraciones exitosas")
        
        # Agregar resultados
        aggregated = aggregate_results(all_results)
        
        # Guardar resultados agregados
        results_dir = project_root / "src/exp4/results"
        aggregate_path = results_dir / "aggregate_results.json"
        
        with open(aggregate_path, 'w') as f:
            json.dump(aggregated, f, indent=2, default=str)
        
        # Crear reporte legible
        report_path = results_dir / "multiple_iterations_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENTO 4 - REPORTE DE MÚLTIPLES ITERACIONES\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Iteraciones ejecutadas: {len(all_results)}/{args.iterations}\n")
            f.write(f"Tiempo total: {total_time:.1f} segundos\n\n")
            
            f.write("RESULTADOS AGREGADOS:\n")
            f.write("-"*30 + "\n")
            
            metrics = aggregated['aggregate_metrics']
            f.write(f"Accuracy: {metrics['accuracy']['mean']:.3f} ± {metrics['accuracy']['std']:.3f}\n")
            f.write(f"F1-Macro: {metrics['f1_macro']['mean']:.3f} ± {metrics['f1_macro']['std']:.3f}\n")
            f.write(f"F1-Weighted: {metrics['f1_weighted']['mean']:.3f} ± {metrics['f1_weighted']['std']:.3f}\n")
            f.write(f"Cohen Kappa: {metrics['cohen_kappa']['mean']:.3f} ± {metrics['cohen_kappa']['std']:.3f}\n")
            f.write(f"AUC-Macro: {metrics['auc_macro']['mean']:.3f} ± {metrics['auc_macro']['std']:.3f}\n\n")
            
            f.write("ANÁLISIS DE ESTABILIDAD:\n")
            f.write("-"*25 + "\n")
            stability = aggregated['stability_analysis']
            f.write(f"Nivel de estabilidad: {stability['stability_level']}\n")
            f.write(f"CV Accuracy: {stability['coefficient_of_variation']['accuracy']:.3f}\n")
            f.write(f"CV F1-Macro: {stability['coefficient_of_variation']['f1_macro']:.3f}\n")
            f.write(f"Rango Accuracy: {stability['interpretation']['accuracy_range']}\n")
            f.write(f"Recomendación: {stability['interpretation']['recommendation']}\n\n")
            
            f.write("FORMATO PARA REPORTE:\n")
            f.write("-"*23 + "\n")
            report_format = aggregated['reporting_format']
            for metric, format_str in report_format.items():
                f.write(f"{metric}: {format_str}\n")
        
        print(f"\n{'='*80}")
        print("🎉 MÚLTIPLES ITERACIONES COMPLETADAS")
        print("="*80)
        print(f"📊 Resultados agregados: {aggregate_path}")
        print(f"📋 Reporte legible: {report_path}")
        print(f"📁 Iteraciones individuales: src/exp4/results/iteration_*")
        print()
        print("📈 RESULTADOS AGREGADOS:")
        agg = aggregated['aggregate_metrics']
        print(f"   Accuracy: {agg['accuracy']['mean']:.3f} ± {agg['accuracy']['std']:.3f}")
        print(f"   F1-Macro: {agg['f1_macro']['mean']:.3f} ± {agg['f1_macro']['std']:.3f}")
        print(f"   Estabilidad: {aggregated['stability_analysis']['stability_level']}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())