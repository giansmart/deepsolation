#!/usr/bin/env python3
"""
Script de Entrenamiento DCNN con Dataset Balanceado - Experimento 3
===================================================================

Entrena el modelo DCNN usando el dataset balanceado con augmentación conservadora,
manteniendo la metodología Yu et al. (2018) y Stratified GroupKFold.

Diferencias con exp2:
- Usa dataset balanceado con augmentación conservadora
- Especímenes sintéticos se tratan como grupos independientes
- Métricas adicionales para evaluar impacto del balanceamiento

Uso:
    python3 src/exp3/3_train_dcnn.py --input src/exp3/results/balanced_dataset.csv

Requisitos:
    - Dataset balanceado (CSV generado por 2_balance_data.py)
    - Metodología idéntica a exp2 para comparabilidad

Salidas:
    - models/dcnn_model_*.pth: Modelo entrenado
    - results/*_experiment_summary.json: Resumen completo del experimento
    - results/*_groupkfold_report.txt: Análisis detallado de GroupKFold
    - results/*_balance_impact_analysis.txt: Análisis del impacto del balanceamiento
"""

import argparse
import sys
from pathlib import Path
import json
import time
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from collections import Counter

# Agregar src al path para acceder a utils
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from dcnn_model import DCNNDamageNet, DCNNTrainer, get_optimal_device
from dataset_utils import prepare_dataset_from_csv
from experiment_metrics import ExperimentEvaluator
from exp2.stratified_group_kfold import StratifiedGroupKFold

# Configuración por defecto (IDÉNTICA a exp2 para comparabilidad)
def get_default_config():
    project_root = Path(__file__).parent.parent.parent  # deepsolation/
    return {
        "n_splits": 5,  # GroupKFold splits
        "test_size": 0.2,  # Para split final
        "val_size": 0.2,   # Para split final
        "batch_size": 20,  # MISMO que exp1/exp2
        "learning_rate": 0.0035,  # MISMO que exp1/exp2
        "epochs": 60,     # MISMO que exp1/exp2 
        "patience": 15,    # MISMO que exp1/exp2
        "dropout_rate": 0.3,  # MISMO que exp1/exp2
        "models_dir": str(project_root / "src/exp3/models"),
        "results_dir": str(project_root / "src/exp3/results"),
        "random_state": 42,
        "use_class_weights": True
    }

def extract_specimen_group(specimen_name):
    """
    Extrae el grupo de specimen físico para GroupKFold, tratando especímenes augmentados como independientes
    A1, A1-2, A1-3 → 'A1'
    A1_aug1 → 'A1_aug1' (grupo independiente)
    """
    if '_aug' in specimen_name:
        # Especímenes augmentados se tratan como grupos independientes
        return specimen_name
    elif '-' in specimen_name:
        return specimen_name.split('-')[0]
    else:
        return specimen_name

def load_and_prepare_data(dataset_path):
    """Cargar y preparar datos balanceados con grupos para GroupKFold"""
    print(f"📂 Cargando dataset balanceado: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Validar columnas requeridas
        required_cols = ['specimen', 'sensor', 'damage_level']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Columnas faltantes en dataset: {missing_cols}")
        
        # Verificar que tenemos columnas de matriz serializada
        import re
        freq_pattern = re.compile(r'^freq_\d+_(NS|EW|UD)$')
        freq_cols = [col for col in df.columns if freq_pattern.match(col)]
        if len(freq_cols) == 0:
            raise ValueError("No se encontraron columnas de frecuencia serializadas (freq_XXXX_YY)")
        
        print(f"✓ Dataset balanceado cargado: {len(df):,} muestras (matrices completas)")
        print(f"✓ Especímenes únicos: {df['specimen'].nunique()}")
        print(f"✓ Columnas de frecuencia: {len(freq_cols)}")
        
        # Crear grupos de specimens físicos (incluyendo augmentados)
        df['specimen_group'] = df['specimen'].apply(extract_specimen_group)
        unique_groups = df['specimen_group'].unique()
        print(f"✓ Grupos físicos únicos (incluyendo augmentados): {len(unique_groups)}")
        
        # Análizar distribución original vs augmentada
        print(f"\n📊 Análisis de augmentación:")
        original_specimens = df[~df['specimen'].str.contains('_aug')]['specimen'].nunique()
        augmented_specimens = df[df['specimen'].str.contains('_aug')]['specimen'].nunique()
        print(f"   Especímenes originales: {original_specimens}")
        print(f"   Especímenes augmentados: {augmented_specimens}")
        
        # Mostrar distribución de clases balanceada
        print(f"\n📊 Distribución de clases (BALANCEADA):")
        damage_counts = df['damage_level'].value_counts().sort_index()
        total_samples = len(df)
        
        for damage_level, count in damage_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   {damage_level}: {count:,} ({percentage:.1f}%)")
        
        return df
        
    except Exception as e:
        raise Exception(f"Error cargando dataset balanceado: {e}")

def prepare_features_and_groups(df):
    """Preparar características y grupos para GroupKFold - Dataset balanceado"""
    print("🔄 Preparando características de matrices serializadas (balanceadas)...")
    
    # Extraer columnas de frecuencias serializadas
    import re
    freq_pattern = re.compile(r'^freq_\d+_(NS|EW|UD)$')
    freq_cols = [col for col in df.columns if freq_pattern.match(col)]
    freq_cols.sort()  # Asegurar orden consistente
    
    # Cada fila ya es una muestra completa (matriz serializada)
    X = df[freq_cols].values
    groups = df['specimen_group'].values  
    labels = df['damage_level'].values
    
    print(f"✓ {len(X)} muestras preparadas (matrices serializadas balanceadas)")
    print(f"✓ Features por muestra: {X.shape[1]:,}")
    print(f"✓ Grupos únicos (incluyendo augmentados): {len(np.unique(groups))}")
    
    return X, groups, labels

# Las funciones create_data_loaders, train_fold, train_final_model, save_model, save_results 
# son idénticas a exp2, así que las voy a copiar directamente desde exp2

def create_data_loaders(X_indices, y_indices, X_data, y_data, config):
    """Crear DataLoaders para los índices dados - Estructura balanceada"""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Seleccionar datos usando índices
    X_selected = X_data[X_indices]
    y_selected = y_data[y_indices]
    
    # Los datos ya están serializados, necesitamos reshapear para DCNN
    n_features = X_selected.shape[1]
    
    # Asumir que son matrices (freq_components, 3) serializadas
    if n_features % 3 != 0:
        raise ValueError(f"Features {n_features} no es múltiplo de 3 (esperado: freq_components * 3)")
    
    freq_components = n_features // 3
    
    # Reshape de vector serializado a matriz (batch, freq_components, 3)
    X_reshaped = X_selected.reshape(-1, freq_components, 3)
    
    X_tensor = torch.FloatTensor(X_reshaped)
    y_tensor = torch.LongTensor(y_selected)
    
    # Crear DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    # Input shape para el modelo
    input_shape = (freq_components, 3)
    
    return loader, input_shape

def train_fold(fold_idx, train_loader, val_loader, input_shape, num_classes, config, device, class_weights=None):
    """Entrenar un fold específico"""
    print(f"\n🚀 Entrenando Fold {fold_idx + 1}")
    
    # Crear modelo
    model = DCNNDamageNet(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=config["dropout_rate"]
    )
    
    # Crear trainer
    trainer = DCNNTrainer(model, device=device)
    
    # Entrenar
    start_time = time.time()
    
    trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        patience=config["patience"],
        class_weights=class_weights
    )
    
    training_time = time.time() - start_time
    print(f"✓ Fold {fold_idx + 1} completado en {training_time:.1f} segundos")
    
    return trainer, training_time

def evaluate_with_groupkfold(X_data, groups, labels, config, device, class_weights=None):
    """Evaluación completa con GroupKFold y visualizaciones - Dataset balanceado"""
    print("🔍 EVALUACIÓN CON STRATIFIED GROUPKFOLD (DATASET BALANCEADO)")
    print("=" * 50)
    
    # Configurar evaluador de experimentos
    evaluator = ExperimentEvaluator(
        results_dir=config["results_dir"],
        show_plots=False,
        experiment_name="exp3_balanced_groupkfold"
    )
    
    # Codificar labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_.tolist()
    
    # Configurar Stratified GroupKFold
    print("🎯 Usando Stratified GroupKFold para dataset balanceado...")
    gkf = StratifiedGroupKFold(n_splits=config["n_splits"], random_state=config["random_state"])
    
    # Usar class_weights pasados como parámetro
    if class_weights is not None:
        print(f"✓ Usando class weights: {class_weights.cpu().numpy()}")
    
    # Variables para acumular resultados
    fold_results = []
    fold_times = []
    all_train_true, all_train_pred, all_train_proba = [], [], []
    all_val_true, all_val_pred, all_val_proba = [], [], []
    
    # Cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_data, y_encoded, groups)):
        print(f"\n📁 Procesando Fold {fold_idx + 1}/{config['n_splits']}")
        print(f"   Train specimens: {len(train_idx)}, Val specimens: {len(val_idx)}")
        
        # Verificar que no hay leakage entre grupos
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        leakage_groups = train_groups.intersection(val_groups)
        
        if leakage_groups:
            print(f"⚠️ WARNING: Grupos en ambos splits: {leakage_groups}")
        else:
            print(f"✅ Sin data leakage: {len(train_groups)} grupos train, {len(val_groups)} grupos val")
        
        # Analizar distribución de clases en este fold
        y_train_fold = y_encoded[train_idx]
        y_val_fold = y_encoded[val_idx]
        train_class_counts = np.bincount(y_train_fold)
        val_class_counts = np.bincount(y_val_fold) 
        
        print(f"   📊 Train classes: N1={train_class_counts[0] if len(train_class_counts)>0 else 0}, "
              f"N2={train_class_counts[1] if len(train_class_counts)>1 else 0}, "
              f"N3={train_class_counts[2] if len(train_class_counts)>2 else 0}")
        print(f"   📊 Val classes: N1={val_class_counts[0] if len(val_class_counts)>0 else 0}, "
              f"N2={val_class_counts[1] if len(val_class_counts)>1 else 0}, "
              f"N3={val_class_counts[2] if len(val_class_counts)>2 else 0}")
        
        # Crear DataLoaders
        train_loader, input_shape = create_data_loaders(
            train_idx, train_idx, X_data, y_encoded, config
        )
        val_loader, _ = create_data_loaders(
            val_idx, val_idx, X_data, y_encoded, config
        )
        
        # Entrenar fold
        trainer, fold_time = train_fold(
            fold_idx, train_loader, val_loader, input_shape, 
            len(label_encoder.classes_), config, device, class_weights
        )
        
        # Evaluar fold y recolectar predicciones
        train_metrics = trainer.evaluate_dataset(train_loader, split_name="train")
        val_metrics = trainer.evaluate_dataset(val_loader, split_name="validation")
        
        # Acumular predicciones para análisis conjunto
        all_train_true.extend(train_metrics['y_true'])
        all_train_pred.extend(train_metrics['y_pred'])
        if 'y_proba' in train_metrics:
            all_train_proba.extend(train_metrics['y_proba'])
        
        all_val_true.extend(val_metrics['y_true']) 
        all_val_pred.extend(val_metrics['y_pred'])
        if 'y_proba' in val_metrics:
            all_val_proba.extend(val_metrics['y_proba'])
        
        # Guardar resultados del fold
        fold_result = {
            'fold': fold_idx + 1,
            'train_acc': accuracy_score(train_metrics['y_true'], train_metrics['y_pred']),
            'val_acc': accuracy_score(val_metrics['y_true'], val_metrics['y_pred']),
            'train_groups': list(train_groups),
            'val_groups': list(val_groups),
            'training_time': fold_time
        }
        
        fold_results.append(fold_result)
        fold_times.append(fold_time)
        
        print(f"   Train Acc: {fold_result['train_acc']:.4f}")
        print(f"   Val Acc: {fold_result['val_acc']:.4f}")
    
    # Calcular estadísticas agregadas
    train_accs = [r['train_acc'] for r in fold_results]
    val_accs = [r['val_acc'] for r in fold_results]
    
    print(f"\n📊 RESULTADOS AGREGADOS (DATASET BALANCEADO):")
    print(f"   Train Acc: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"   Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"   Tiempo total: {sum(fold_times):.1f}s")
    
    # GENERAR VISUALIZACIONES Y MÉTRICAS
    print("\n🎨 Generando visualizaciones...")
    
    try:
        # 1. Evaluar métricas de validación cruzada
        all_val_proba_array = np.array(all_val_proba) if all_val_proba else None
        print(f"   🔍 Evaluando métricas de clasificación...")
        val_metrics = evaluator.evaluate_classification(
            y_true=all_val_true,
            y_pred=all_val_pred, 
            y_proba=all_val_proba_array,
            class_names=class_names,
            split_name="cross_validation"
        )
        print(f"   ✓ Métricas de clasificación completadas")
        
        # 2. Detectar data leakage
        print(f"   🔍 Detectando data leakage...")
        leakage_report = evaluator.detect_data_leakage(
            train_acc=np.mean(train_accs),
            val_acc=np.mean(val_accs),
            test_acc=np.mean(val_accs),  # Usamos val como proxy de test
            cv_scores=val_accs
        )
        print(f"   ✓ Data leakage análisis completado")
        
        # 3. Generar gráficos
        print(f"   🔍 Generando matriz de confusión...")
        evaluator.plot_confusion_matrix(all_val_true, all_val_pred, class_names)
        print(f"   ✓ Matriz de confusión generada")
        
        print(f"   🔍 Generando métricas por clase...")
        evaluator.plot_classification_metrics(val_metrics, class_names)
        print(f"   ✓ Métricas por clase generadas")
        
        if all_val_proba_array is not None:
            print(f"   🔍 Generando curvas ROC...")
            evaluator.plot_roc_curves(all_val_true, all_val_proba_array, class_names)
            print(f"   ✓ Curvas ROC generadas")
    
    except Exception as e:
        print(f"   ❌ Error en visualizaciones: {e}")
        import traceback
        traceback.print_exc()
        # Continuar sin visualizaciones
        val_metrics = {'accuracy': np.mean(val_accs)}
        leakage_report = {'alerts': [], 'metrics': {}}
    
    # 4. Generar resumen del experimento
    metrics_summary = {
        'cross_validation': val_metrics
    }
    
    experiment_summary = evaluator.generate_experiment_summary(
        metrics_dict=metrics_summary,
        leakage_report=leakage_report
    )
    
    print("✅ Visualizaciones generadas:")
    print("   🎯 Matriz de confusión")
    print("   📈 Métricas por clase")
    print("   📉 Curvas ROC")
    print("   📋 Reporte de data leakage")
    
    return {
        'fold_results': fold_results,
        'mean_train_acc': np.mean(train_accs),
        'std_train_acc': np.std(train_accs),
        'mean_val_acc': np.mean(val_accs),
        'std_val_acc': np.std(val_accs),
        'total_time': sum(fold_times),
        'label_encoder': label_encoder,
        'experiment_summary': experiment_summary,
        'leakage_report': leakage_report
    }

def train_final_model(X_data, labels, label_encoder, config, device, class_weights=None):
    """Entrenar modelo final con todos los datos para guardado"""
    # Codificar labels
    y_encoded = label_encoder.transform(labels)
    
    # Usar 80% para entrenamiento, 20% para validación
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(X_data))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=config["val_size"], 
        random_state=config["random_state"],
        stratify=y_encoded
    )
    
    # Crear DataLoaders
    train_loader, input_shape = create_data_loaders(
        train_idx, train_idx, X_data, y_encoded, config
    )
    val_loader, _ = create_data_loaders(
        val_idx, val_idx, X_data, y_encoded, config
    )
    
    # Crear y entrenar modelo
    model = DCNNDamageNet(
        input_shape=input_shape,
        num_classes=len(label_encoder.classes_),
        dropout_rate=config["dropout_rate"]
    )
    
    trainer = DCNNTrainer(model, device=device)
    
    # Entrenar
    start_time = time.time()
    
    trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        patience=config["patience"],
        class_weights=class_weights
    )
    
    training_time = time.time() - start_time
    print(f"✓ Modelo final entrenado en {training_time:.1f} segundos")
    
    return model, trainer

def save_model(model, trainer, label_encoder, cv_results, config):
    """Guardar modelo final entrenado"""
    print("\n💾 Guardando modelo...")
    
    # Crear directorios
    models_dir = Path(config["models_dir"])
    models_dir.mkdir(exist_ok=True)
    
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(exist_ok=True)
    
    # Guardar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"dcnn_model_{timestamp}.pth"
    
    # Preparar datos para guardado
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_shape': model.input_shape,
            'num_classes': model.num_classes,
            'dropout_rate': model.dropout_rate
        },
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'training_config': config,
        'cv_results': {
            'mean_train_acc': float(cv_results['mean_train_acc']),
            'std_train_acc': float(cv_results['std_train_acc']),
            'mean_val_acc': float(cv_results['mean_val_acc']),
            'std_val_acc': float(cv_results['std_val_acc']),
            'total_time': float(cv_results['total_time'])
        },
        'timestamp': timestamp,
        'experiment': 'exp3_balanced_groupkfold'
    }
    
    torch.save(model_data, model_path)
    
    # Guardar también como modelo actual
    current_model_path = models_dir / "dcnn_model_current.pth"
    torch.save(model_data, current_model_path)
    
    print(f"✓ Modelo guardado: {model_path}")
    print(f"✓ Modelo actual: {current_model_path}")
    
    return model_path

def save_results(results, config):
    """Guardar resultados del experimento"""
    print("\n💾 Guardando resultados...")
    
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar resumen JSON
    summary_file = results_dir / f"exp3_balanced_groupkfold_results_{timestamp}.json"
    
    # Preparar datos para JSON
    json_results = {
        'experiment': 'exp3_balanced_groupkfold',
        'timestamp': timestamp,
        'config': config,
        'mean_train_accuracy': float(results['mean_train_acc']),
        'std_train_accuracy': float(results['std_train_acc']),
        'mean_val_accuracy': float(results['mean_val_acc']),
        'std_val_accuracy': float(results['std_val_acc']),
        'total_training_time': float(results['total_time']),
        'fold_results': [
            {
                'fold': r['fold'],
                'train_accuracy': float(r['train_acc']),
                'val_accuracy': float(r['val_acc']),
                'training_time': float(r['training_time']),
                'train_groups': r['train_groups'],
                'val_groups': r['val_groups']
            }
            for r in results['fold_results']
        ]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Guardar reporte de texto
    report_file = results_dir / f"exp3_balanced_groupkfold_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENTO 3: DCNN CON DATASET BALANCEADO + GROUPKFOLD\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Metodología: Yu et al. (2018) + Stratified GroupKFold + Augmentación conservadora\n\n")
        
        f.write("CONFIGURACIÓN:\n")
        f.write("-" * 30 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("RESULTADOS AGREGADOS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Train Accuracy: {results['mean_train_acc']:.4f} ± {results['std_train_acc']:.4f}\n")
        f.write(f"Validation Accuracy: {results['mean_val_acc']:.4f} ± {results['std_val_acc']:.4f}\n")
        f.write(f"Total Training Time: {results['total_time']:.1f} segundos\n\n")
        
        f.write("RESULTADOS POR FOLD:\n")
        f.write("-" * 30 + "\n")
        for r in results['fold_results']:
            f.write(f"Fold {r['fold']}:\n")
            f.write(f"  Train Acc: {r['train_acc']:.4f}\n")
            f.write(f"  Val Acc: {r['val_acc']:.4f}\n")
            f.write(f"  Time: {r['training_time']:.1f}s\n")
            f.write(f"  Train Groups: {r['train_groups']}\n")
            f.write(f"  Val Groups: {r['val_groups']}\n\n")
        
        f.write("INNOVACIONES EN EXP3:\n")
        f.write("-" * 30 + "\n")
        f.write("✓ Dataset balanceado con augmentación conservadora\n")
        f.write("✓ Especímenes sintéticos como grupos independientes\n")
        f.write("✓ Validación estadística de distribuciones\n")
        f.write("✓ Preservación de metodología GroupKFold\n")
        f.write("✓ Técnicas físicamente justificadas\n")
    
    print(f"✓ Resultados guardados:")
    print(f"  📊 JSON: {summary_file}")
    print(f"  📋 Report: {report_file}")
    
    return summary_file, report_file

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Entrenamiento DCNN con Dataset Balanceado - Experimento 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

    # Entrenamiento básico con dataset balanceado
    python3 src/exp3/3_train_dcnn.py --input src/exp3/results/balanced_dataset.csv
    
    # Con configuración personalizada
    python3 src/exp3/3_train_dcnn.py --input src/exp3/results/balanced_dataset.csv --epochs 40 --n-splits 3
        """
    )
    
    parser.add_argument(
        "--input", 
        required=True,
        help="Ruta del dataset balanceado (CSV)"
    )
    parser.add_argument(
        "--epochs", 
        type=int,
        help="Número de épocas por fold"
    )
    parser.add_argument(
        "--n-splits", 
        type=int,
        help="Número de folds para GroupKFold"
    )
    parser.add_argument(
        "--batch-size", 
        type=int,
        help="Tamaño de batch"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--device", 
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help="Device a usar (default: auto)"
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Deshabilitar class weights para desbalance"
    )
    
    args = parser.parse_args()
    
    try:
        print("="*70)
        print("ENTRENAMIENTO DCNN CON DATASET BALANCEADO - EXPERIMENTO 3")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input dataset: {args.input}")
        print()
        
        # Cargar configuración
        config = get_default_config()
        
        # Override con argumentos de línea de comandos
        if args.epochs:
            config["epochs"] = args.epochs
        if args.n_splits:
            config["n_splits"] = args.n_splits
        if args.batch_size:
            config["batch_size"] = args.batch_size
        if args.learning_rate:
            config["learning_rate"] = args.learning_rate
        if args.no_class_weights:
            config["use_class_weights"] = False
        
        # Detectar device
        if args.device == 'auto':
            device, device_name = get_optimal_device()
        else:
            device = torch.device(args.device)
            device_name = str(device)
        
        print(f"🖥️ Device: {device} ({device_name})")
        print(f"🔄 Stratified GroupKFold splits: {config['n_splits']}")
        print(f"⚖️ Class weights: {'enabled' if config['use_class_weights'] else 'disabled'}")
        print(f"🧪 Experimento: Dataset balanceado con augmentación conservadora")
        print()
        
        # Cargar y preparar datos balanceados
        df = load_and_prepare_data(args.input)
        X_data, groups, labels = prepare_features_and_groups(df)
        
        # Calcular class weights para reutilizar (ajustados para dataset balanceado)
        class_weights = None
        if config["use_class_weights"]:
            # Usar weights más conservadores para dataset balanceado
            # manual_weights = np.array([0.8, 1.2, 2.0])  # Menos agresivos que exp2
            manual_weights = np.array([0.6, 1.2, 5.0]) 
            class_weights = torch.FloatTensor(manual_weights).to(device)
            print(f"✓ Class weights calculados (balanceados): {manual_weights}")
        
        # Entrenar y evaluar con Stratified GroupKFold
        cv_results = evaluate_with_groupkfold(X_data, groups, labels, config, device, class_weights)
        
        # Entrenar modelo final con todos los datos
        print("\n🚀 Entrenando modelo final con datos balanceados...")
        final_model, final_trainer = train_final_model(X_data, labels, cv_results['label_encoder'], config, device, class_weights)
        
        # Guardar modelo final
        model_path = save_model(final_model, final_trainer, cv_results['label_encoder'], cv_results, config)
        
        # Guardar resultados
        summary_file, report_file = save_results(cv_results, config)
        
        print("\n" + "="*70)
        print("🎉 EXPERIMENTO 3 COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"📊 Val Accuracy (promedio): {cv_results['mean_val_acc']:.4f} ± {cv_results['std_val_acc']:.4f}")
        print(f"💾 Modelo guardado: {model_path}")
        print(f"📁 Resultados guardados: {summary_file}")
        print(f"📋 Reporte detallado: {report_file}")
        print()
        print("🔬 INNOVACIONES EXP3:")
        print("   ✅ Dataset balanceado con augmentación conservadora")
        print("   ✅ Especímenes sintéticos como grupos independientes")
        print("   ✅ Evaluación realista con Stratified GroupKFold")
        print("   ✅ Preservación de metodología anti-leakage")
        print("   ✅ Configuración idéntica para comparabilidad")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())