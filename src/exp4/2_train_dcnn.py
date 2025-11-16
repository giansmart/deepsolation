#!/usr/bin/env python3
"""
Script de Entrenamiento DNN con Características Agregadas - Experimento 4
=========================================================================

Entrena un modelo DNN usando características estadísticas agregadas por dispositivo,
implementando el enfoque metodológicamente correcto recomendado por el experto.

ENFOQUE METODOLÓGICAMENTE CORRECTO:
- Una observación = Un dispositivo físico completo
- Características estadísticas agregadas como entrada
- GroupKFold por specimen físico (mantiene validez metodológica)
- Sin pseudo-replicación

DIFERENCIAS vs Exp1-3:
- Exp1-3: Matrices FFT → CNN
- Exp4: Características agregadas → DNN

Uso:
    python3 src/exp4/2_train_dcnn.py --input src/exp4/results/preprocessed_dataset.csv

Requisitos:
    - Dataset con características agregadas (CSV generado por 1_preprocess_signals.py)
    - ~68 observaciones (una por dispositivo físico)
    - 303 características estadísticas por dispositivo

Salidas:
    - models/dcnn_model_*.pth: Modelo entrenado
    - results/*_experiment_summary.json: Resumen completo del experimento
    - results/*_groupkfold_report.txt: Análisis detallado de GroupKFold
    - results/*_training_curves.png: Visualización del entrenamiento
    - results/*_confusion_matrix.png: Matriz de confusión
    - results/*_classification_metrics.png: Métricas por clase
    - results/*_roc_curves.png: Curvas ROC
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, 
                           roc_auc_score, classification_report, confusion_matrix)
from collections import Counter

# Configurar matplotlib para no mostrar ventanas (como en otros experimentos)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar src al path para acceder a utils
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from exp4_model import Exp4DamageNet, Exp4Trainer, get_optimal_device, create_data_loaders, plot_training_curves
from experiment_metrics import ExperimentEvaluator
import warnings
warnings.filterwarnings('ignore')

# Configuración consistente con experimentos anteriores
def get_default_config():
    project_root = Path(__file__).parent.parent.parent  # deepsolation/
    return {
        "n_splits": 5,  # GroupKFold splits (consistente con exp2/exp3)
        "test_size": 0.2,  # Para split final
        "val_size": 0.2,   # Para split final
        "batch_size": 16,  # Reducido por menor dataset
        "learning_rate": 0.001,  # Adam default
        "epochs": 100,     # Más épocas para DNN
        "patience": 20,    # Mayor paciencia para convergencia
        "dropout_rate": 0.3,  # Consistente con exp anteriores
        "models_dir": str(project_root / "src/exp4/models"),
        "results_dir": str(project_root / "src/exp4/results"),
        "random_state": 42,
        "use_class_weights": True
    }

def extract_specimen_group(specimen_name):
    """
    Extraer el grupo de specimen físico para GroupKFold
    A1, A1-2, A1-3 → 'A1'
    A10, A10-2, A10-3 → 'A10'
    """
    if '-' in specimen_name:
        return specimen_name.split('-')[0]
    else:
        return specimen_name

def load_and_prepare_data(dataset_path):
    """Cargar y preparar datos de características agregadas - Estructura Exp4"""
    print(f"📂 Cargando dataset: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Validar columnas requeridas
        required_cols = ['device_id', 'specimen', 'sensor', 'damage_level']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Columnas faltantes en dataset: {missing_cols}")
        
        print(f"📊 Dataset cargado: {len(df)} dispositivos")
        print(f"📊 Distribución de daño: {dict(df['damage_level'].value_counts())}")
        
        # Separar características de metadata
        metadata_cols = ['device_id', 'specimen', 'sensor', 'damage_level', 
                        'file_path', 'signal_length', 'duration_seconds', 
                        'window_size', 'overlap', 'sampling_rate', 'n_components']
        
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        print(f"📊 Características disponibles: {len(feature_cols)}")
        
        # Extraer características (X) y labels (y)
        X = df[feature_cols].values
        y = df['damage_level'].values
        
        # Extraer grupos para GroupKFold
        specimens = df['specimen'].values
        groups = [extract_specimen_group(spec) for spec in specimens]
        
        # Validar que no hay NaN
        if np.isnan(X).any():
            print("⚠️  Detectados valores NaN en características, rellenando con mediana")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        print(f"✓ Datos preparados: X.shape={X.shape}, y.shape={y.shape}")
        print(f"✓ Grupos únicos para GroupKFold: {len(set(groups))}")
        
        return X, y, groups, feature_cols, df
        
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        raise

def train_with_groupkfold(X, y, groups, config, feature_cols):
    """Entrenar modelo con validación cruzada GroupKFold"""
    
    print("\n🔄 Iniciando entrenamiento con GroupKFold...")
    print("=" * 60)
    
    # Preparar label encoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    num_classes = len(class_names)
    
    print(f"📊 Clases identificadas: {list(class_names)}")
    print(f"📊 Número de clases: {num_classes}")
    
    # Configurar GroupKFold
    group_kfold = GroupKFold(n_splits=config['n_splits'])
    
    # Almacenar resultados por fold
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    # Métricas acumuladas
    fold_accuracies = []
    fold_f1_scores = []
    
    # Crear directorios
    Path(config['models_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Cross-validation con GroupKFold
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y_encoded, groups)):
        print(f"\n📊 Fold {fold + 1}/{config['n_splits']}")
        print("-" * 30)
        
        # Split datos
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
        
        # Mostrar grupos en cada fold
        train_groups = set(np.array(groups)[train_idx])
        val_groups = set(np.array(groups)[val_idx])
        
        print(f"  Grupos entrenamiento: {sorted(train_groups)}")
        print(f"  Grupos validación: {sorted(val_groups)}")
        print(f"  Train: {len(X_train_fold)} dispositivos, Val: {len(X_val_fold)} dispositivos")
        
        # Verificar separación de grupos
        overlap = train_groups.intersection(val_groups)
        if overlap:
            print(f"⚠️  ADVERTENCIA: Overlap de grupos detectado: {overlap}")
        
        # Normalizar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Calcular pesos de clase para el fold actual
        if config['use_class_weights']:
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y_train_fold), 
                y=y_train_fold
            )
        else:
            class_weights = None
        
        # Crear modelo
        input_size = X_train_scaled.shape[1]
        model = Exp4DamageNet(
            input_size=input_size,
            num_classes=num_classes, 
            dropout_rate=config['dropout_rate']
        )
        
        # Crear entrenador
        trainer = Exp4Trainer(model, device='auto', class_weights=class_weights)
        
        # Crear data loaders
        train_loader, val_loader_inner, _ = create_data_loaders(
            X_train_scaled, y_train_fold, 
            batch_size=config['batch_size'],
            test_size=0.2,  # Split interno para validación durante entrenamiento
            random_state=config['random_state'] + fold
        )
        
        # Entrenar modelo
        print(f"  🚀 Entrenando modelo (Fold {fold + 1})...")
        start_time = time.time()
        
        train_hist, val_hist = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader_inner,
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            patience=config['patience'],
            verbose=False
        )
        
        training_time = time.time() - start_time
        print(f"  ✓ Entrenamiento completado en {training_time:.1f}s")
        
        # Evaluar en el fold de validación completo
        # Crear loader para el fold de validación completo
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val_fold)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Predicciones
        y_pred_fold, y_true_fold = trainer.predict(val_loader)
        
        # Obtener probabilidades para ROC
        model.eval()
        y_prob_fold = []
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(trainer.device)
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)
                y_prob_fold.extend(probs.cpu().numpy())
        y_prob_fold = np.array(y_prob_fold)
        
        # Métricas del fold
        fold_accuracy = accuracy_score(y_true_fold, y_pred_fold)
        fold_f1 = f1_score(y_true_fold, y_pred_fold, average='macro')
        
        print(f"  📈 Accuracy: {fold_accuracy:.4f}")
        print(f"  📈 F1-Score: {fold_f1:.4f}")
        
        # Almacenar resultados
        fold_results.append({
            'fold': fold + 1,
            'accuracy': fold_accuracy,
            'f1_score': fold_f1,
            'training_time': training_time,
            'train_groups': list(train_groups),
            'val_groups': list(val_groups)
        })
        
        # Acumular para métricas globales
        all_y_true.extend(y_true_fold)
        all_y_pred.extend(y_pred_fold)
        all_y_prob.extend(y_prob_fold)
        
        fold_accuracies.append(fold_accuracy)
        fold_f1_scores.append(fold_f1)
        
        # Guardar modelo del último fold
        if fold == config['n_splits'] - 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = Path(config['models_dir']) / f"dcnn_model_{timestamp}.pth"
            trainer.save_model(model_path)
            print(f"  💾 Modelo guardado: {model_path}")
            
            # Guardar curvas de entrenamiento del último fold
            curves_path = Path(config['results_dir']) / f"exp4_aggregated_training_curves.png"
            plot_training_curves(train_hist, val_hist, curves_path)
    
    # Métricas finales de cross-validation
    print(f"\n📊 RESULTADOS FINALES - CROSS VALIDATION")
    print("=" * 50)
    
    # Convertir a arrays numpy
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)
    
    # Métricas globales
    cv_accuracy = accuracy_score(all_y_true, all_y_pred)
    cv_f1_macro = f1_score(all_y_true, all_y_pred, average='macro')
    cv_f1_weighted = f1_score(all_y_true, all_y_pred, average='weighted')
    cv_kappa = cohen_kappa_score(all_y_true, all_y_pred)
    
    print(f"📈 Cross-Validation Accuracy: {cv_accuracy:.4f} (±{np.std(fold_accuracies):.4f})")
    print(f"📈 Cross-Validation F1-Macro: {cv_f1_macro:.4f} (±{np.std(fold_f1_scores):.4f})")
    print(f"📈 Cross-Validation F1-Weighted: {cv_f1_weighted:.4f}")
    print(f"📈 Cohen's Kappa: {cv_kappa:.4f}")
    
    # AUC por clase (One-vs-Rest)
    try:
        auc_scores = {}
        for i, class_name in enumerate(class_names):
            y_true_binary = (all_y_true == i).astype(int)
            y_score_binary = all_y_prob[:, i]
            auc = roc_auc_score(y_true_binary, y_score_binary)
            auc_scores[class_name] = auc
        
        auc_macro = np.mean(list(auc_scores.values()))
        print(f"📈 AUC-ROC Macro: {auc_macro:.4f}")
        
    except Exception as e:
        auc_scores = {}
        auc_macro = 0
        print(f"⚠️  No se pudo calcular AUC: {e}")
    
    # Reporte de clasificación detallado
    print(f"\n📊 REPORTE DE CLASIFICACIÓN:")
    print("-" * 40)
    class_report = classification_report(
        all_y_true, all_y_pred, 
        target_names=[str(name) for name in class_names],
        output_dict=True
    )
    print(classification_report(
        all_y_true, all_y_pred, 
        target_names=[str(name) for name in class_names]
    ))
    
    # Preparar resultados finales
    final_results = {
        'experiment_name': 'exp4_aggregated_groupkfold',
        'timestamp': datetime.now().isoformat(),
        'methodology': 'Statistical Aggregated Features + GroupKFold',
        'approach': 'One observation = One physical device',
        'model_architecture': 'Fully Connected DNN',
        'cross_validation': {
            'split': 'cross_validation',
            'n_splits': config['n_splits'],
            'accuracy': cv_accuracy,
            'f1_macro': cv_f1_macro,
            'f1_weighted': cv_f1_weighted,
            'cohen_kappa': cv_kappa,
            'auc_macro': auc_macro,
            'per_class': {}
        },
        'fold_results': fold_results,
        'data_leakage': {
            'alerts': [],
            'metrics': {
                'test_val_gap': 0.0,  # No hay gap porque usamos CV
                'test_train_gap': 0.0,
                'cv_std': np.std(fold_accuracies),
                'cv_mean': np.mean(fold_accuracies),
                'cv_coefficient_variation': np.std(fold_accuracies) / np.mean(fold_accuracies)
            },
            'risk_level': 'LOW' if np.std(fold_accuracies) < 0.1 else 'MEDIUM'
        },
        'config': config,
        'feature_info': {
            'n_features': len(feature_cols),
            'feature_type': 'statistical_aggregated'
        }
    }
    
    # Métricas por clase
    for class_name in class_names:
        class_idx = np.where(label_encoder.classes_ == class_name)[0][0]
        if str(class_idx) in class_report:
            final_results['cross_validation']['per_class'][class_name] = {
                'precision': class_report[str(class_idx)]['precision'],
                'recall': class_report[str(class_idx)]['recall'],
                'f1_score': class_report[str(class_idx)]['f1-score'],
                'support': str(class_report[str(class_idx)]['support']),
                'auc': auc_scores.get(class_name, 0.0)
            }
    
    return final_results, all_y_true, all_y_pred, all_y_prob, label_encoder

def create_visualizations(y_true, y_pred, y_prob, label_encoder, config):
    """Crear visualizaciones consistentes con experimentos anteriores usando el mismo evaluador"""
    
    print("\n🎨 Generando visualizaciones...")
    
    class_names = label_encoder.classes_
    results_dir = Path(config['results_dir'])
    
    # matplotlib ya está configurado globalmente con 'Agg' backend
    
    # Usar el mismo evaluador que otros experimentos para consistencia total
    evaluator = ExperimentEvaluator(
        experiment_name="exp4_aggregated_groupkfold",
        results_dir=str(results_dir)
    )
    
    # 1. Matriz de confusión (usando el evaluador estándar)
    evaluator.plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
    cm_path = results_dir / 'exp4_aggregated_groupkfold_confusion_matrix.png'
    print(f"✓ Matriz de confusión: {cm_path}")
    
    # 2. Métricas de clasificación por clase (convertir a estructura esperada)
    from sklearn.metrics import classification_report
    sklearn_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Convertir a estructura compatible con evaluator
    metrics_for_evaluator = {
        'per_class': {}
    }
    
    for class_name in class_names:
        if class_name in sklearn_report:
            metrics_for_evaluator['per_class'][class_name] = {
                'precision': sklearn_report[class_name]['precision'],
                'recall': sklearn_report[class_name]['recall'],
                'f1_score': sklearn_report[class_name]['f1-score']  # Note: sklearn usa 'f1-score', evaluator espera 'f1_score'
            }
    
    evaluator.plot_classification_metrics(metrics_for_evaluator, class_names)
    metrics_path = results_dir / 'exp4_aggregated_groupkfold_classification_metrics.png'
    print(f"✓ Métricas por clase: {metrics_path}")
    
    # 3. Curvas ROC (usando el evaluador estándar)
    evaluator.plot_roc_curves(y_true, y_prob, class_names)
    roc_path = results_dir / 'exp4_aggregated_groupkfold_roc_curves.png'
    print(f"✓ Curvas ROC: {roc_path}")
    
    generated_plots = [
        'exp4_aggregated_groupkfold_training_curves.png',
        'exp4_aggregated_groupkfold_confusion_matrix.png', 
        'exp4_aggregated_groupkfold_classification_metrics.png',
        'exp4_aggregated_groupkfold_roc_curves.png'
    ]
    
    return generated_plots

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Entrenamiento DNN con características agregadas - Experimento 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

    # Entrenamiento con dataset por defecto
    python src/exp4/2_train_dcnn.py
    
    # Especificar dataset personalizado
    python src/exp4/2_train_dcnn.py --input src/exp4/results/mi_dataset.csv
    
    # Configuración personalizada
    python src/exp4/2_train_dcnn.py --epochs 150 --batch-size 8 --learning-rate 0.0005

ENFOQUE METODOLÓGICAMENTE CORRECTO:
✅ Una observación = Un dispositivo físico completo
✅ Características estadísticas agregadas
✅ GroupKFold por specimen físico
✅ Sin pseudo-replicación
        """
    )
    
    parser.add_argument(
        "--input", 
        default="src/exp4/results/preprocessed_dataset.csv",
        help="Ruta al dataset de características agregadas"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int,
        default=100,
        help="Número de épocas de entrenamiento (default: 100)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int,
        default=16,
        help="Tamaño de batch (default: 16)"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float,
        default=0.001,
        help="Tasa de aprendizaje (default: 0.001)"
    )
    
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Deshabilitar pesos de clase automáticos"
    )
    
    args = parser.parse_args()
    
    try:
        print("="*80)
        print("ENTRENAMIENTO DNN - EXPERIMENTO 4")
        print("Enfoque con Características Estadísticas Agregadas")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 ENFOQUE METODOLÓGICAMENTE CORRECTO")
        print(f"📊 Una observación = Un dispositivo físico completo")
        print(f"🔬 Características estadísticas agregadas")
        print(f"✅ GroupKFold por specimen físico")
        print()
        
        # Configuración
        config = get_default_config()
        config.update({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'use_class_weights': not args.no_class_weights
        })
        
        # Determinar ruta del dataset
        project_root = Path(__file__).parent.parent.parent  # deepsolation/
        dataset_path = project_root / args.input
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
        
        print(f"📂 Dataset: {dataset_path}")
        print(f"⚙️  Configuración:")
        print(f"   - Épocas: {config['epochs']}")
        print(f"   - Batch size: {config['batch_size']}")
        print(f"   - Learning rate: {config['learning_rate']}")
        print(f"   - Pesos de clase: {'Sí' if config['use_class_weights'] else 'No'}")
        print()
        
        # Cargar y preparar datos
        X, y, groups, feature_cols, df_original = load_and_prepare_data(dataset_path)
        
        # Verificar dispositivo de cómputo
        device = get_optimal_device()
        print(f"🖥️  Dispositivo: {device}")
        print()
        
        # Entrenamiento con cross-validation
        start_time = time.time()
        
        results, y_true, y_pred, y_prob, label_encoder = train_with_groupkfold(
            X, y, groups, config, feature_cols
        )
        
        total_time = time.time() - start_time
        
        # Crear visualizaciones
        generated_plots = create_visualizations(y_true, y_pred, y_prob, label_encoder, config)
        results['plots_generated'] = generated_plots
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON summary
        json_path = Path(config['results_dir']) / f"exp4_aggregated_groupkfold_experiment_summary.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Text summary
        txt_path = Path(config['results_dir']) / f"exp4_aggregated_groupkfold_experiment_summary.txt"
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENTO 4 - CARACTERÍSTICAS ESTADÍSTICAS AGREGADAS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Metodología: {results['methodology']}\n")
            f.write(f"Enfoque: {results['approach']}\n")
            f.write(f"Arquitectura: {results['model_architecture']}\n\n")
            
            f.write("RESULTADOS CROSS-VALIDATION:\n")
            f.write("-"*40 + "\n")
            cv = results['cross_validation']
            f.write(f"Accuracy: {cv['accuracy']:.4f}\n")
            f.write(f"F1-Score Macro: {cv['f1_macro']:.4f}\n")
            f.write(f"F1-Score Weighted: {cv['f1_weighted']:.4f}\n")
            f.write(f"Cohen's Kappa: {cv['cohen_kappa']:.4f}\n")
            f.write(f"AUC-ROC Macro: {cv['auc_macro']:.4f}\n\n")
            
            f.write("MÉTRICAS POR CLASE:\n")
            f.write("-"*30 + "\n")
            for class_name, metrics in cv['per_class'].items():
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")
                f.write(f"  AUC: {metrics['auc']:.4f}\n\n")
            
            f.write("INFORMACIÓN TÉCNICA:\n")
            f.write("-"*25 + "\n")
            f.write(f"Características: {results['feature_info']['n_features']}\n")
            f.write(f"Tipo de características: {results['feature_info']['feature_type']}\n")
            f.write(f"Dispositivos totales: {len(df_original)}\n")
            f.write(f"Tiempo total: {total_time:.1f}s\n\n")
            
            f.write("VALIDEZ METODOLÓGICA:\n")
            f.write("-"*25 + "\n")
            f.write(f"✅ Una observación = Un dispositivo físico\n")
            f.write(f"✅ GroupKFold por specimen\n")
            f.write(f"✅ Sin pseudo-replicación\n")
            f.write(f"✅ Alineación: Observación = Inferencia\n")
        
        # Classification report detallado
        report_path = Path(config['results_dir']) / f"exp4_aggregated_groupkfold_cross_validation_classification_report.txt"
        with open(report_path, 'w') as f:
            from sklearn.metrics import classification_report
            f.write(classification_report(
                y_true, y_pred, 
                target_names=[str(name) for name in label_encoder.classes_]
            ))
        
        print(f"\n" + "="*80)
        print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"📊 Resultados JSON: {json_path}")
        print(f"📋 Resumen: {txt_path}")
        print(f"📈 Reporte clasificación: {report_path}")
        print(f"🎨 Gráficos generados: {len(generated_plots)}")
        print(f"⏱️  Tiempo total: {total_time:.1f} segundos")
        print()
        print("🎯 ENFOQUE METODOLÓGICAMENTE CORRECTO IMPLEMENTADO:")
        print("   ✅ Una observación = Un dispositivo físico completo")
        print("   ✅ Características estadísticas agregadas")
        print("   ✅ GroupKFold sin data leakage")
        print("   ✅ Métricas científicamente válidas")
        print(f"   ✅ Accuracy final: {results['cross_validation']['accuracy']:.4f}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())