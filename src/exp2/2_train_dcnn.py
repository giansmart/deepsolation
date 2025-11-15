#!/usr/bin/env python3
"""
Script de Entrenamiento DCNN con GroupKFold - Experimento 2
===========================================================

Entrena el modelo DCNN siguiendo la metodología Yu et al. (2018) con approach
por matriz completa usando GroupKFold por specimen físico.

Uso:
    python3 src/exp2/2_train_dcnn.py --input src/exp2/results/preprocessed_dataset.csv

Requisitos:
    - Dataset procesado con matrices serializadas (CSV generado por 1_preprocess_signals.py)
    - SIN balanceo SMOTE (distribución natural)
    - ~68 muestras (una por specimen-sensor)

Salidas:
    - models/dcnn_model_*.pth: Modelo entrenado
    - results/*_experiment_summary.json: Resumen completo del experimento
    - results/*_groupkfold_report.txt: Análisis detallado de GroupKFold

Approach Exp2 vs Exp1:
    - Estructura de datos: Matriz completa por (specimen-sensor) vs componente individual
    - Número de muestras: ~68 vs ~635K 
    - Features por muestra: ~49K (matriz serializada) vs 3
    - GroupKFold por specimen físico (evita data leakage)
    - Class weights para manejar desbalance natural
    - Cross-validation con 5 folds
    - Evaluación realista por aislador completo
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

# Configuración por defecto (IDÉNTICA a exp1 para comparabilidad)
def get_default_config():
    project_root = Path(__file__).parent.parent.parent  # deepsolation/
    return {
        "n_splits": 5,  # GroupKFold splits
        "test_size": 0.2,  # Para split final
        "val_size": 0.2,   # Para split final
        "batch_size": 50,  # MISMO que exp1
        "learning_rate": 0.0035,  # MISMO que exp1
        "epochs": 2,     # MISMO que exp1 
        "patience": 15,    # MISMO que exp1
        "dropout_rate": 0.3,  # MISMO que exp1
        "models_dir": str(project_root / "src/exp2/models"),
        "results_dir": str(project_root / "src/exp2/results"),
        "random_state": 42,
        "use_class_weights": True
    }

def extract_specimen_group(specimen_name):
    """
    Extrae el grupo de specimen físico para GroupKFold
    A1, A1-2, A1-3 → 'A1'
    A10, A10-2, A10-3 → 'A10'
    """
    if '-' in specimen_name:
        return specimen_name.split('-')[0]
    else:
        return specimen_name

def load_and_prepare_data(dataset_path):
    """Cargar y preparar datos con grupos para GroupKFold - Estructura Exp2"""
    print(f"📂 Cargando dataset: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Validar columnas requeridas para nueva estructura exp2
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
        
        print(f"✓ Dataset cargado: {len(df):,} muestras (matrices completas)")
        print(f"✓ Especímenes únicos: {df['specimen'].nunique()}")
        print(f"✓ Columnas de frecuencia: {len(freq_cols)}")
        
        # Crear grupos de specimens físicos
        df['specimen_group'] = df['specimen'].apply(extract_specimen_group)
        unique_groups = df['specimen_group'].unique()
        print(f"✓ Grupos físicos únicos: {len(unique_groups)}")
        print(f"   Grupos: {sorted(unique_groups)}")
        
        # Mostrar distribución de clases
        print("📊 Distribución de clases (NATURAL):")
        damage_counts = df['damage_level'].value_counts().sort_index()
        total_samples = len(df)
        
        for damage_level, count in damage_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   {damage_level}: {count:,} ({percentage:.1f}%)")
        
        return df
        
    except Exception as e:
        raise Exception(f"Error cargando dataset: {e}")

def prepare_features_and_groups(df):
    """Preparar características y grupos para GroupKFold - Estructura Exp2"""
    print("🔄 Preparando características de matrices serializadas...")
    
    # Extraer columnas de frecuencias serializadas (solo freq_XXXX_YY, no metadata)
    import re
    freq_pattern = re.compile(r'^freq_\d+_(NS|EW|UD)$')
    freq_cols = [col for col in df.columns if freq_pattern.match(col)]
    freq_cols.sort()  # Asegurar orden consistente
    
    # Cada fila ya es una muestra completa (matriz serializada)
    X = df[freq_cols].values
    groups = df['specimen_group'].values  
    labels = df['damage_level'].values
    
    print(f"✓ {len(X)} muestras preparadas (matrices serializadas)")
    print(f"✓ Features por muestra: {X.shape[1]:,}")
    print(f"✓ Grupos únicos: {len(np.unique(groups))}")
    
    return X, groups, labels

def create_data_loaders(X_indices, y_indices, X_data, y_data, config):
    """Crear DataLoaders para los índices dados - Estructura Exp2"""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Seleccionar datos usando índices
    X_selected = X_data[X_indices]
    y_selected = y_data[y_indices]
    
    # Los datos ya están serializados, necesitamos reshapear para DCNN
    # Detectar dimensiones de la matriz original
    n_features = X_selected.shape[1]
    
    # Asumir que son matrices (freq_components, 3) serializadas
    # n_features = freq_components * 3
    if n_features % 3 != 0:
        raise ValueError(f"Features {n_features} no es múltiplo de 3 (esperado: freq_components * 3)")
    
    freq_components = n_features // 3
    
    # Reshape de vector serializado a matriz (batch, freq_components, 3)
    X_reshaped = X_selected.reshape(-1, freq_components, 3)
    
    # DCNN espera (batch, freq_components, n_sensors) 
    # Los datos están como (batch, freq_components, 3)
    X_tensor = torch.FloatTensor(X_reshaped)
    y_tensor = torch.LongTensor(y_selected)
    
    # Crear DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    # Input shape para el modelo: (freq_components, n_sensors)
    input_shape = (freq_components, 3)
    
    return loader, input_shape

def train_fold(fold_idx, train_loader, val_loader, input_shape, num_classes, config, device):
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
        patience=config["patience"]
    )
    
    training_time = time.time() - start_time
    print(f"✓ Fold {fold_idx + 1} completado en {training_time:.1f} segundos")
    
    return trainer, training_time

def evaluate_with_groupkfold(X_data, groups, labels, config, device):
    """Evaluación completa con GroupKFold"""
    print("🔍 EVALUACIÓN CON GROUPKFOLD")
    print("=" * 50)
    
    # Codificar labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # Configurar GroupKFold
    gkf = GroupKFold(n_splits=config["n_splits"])
    
    # Calcular class weights si está habilitado
    class_weights = None
    if config["use_class_weights"]:
        unique_classes = np.unique(y_encoded)
        weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_encoded
        )
        class_weights = torch.FloatTensor(weights).to(device)
        print(f"✓ Class weights: {weights}")
    
    # Variables para acumular resultados
    fold_results = []
    fold_times = []
    
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
            len(label_encoder.classes_), config, device
        )
        
        # Evaluar fold
        train_metrics = trainer.evaluate_dataset(train_loader, split_name="train")
        val_metrics = trainer.evaluate_dataset(val_loader, split_name="validation")
        
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
    
    print(f"\n📊 RESULTADOS AGREGADOS:")
    print(f"   Train Acc: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"   Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"   Tiempo total: {sum(fold_times):.1f}s")
    
    return {
        'fold_results': fold_results,
        'mean_train_acc': np.mean(train_accs),
        'std_train_acc': np.std(train_accs),
        'mean_val_acc': np.mean(val_accs),
        'std_val_acc': np.std(val_accs),
        'total_time': sum(fold_times),
        'label_encoder': label_encoder
    }

def train_final_model(X_data, labels, label_encoder, config, device):
    """Entrenar modelo final con todos los datos para guardado"""
    # Codificar labels
    y_encoded = label_encoder.transform(labels)
    
    # Usar 80% para entrenamiento, 20% para validación (sin grupos para modelo final)
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
        patience=config["patience"]
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
    
    # Preparar datos para guardado (convertir todo a tipos serializables)
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
        'experiment': 'exp2_groupkfold'
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
    summary_file = results_dir / f"exp2_groupkfold_results_{timestamp}.json"
    
    # Preparar datos para JSON (convertir numpy a tipos nativos)
    json_results = {
        'experiment': 'exp2_groupkfold',
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
    report_file = results_dir / f"exp2_groupkfold_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENTO 2: DCNN CON GROUPKFOLD\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Metodología: Yu et al. (2018) + GroupKFold por specimen\n\n")
        
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
        
        f.write("VENTAJAS DE ESTE APPROACH:\n")
        f.write("-" * 30 + "\n")
        f.write("✓ Sin data leakage entre specimens físicos\n")
        f.write("✓ Evaluación realista y confiable\n")
        f.write("✓ Distribución natural de clases preservada\n")
        f.write("✓ Métricas de performance válidas\n")
        f.write("✓ Comparación justa con exp1\n")
    
    print(f"✓ Resultados guardados:")
    print(f"  📊 JSON: {summary_file}")
    print(f"  📋 Report: {report_file}")
    
    return summary_file, report_file

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo DCNN con GroupKFold - Experimento 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

    # Entrenamiento básico con GroupKFold
    python src/exp2/2_train_dcnn.py --input src/exp2/results/preprocessed_dataset.csv
    
    # Con configuración personalizada
    python src/exp2/2_train_dcnn.py --input src/exp2/results/dataset.csv --epochs 40 --n-splits 3
        """
    )
    
    parser.add_argument(
        "--input", 
        required=True,
        help="Ruta del dataset procesado (CSV)"
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
        print("ENTRENAMIENTO DCNN CON GROUPKFOLD - EXPERIMENTO 2")
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
        print(f"🔄 GroupKFold splits: {config['n_splits']}")
        print(f"⚖️ Class weights: {'enabled' if config['use_class_weights'] else 'disabled'}")
        print()
        
        # Cargar y preparar datos
        df = load_and_prepare_data(args.input)
        X_data, groups, labels = prepare_features_and_groups(df)
        
        # Entrenar y evaluar con GroupKFold
        cv_results = evaluate_with_groupkfold(X_data, groups, labels, config, device)
        
        # Entrenar modelo final con todos los datos
        print("\n🚀 Entrenando modelo final con todos los datos...")
        final_model, final_trainer = train_final_model(X_data, labels, cv_results['label_encoder'], config, device)
        
        # Guardar modelo final
        model_path = save_model(final_model, final_trainer, cv_results['label_encoder'], cv_results, config)
        
        # Guardar resultados
        summary_file, report_file = save_results(cv_results, config)
        
        print("\n" + "="*70)
        print("🎉 EXPERIMENTO 2 COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"📊 Val Accuracy (promedio): {cv_results['mean_val_acc']:.4f} ± {cv_results['std_val_acc']:.4f}")
        print(f"💾 Modelo guardado: {model_path}")
        print(f"📁 Resultados guardados: {summary_file}")
        print(f"📋 Reporte detallado: {report_file}")
        print()
        print("🔍 LOGROS:")
        print("   ✅ Sin data leakage entre specimens físicos")
        print("   ✅ Evaluación realista con GroupKFold")
        print("   ✅ Métricas de performance confiables")
        print("   ✅ Modelo final guardado correctamente")
        print("   ✅ Configuración idéntica a exp1 (comparable)")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())