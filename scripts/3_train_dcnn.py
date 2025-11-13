#!/usr/bin/env python3
"""
Script de Entrenamiento DCNN
============================

Entrena el modelo DCNN siguiendo la metodología Yu et al. (2018) usando
el dataset procesado generado por preprocess_signals.py

Uso:
    python scripts/train_dcnn.py --input DATASET_PATH [--config CONFIG_FILE]

Requisitos:
    - Dataset procesado (CSV generado por preprocess_signals.py)
    - Opcionalmente dataset balanceado (generado por balance_dataset.py)

Salidas:
    - models/dcnn_model.pth: Modelo entrenado
    - results/training_history.png: Gráficas de entrenamiento
    - results/evaluation_report.txt: Reporte de evaluación
"""

import argparse
import sys
from pathlib import Path
import json
import time
import pandas as pd
import torch
from datetime import datetime

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dcnn_model import DCNNDamageNet, DCNNTrainer, get_optimal_device
from utils import prepare_dataset_from_csv

# Configuración por defecto
DEFAULT_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.2,
    "batch_size": 50,
    "learning_rate": 0.0035,
    "epochs": 100,
    "patience": 15,
    "dropout_rate": 0.3,
    "models_dir": "models",
    "results_dir": "results",
    "random_state": 42
}

def load_config(config_file=None):
    """Cargar configuración desde archivo JSON o usar defaults"""
    config = DEFAULT_CONFIG.copy()
    
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
                print(f"✓ Configuración cargada desde: {config_file}")
        except Exception as e:
            print(f"⚠️ Error cargando configuración: {e}")
            print("Usando configuración por defecto")
    
    return config

def load_dataset(dataset_path):
    """Cargar y validar dataset CSV"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
    
    print(f"📂 Cargando dataset: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Validar columnas requeridas
        required_cols = ['specimen', 'sensor', 'component_NS', 'component_EW', 'component_UD', 'damage_level']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Columnas faltantes en dataset: {missing_cols}")
        
        print(f"✓ Dataset cargado: {len(df):,} observaciones")
        print(f"✓ Especímenes: {df['specimen'].nunique()}")
        
        # Mostrar distribución de clases
        print("📊 Distribución de clases:")
        damage_counts = df['damage_level'].value_counts().sort_index()
        total_samples = len(df)
        
        for damage_level, count in damage_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   {damage_level}: {count:,} ({percentage:.1f}%)")
        
        return df
        
    except Exception as e:
        raise Exception(f"Error cargando dataset: {e}")

def prepare_data_loaders(df, config):
    """Preparar DataLoaders para PyTorch"""
    print("\n📦 Preparando DataLoaders...")
    
    try:
        train_loader, val_loader, test_loader, label_encoder, input_shape = prepare_dataset_from_csv(
            df, 
            test_size=config["test_size"],
            val_size=config["val_size"],
            batch_size=config["batch_size"],
            random_state=config["random_state"]
        )
        
        print(f"✓ Training batches: {len(train_loader)}")
        if val_loader:
            print(f"✓ Validation batches: {len(val_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")
        print(f"✓ Classes: {list(label_encoder.classes_)}")
        print(f"✓ Input shape: {input_shape}")
        
        return train_loader, val_loader, test_loader, label_encoder, input_shape
        
    except Exception as e:
        raise Exception(f"Error preparando DataLoaders: {e}")

def create_model(input_shape, num_classes, config):
    """Crear modelo DCNN"""
    print("\n🏗️ Creando modelo DCNN...")
    
    model = DCNNDamageNet(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=config["dropout_rate"]
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Modelo inicializado")
    print(f"✓ Parámetros totales: {total_params:,}")
    print(f"✓ Parámetros entrenables: {trainable_params:,}")
    print(f"✓ Input: {input_shape}")
    print(f"✓ Output: {num_classes} clases")
    print(f"✓ Dropout rate: {config['dropout_rate']}")
    
    return model

def train_model(model, train_loader, val_loader, config, device):
    """Entrenar modelo"""
    print("\n🚀 Iniciando entrenamiento...")
    print(f"Metodología Yu et al. (2018)")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Early stopping patience: {config['patience']}")
    print(f"Device: {device}")
    print()
    
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
    
    print(f"\n✓ Entrenamiento completado en {training_time:.1f} segundos")
    
    return trainer

def evaluate_model(trainer, test_loader, label_encoder, config):
    """Evaluar modelo"""
    print("\n📊 Evaluando modelo...")
    
    results = trainer.evaluate_model(test_loader, class_names=label_encoder.classes_)
    
    print("="*50)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*50)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print()
    print("Classification Report:")
    print(results['classification_report'])
    
    return results

def save_model(model, trainer, label_encoder, results, config):
    """Guardar modelo y resultados"""
    print("\n💾 Guardando modelo...")
    
    # Crear directorios
    models_dir = Path(config["models_dir"])
    models_dir.mkdir(exist_ok=True)
    
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(exist_ok=True)
    
    # Guardar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"dcnn_model_{timestamp}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_shape': model.input_shape,
            'num_classes': model.num_classes,
            'dropout_rate': model.dropout_rate
        },
        'label_encoder': label_encoder,
        'training_config': config,
        'results': results,
        'timestamp': timestamp
    }, model_path)
    
    # Guardar también como modelo actual
    current_model_path = models_dir / "dcnn_model_current.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_shape': model.input_shape,
            'num_classes': model.num_classes,
            'dropout_rate': model.dropout_rate
        },
        'label_encoder': label_encoder,
        'training_config': config,
        'results': results,
        'timestamp': timestamp
    }, current_model_path)
    
    # Guardar gráficas de entrenamiento
    history_plot_path = results_dir / f"training_history_{timestamp}.png"
    trainer.plot_training_history(save_path=str(history_plot_path))
    
    # Guardar reporte de evaluación
    report_path = results_dir / f"evaluation_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE EVALUACIÓN - DCNN DAMAGE IDENTIFICATION\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Metodología: Yu et al. (2018)\n\n")
        
        f.write("CONFIGURACIÓN DE ENTRENAMIENTO:\n")
        f.write("-" * 30 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("RESULTADOS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
        f.write("\n")
        
        f.write("ARCHIVOS GENERADOS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Modelo: {model_path}\n")
        f.write(f"Modelo actual: {current_model_path}\n")
        f.write(f"Gráficas: {history_plot_path}\n")
        f.write(f"Reporte: {report_path}\n")
    
    print(f"✓ Modelo guardado: {model_path}")
    print(f"✓ Modelo actual: {current_model_path}")
    print(f"✓ Gráficas: {history_plot_path}")
    print(f"✓ Reporte: {report_path}")
    
    return model_path, report_path

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo DCNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

    # Entrenamiento básico
    python scripts/train_dcnn.py --input results/preprocessed_dataset.csv
    
    # Con dataset balanceado
    python scripts/train_dcnn.py --input results/balanced_dataset.csv
    
    # Con configuración personalizada
    python scripts/train_dcnn.py --input results/dataset.csv --config config/training.json
    
    # Entrenamiento rápido (pocas épocas)
    python scripts/train_dcnn.py --input results/dataset.csv --epochs 20

El archivo de configuración debe tener formato JSON:
{
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 32,
    "patience": 10
}
        """
    )
    
    parser.add_argument(
        "--input", 
        required=True,
        help="Ruta del dataset procesado (CSV)"
    )
    parser.add_argument(
        "--config", 
        help="Archivo de configuración JSON"
    )
    parser.add_argument(
        "--epochs", 
        type=int,
        help="Número de épocas (override config)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int,
        help="Tamaño de batch (override config)"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float,
        help="Learning rate (override config)"
    )
    parser.add_argument(
        "--device", 
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help="Device a usar (default: auto)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Modo silencioso"
    )
    
    args = parser.parse_args()
    
    try:
        print("="*70)
        print("ENTRENAMIENTO DCNN - STRUCTURAL DAMAGE IDENTIFICATION")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input dataset: {args.input}")
        print()
        
        # Cargar configuración
        config = load_config(args.config)
        
        # Override con argumentos de línea de comandos
        if args.epochs:
            config["epochs"] = args.epochs
        if args.batch_size:
            config["batch_size"] = args.batch_size
        if args.learning_rate:
            config["learning_rate"] = args.learning_rate
        
        # Detectar device
        if args.device == 'auto':
            device, device_name = get_optimal_device()
        else:
            device = torch.device(args.device)
            device_name = str(device)
        
        print(f"🖥️ Device: {device} ({device_name})")
        print()
        
        # Cargar dataset
        df = load_dataset(args.input)
        
        # Preparar datos
        train_loader, val_loader, test_loader, label_encoder, input_shape = prepare_data_loaders(df, config)
        
        # Crear modelo
        num_classes = len(label_encoder.classes_)
        model = create_model(input_shape, num_classes, config)
        
        # Entrenar
        trainer = train_model(model, train_loader, val_loader, config, device)
        
        # Evaluar
        results = evaluate_model(trainer, test_loader, label_encoder, config)
        
        # Guardar
        model_path, report_path = save_model(model, trainer, label_encoder, results, config)
        
        print("\n" + "="*70)
        print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"📊 Test Accuracy: {results['accuracy']:.4f}")
        print(f"💾 Modelo guardado: {model_path}")
        print(f"📋 Reporte: {report_path}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())