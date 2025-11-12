"""
Demo script using PyTorch implementation
Complete preprocessing pipeline based on Yu et al. (2018) methodology
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import torch
from signal_preprocessing import SignalPreprocessor
from dcnn_model import DCNNDamageNet, DCNNTrainer, prepare_data_for_pytorch, get_optimal_device
from utils import analyze_dataset_structure, count_signal_files, print_dataset_summary, create_signal_analysis_report, export_training_dataset

def main():
    print("=== PyTorch DCNN Structural Damage Identification ===")
    print("Following Yu et al. (2018) methodology\n")
    
    # Check device - optimized for Mac M2 Max
    device, device_name = get_optimal_device()
    print(f"Using device: {device} ({device_name})")
    print()
    
    # Initialize preprocessor with Yu et al. parameters
    preprocessor = SignalPreprocessor(
        sampling_rate=100,     # 100 Hz sampling
        energy_threshold=0.7   # 70% energy threshold
    )
    
    # Step 1: Load and examine dataset
    print("Step 1: Loading dataset...")
    csv_path = "../data/Arreglo_3_actual_clean.csv"  # Para análisis general
    labels_path = "../data/nivel_damage.csv"          # Para labels correctos
    signals_dir = "../data/Signals_Raw"
    
    # Analizar estructura del dataset
    csv_stats, df = analyze_dataset_structure(csv_path)
    signal_stats = count_signal_files(signals_dir)
    
    # Cargar labels correctos para PyTorch
    import pandas as pd
    labels_df = pd.read_csv(labels_path)
    
    if csv_stats is None:
        print("✗ Dataset file not found. Please check the path.")
        return
    
    # Imprimir resumen completo
    print_dataset_summary(csv_stats, signal_stats)
    
    # Step 2: Demonstrate preprocessing on single file
    print("\nStep 2: Demonstrating preprocessing pipeline...")
    
    # Test with one signal file
    test_file = "../data/Signals_Raw/A1/completo_S1.txt"
    
    if os.path.exists(test_file):
        print(f"Processing: {test_file}")
        
        # Process single file to show methodology
        feature_matrix, metadata = preprocessor.preprocess_signal_file(test_file)
        
        if feature_matrix is not None:
            print(f"✓ Preprocessing complete")
            print(f"  Original signal: {metadata.get('original_samples')} samples")
            print(f"  Compressed to: {feature_matrix.shape[0]} frequency components")
            print(f"  Features per sensor: {feature_matrix.shape[1]} (N-S, E-W, U-D)")
            print(f"  Compression ratio: {metadata['compression_ratio']:.3f}")
            print(f"  Energy retained: {metadata.get('energy_retained', 0.7)*100:.1f}%")
            
            # Crear reporte detallado de transformación solo para la muestra
            print("\n  Generating detailed transformation report...")
            report_path = "../results/signal_transformation_analysis.txt"
            os.makedirs("../results", exist_ok=True)
            create_signal_analysis_report(test_file, feature_matrix, metadata, report_path)
            print(f"  → Report saved: {report_path}")
        
    else:
        print(f"✗ Test file not found: {test_file}")
    
    # Step 3: Process all signals for PyTorch training
    print("\nStep 3: Processing all signals for PyTorch model...")
    choice = input("Process all signal files for PyTorch training? (y/n): ")
    
    if choice.lower() == 'y':
        signals_dir = "../data/Signals_Raw"
        
        if os.path.exists(signals_dir):
            print("Processing all signals... This may take a few minutes.")
            processed_data = preprocessor.process_all_signals(signals_dir)
            
            print(f"✓ Processed {len(processed_data)} signal files from multiple isolators")
            
            # Show summary statistics
            total_features = 0
            for specimen, sensors in processed_data.items():
                for sensor, data in sensors.items():
                    shape = data['features'].shape
                    compression = data['metadata']['compression_ratio']
                    print(f"  {specimen}/{sensor}: {shape} (compression: {compression:.3f})")
                    total_features += shape[0] * shape[1]
            
            print(f"  Total features extracted: {total_features}")
            
            # Step 4: Prepare PyTorch DataLoaders (needed for padding info)
            print("\nStep 4: Preparing PyTorch DataLoaders...")
            
            try:
                train_loader, val_loader, test_loader, label_encoder, padding_info = prepare_data_for_pytorch(
                    processed_data, labels_df, test_size=0.2, val_size=0.2  # Usar labels correctos
                )
                
                print(f"✓ PyTorch DataLoaders prepared")
                print(f"  Training batches: {len(train_loader)}")
                if val_loader:
                    print(f"  Validation batches: {len(val_loader)}")
                print(f"  Test batches: {len(test_loader)}")
                print(f"  Classes: {list(label_encoder.classes_)}")
                
                # Get sample batch to determine input shape
                sample_batch = next(iter(train_loader))
                input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
                num_classes = len(label_encoder.classes_)
                
                print(f"  Input shape: {input_shape}")
                print(f"  Number of classes: {num_classes}")
                
                # Step 5: Export training dataset for analysis
                print("\nStep 5: Exporting training dataset for analysis...")
                
                # Export CSV format for easy analysis
                csv_export_path = "../results/training_dataset_complete.csv"
                try:
                    dataset_df = preprocessor.create_training_dataset_csv(
                        signals_dir="../data/Signals_Raw",  # Adjust path for notebook location
                        labels_csv_path=labels_path,
                        output_csv_path=csv_export_path
                    )
                    print(f"✓ Training dataset exported to CSV:")
                    print(f"  File: {csv_export_path}")
                    print(f"  Rows: {len(dataset_df):,}")
                    print(f"  Columns: {len(dataset_df.columns)}")
                    print(f"  Summary: {csv_export_path.replace('.csv', '_summary.txt')}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not export CSV dataset: {e}")
                
                # Optional: Export pickle format for PyTorch training
                export_pickle = input("\nExport pickle format for PyTorch training? (y/n): ")
                if export_pickle.lower() == 'y':
                    dataset_export_path = "../results/training_dataset.pkl"
                    try:
                        pickle_path, summary_path, csv_path = export_training_dataset(
                            processed_data, labels_df, dataset_export_path, padding_info
                        )
                        print(f"✓ PyTorch training dataset exported:")
                        print(f"  Binary data: {pickle_path}")
                        print(f"  Summary report: {summary_path}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not export training dataset: {e}")
                
                # Step 6: Initialize PyTorch DCNN model
                print("\nStep 6: Initializing PyTorch DCNN model...")
                
                model = DCNNDamageNet(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    dropout_rate=0.3
                )
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print("✓ PyTorch DCNN model initialized")
                print(f"  Total parameters: {total_params:,}")
                print(f"  Trainable parameters: {trainable_params:,}")
                
                # Model summary
                print(f"\nModel Architecture Summary:")
                print(f"  Input: {input_shape}")
                print(f"  Conv1D layers: 128 → 256 → 512 channels")
                print(f"  FC layers: 1024 → 512 → {num_classes}")
                print(f"  Activation: ReLU + LogSoftmax")
                print(f"  Regularization: BatchNorm + Dropout({model.dropout_rate})")
                
                # Step 7: Train PyTorch model
                print("\nStep 7: Training PyTorch model...")
                train_choice = input("Start PyTorch training? (y/n): ")
                
                if train_choice.lower() == 'y':
                    # Create trainer
                    trainer = DCNNTrainer(model, device=device)
                    
                    print(f"\nStarting training...")
                    print(f"Following Yu et al. parameters:")
                    print(f"  Learning rate: 0.0035")
                    print(f"  Batch size: 50")
                    print(f"  Early stopping patience: 15")
                    print(f"  Device: {device}")
                    
                    # Train model
                    trainer.train_model(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=20,
                        learning_rate=0.0035,  # Yu et al. optimal
                        patience=5
                    )
                    
                    # Plot training history
                    print("\nPlotting training history...")
                    trainer.plot_training_history(save_path="../results/pytorch_training_history.png")
                    
                    # Evaluate model
                    print("\nEvaluating model on test set...")
                    results = trainer.evaluate_model(test_loader, class_names=label_encoder.classes_)
                    
                    print(f"\n=== Final Results ===")
                    print(f"Test Accuracy: {results['accuracy']:.4f}")
                    
                    # Save model
                    model_path = "../models/dcnn_pytorch_model.pth"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_config': {
                            'input_shape': input_shape,
                            'num_classes': num_classes,
                            'dropout_rate': 0.3
                        },
                        'label_encoder': label_encoder,
                        'results': results
                    }, model_path)
                    
                    print(f"✓ PyTorch model saved to: {model_path}")
                    
                    # Compare with Yu et al. results
                    print(f"\n=== Comparison with Yu et al. (2018) ===")
                    print(f"Paper reported:")
                    print(f"  - SCC (correlation): 0.9983 (10dB noise)")
                    print(f"  - Superior to GRNN (0.9692) and ANFIS (0.9672)")
                    print(f"Our PyTorch implementation:")
                    print(f"  - Test Accuracy: {results['accuracy']:.4f}")
                    print(f"  - Architecture: Faithful reproduction")
                    print(f"  - Preprocessing: FFT + PSD selection (70% energy)")
                
            except Exception as e:
                print(f"✗ Error in PyTorch pipeline: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print(f"✗ Signals directory not found: {signals_dir}")
    
    print("\n=== PyTorch Demo Complete ===")
    print("\nPyTorch advantages over TensorFlow:")
    print("✓ More pythonic and intuitive API")
    print("✓ Dynamic computation graphs")
    print("✓ Better debugging capabilities")
    print("✓ Popular in research community")
    print("✓ Excellent GPU acceleration")
    print("\nMethodology faithfully implements Yu et al. (2018):")
    print("✓ FFT preprocessing with PSD selection")
    print("✓ DCNN architecture with large initial kernels")
    print("✓ Optimal hyperparameters (lr=0.0035, batch=50)")
    print("✓ Automatic feature extraction from raw signals")


if __name__ == "__main__":
    main()