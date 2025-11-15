"""
Signal Preprocessing Pipeline based on Yu et al. (2018)
Deep Learning approach for structural damage identification
"""

import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

class SignalPreprocessor:
    def __init__(self, sampling_rate=100, energy_threshold=0.7):
        """
        Initialize signal preprocessor
        
        Args:
            sampling_rate: Sampling frequency in Hz
            energy_threshold: Energy threshold for PSD selection (0.7 = 70%)
        """
        self.fs = sampling_rate
        self.energy_threshold = energy_threshold
        self.selected_freqs = None
        
    def load_signal_file(self, file_path):
        """Load signal from txt file"""
        try:
            df = pd.read_csv(file_path, delimiter=' ', skiprows=1)
            # Extract N-S, E-W, U-D components (skip Fecha and Hora columns)
            signals = df.iloc[:, [2, 3, 4]].values  # Skip Fecha and Hora columns
            return signals
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def apply_fft(self, signal_data):
        """
        Apply FFT transformation following Yu et al. methodology
        
        Args:
            signal_data: (n_samples, n_components) array
            
        Returns:
            fft_result: Complex FFT coefficients
            freqs: Frequency bins
            power_spectrum: Power spectral density
        """
        n_samples = signal_data.shape[0]
        
        # Apply FFT to each component
        fft_result = fft(signal_data, axis=0)
        freqs = fftfreq(n_samples, 1/self.fs)
        
        # Calculate power spectrum (PSD)
        power_spectrum = np.abs(fft_result)**2
        
        # Use only positive frequencies
        positive_freq_mask = freqs >= 0
        fft_result = fft_result[positive_freq_mask]
        freqs = freqs[positive_freq_mask]
        power_spectrum = power_spectrum[positive_freq_mask]
        
        return fft_result, freqs, power_spectrum
    
    def select_significant_components(self, power_spectrum):
        """
        Select frequency components that contain >threshold% of total energy
        Following Yu et al. methodology: first nm components with >70% energy
        
        Args:
            power_spectrum: Power spectral density array (n_freqs, n_components)
            
        Returns:
            selected_indices: Indices of selected frequency components
        """
        # Calculate total energy for each component
        total_energy = np.sum(power_spectrum, axis=0)
        
        # Sort frequencies by power (descending) for each component
        selected_indices_per_component = []
        
        for comp_idx in range(power_spectrum.shape[1]):
            # Sort by power for this component
            sorted_indices = np.argsort(power_spectrum[:, comp_idx])[::-1]
            
            # Select components until threshold energy is reached
            cumulative_energy = 0
            selected_indices = []
            
            for idx in sorted_indices:
                selected_indices.append(idx)
                cumulative_energy += power_spectrum[idx, comp_idx]
                
                if cumulative_energy >= self.energy_threshold * total_energy[comp_idx]:
                    break
                    
            selected_indices_per_component.append(selected_indices)
        
        # Take union of selected indices across all components
        all_selected = set()
        for indices in selected_indices_per_component:
            all_selected.update(indices)
            
        return sorted(list(all_selected))
    
    def build_feature_matrix(self, fft_data, selected_indices):
        """
        Build 2D feature matrix for DCNN input
        
        Args:
            fft_data: Complex FFT coefficients
            selected_indices: Selected frequency component indices
            
        Returns:
            feature_matrix: (n_selected_freqs, n_components) real-valued matrix
        """
        # Extract selected frequency components
        selected_fft = fft_data[selected_indices, :]
        
        # Use magnitude of complex coefficients
        feature_matrix = np.abs(selected_fft)
        
        return feature_matrix
    
    def preprocess_signal_file(self, file_path):
        """
        Complete preprocessing pipeline for a single signal file
        
        Args:
            file_path: Path to signal txt file
            
        Returns:
            feature_matrix: Preprocessed feature matrix for DCNN
            metadata: Dictionary with processing information
        """
        # Load signal
        signal_data = self.load_signal_file(file_path)
        if signal_data is None:
            return None, None
            
        # Apply FFT
        fft_result, freqs, power_spectrum = self.apply_fft(signal_data)
        
        # Select significant components
        selected_indices = self.select_significant_components(power_spectrum)
        self.selected_freqs = freqs[selected_indices]
        
        # Build feature matrix
        feature_matrix = self.build_feature_matrix(fft_result, selected_indices)
        
        # Calculate additional metadata for detailed reporting
        total_energy = np.sum(power_spectrum)
        selected_energy = np.sum(power_spectrum[selected_indices])
        energy_retained = selected_energy / total_energy
        
        metadata = {
            'file_path': file_path,
            'original_samples': signal_data.shape[0],
            'selected_frequencies': freqs[selected_indices],
            'compression_ratio': len(selected_indices) / len(freqs),
            'energy_retained': energy_retained,
            'energy_threshold': self.energy_threshold,
            'sampling_rate': self.fs,
            'duration_seconds': signal_data.shape[0] / self.fs,
            'fft_components': len(freqs),
            'freq_resolution': freqs[1] - freqs[0] if len(freqs) > 1 else 0,
            'nyquist_freq': self.fs / 2,
            'num_axes': signal_data.shape[1],
            'frequency_range': [freqs[selected_indices].min(), freqs[selected_indices].max()] if len(selected_indices) > 0 else [0, 0],
            'feature_matrix_shape': feature_matrix.shape
        }
        
        return feature_matrix, metadata
    
    def process_all_signals(self, signals_dir, output_dir=None):
        """
        Process all signal files in directory structure
        
        Args:
            signals_dir: Root directory containing signal folders (A1, A2, etc.)
            output_dir: Directory to save processed features
            
        Returns:
            processed_data: Dictionary with processed features and metadata
        """
        signals_dir = Path(signals_dir)
        processed_data = {}
        
        # Find all signal files
        signal_files = list(signals_dir.glob('*/completo_S*.txt'))
        
        print(f"Found {len(signal_files)} signal files")
        
        for file_path in signal_files:
            # Extract specimen and sensor info from path
            specimen = file_path.parent.name
            sensor = file_path.stem.split('_')[-1]  # S1 or S2
            
            print(f"Processing {specimen}/{sensor}...")
            
            # Process signal
            feature_matrix, metadata = self.preprocess_signal_file(file_path)
            
            if feature_matrix is not None:
                if specimen not in processed_data:
                    processed_data[specimen] = {}
                    
                processed_data[specimen][sensor] = {
                    'features': feature_matrix,
                    'metadata': metadata
                }
                
                print(f"  ✓ Shape: {feature_matrix.shape}, "
                      f"Compression: {metadata['compression_ratio']:.3f}")
            else:
                print(f"  ✗ Failed to process {file_path}")
                
        return processed_data

    def export_to_csv(self, processed_data, labels_df, output_path):
        """
        Export processed data to CSV format following Yu et al. (2018) methodology
        
        Exp2 approach: Una fila por (specimen, sensor) completo,
        representando la matriz completa de frecuencias.
        
        Args:
            processed_data: Dictionary from process_all_signals()
            labels_df: DataFrame with damage labels
            output_path: Path for output CSV file
        """
        import numpy as np
        print("Exporting processed dataset to CSV (Yu et al. 2018 methodology)...")
        print("📋 Exp2 approach: Una muestra = matriz completa por (specimen, sensor)")
        
        # Prepare data for CSV export
        dataset_rows = []
        matrix_shapes = []
        
        # Find the maximum frequency components for consistent padding
        max_freqs = 0
        for specimen, sensors in processed_data.items():
            for sensor, data in sensors.items():
                features = data['features']
                max_freqs = max(max_freqs, features.shape[0])
        
        print(f"📊 Maximum frequency components: {max_freqs}")
        print(f"📊 Padding all matrices to shape: ({max_freqs}, 3)")
        
        for specimen, sensors in processed_data.items():
            # Get damage level for this specimen
            damage_level = 'N1'  # Default
            if labels_df is not None:
                matching_rows = labels_df[labels_df['ID'] == specimen]
                if not matching_rows.empty:
                    damage_level = matching_rows['Ndano'].iloc[0]
            
            # Process each sensor (S1, S2) - UNA FILA POR SENSOR
            for sensor, data in sensors.items():
                features = data['features']  # Shape: (n_freqs, 3 components)
                metadata = data['metadata']
                
                # PAD matrix to consistent shape for all specimens
                padded_features = np.zeros((max_freqs, 3))
                padded_features[:features.shape[0], :] = features
                
                # Serialize the complete frequency matrix
                # Flatten to 1D array: [freq0_NS, freq0_EW, freq0_UD, freq1_NS, ...]
                flattened_matrix = padded_features.flatten()
                
                # Create column names for the flattened matrix
                matrix_cols = {}
                for freq_idx in range(max_freqs):
                    matrix_cols[f'freq_{freq_idx:04d}_NS'] = flattened_matrix[freq_idx * 3 + 0]
                    matrix_cols[f'freq_{freq_idx:04d}_EW'] = flattened_matrix[freq_idx * 3 + 1] 
                    matrix_cols[f'freq_{freq_idx:04d}_UD'] = flattened_matrix[freq_idx * 3 + 2]
                
                # Create row with specimen info + complete matrix + label
                row = {
                    'specimen': specimen,
                    'sensor': sensor,
                    'original_freq_components': features.shape[0],
                    'padded_freq_components': max_freqs,
                    'compression_ratio': metadata['compression_ratio'],
                    'energy_threshold': metadata['energy_threshold'],
                    'sampling_rate': metadata['sampling_rate'],
                    'original_samples': metadata['original_samples'],
                    'freq_resolution': metadata['freq_resolution'],
                    'nyquist_freq': metadata['nyquist_freq'],
                    **matrix_cols,  # Add all matrix columns
                    'damage_level': damage_level  # Target variable at the end
                }
                
                dataset_rows.append(row)
                matrix_shapes.append(features.shape)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(dataset_rows)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset exported to: {output_path}")
        print(f"📊 NUEVO FORMATO (Yu et al. 2018):")
        print(f"  Total muestras: {len(df)} (una por specimen-sensor)")
        print(f"  Specimens únicos: {df['specimen'].nunique()}")
        print(f"  Sensores por specimen: {df.groupby('specimen')['sensor'].nunique().iloc[0] if len(df) > 0 else 0}")
        print(f"  Distribución de daño: {dict(df['damage_level'].value_counts())}")
        print(f"  Columnas totales: {len(df.columns)} (metadata + matrix + label)")
        print(f"  Forma de matrices: {matrix_shapes[0] if matrix_shapes else 'N/A'} → padded to ({max_freqs}, 3)")
        print()
        print("🎯 Exp2: Una etiqueta por matriz completa (approach por specimen-sensor)")
        
        return df

    def create_training_dataset_csv(self, signals_dir, labels_csv_path, output_csv_path):
        """
        Complete pipeline: process signals and export training dataset as CSV
        
        Args:
            signals_dir: Directory with raw signal files
            labels_csv_path: Path to labels CSV file  
            output_csv_path: Path for output training dataset CSV
        """
        print("CREATING TRAINING DATASET (CSV FORMAT)")
        print("="*50)
        
        # Load labels
        labels_df = None
        if labels_csv_path and Path(labels_csv_path).exists():
            labels_df = pd.read_csv(labels_csv_path)
            print(f"✓ Labels loaded from: {labels_csv_path}")
        else:
            print("⚠ No labels file found")
            raise ValueError("No labels file found")
        
        # Process all signals
        print("\nProcessing signals...")
        processed_data = self.process_all_signals(signals_dir)
        
        # Export to CSV
        print("\nExporting to CSV...")
        dataset_df = self.export_to_csv(processed_data, labels_df, output_csv_path)
        
        # Generate summary report
        summary_path = str(output_csv_path).replace('.csv', '_summary.txt')
        self._generate_summary_report(processed_data, dataset_df, summary_path)
        
        print(f"\n✓ Training dataset created: {output_csv_path}")
        print(f"✓ Summary report: {summary_path}")
        
        return dataset_df

    def _generate_summary_report(self, processed_data, dataset_df, summary_path):
        """Generate detailed summary report"""
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING DATASET EXPORT SUMMARY (CSV FORMAT)\n") 
            f.write("Generated for DCNN Training Analysis\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Export Date: {pd.Timestamp.now()}\n")
            f.write("Methodology: Yu et al. (2018) - FFT + PSD selection\n\n")
            
            f.write("="*50 + "\n")
            f.write("DATASET COMPOSITION\n")
            f.write("="*50 + "\n")
            f.write(f"Total specimens: {dataset_df['specimen'].nunique()}\n")
            f.write(f"Total observations: {len(dataset_df)}\n")
            f.write(f"Sensors per specimen: {dataset_df.groupby('specimen')['sensor'].nunique().iloc[0]}\n")
            f.write("Damage level distribution:\n")
            for level, count in dataset_df['damage_level'].value_counts().items():
                pct = count / len(dataset_df) * 100
                f.write(f"  {level}: {count} observations ({pct:.1f}%)\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("PREPROCESSING STATISTICS\n") 
            f.write("="*50 + "\n")
            f.write(f"Sampling rate: {dataset_df['sampling_rate'].iloc[0]} Hz\n")
            f.write(f"Energy threshold: {dataset_df['energy_threshold'].iloc[0]*100}%\n")
            f.write(f"Average compression ratio: {dataset_df['compression_ratio'].mean():.3f}\n")
            f.write(f"Compression range: {dataset_df['compression_ratio'].min():.3f} - {dataset_df['compression_ratio'].max():.3f}\n")
            f.write(f"Frequency resolution: {dataset_df['freq_resolution'].iloc[0]:.3f} Hz\n")
            f.write(f"Nyquist frequency: {dataset_df['nyquist_freq'].iloc[0]:.1f} Hz\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("CSV STRUCTURE\n")
            f.write("="*50 + "\n")
            f.write("Columns:\n")
            for col in dataset_df.columns:
                f.write(f"  - {col}\n")
            
            f.write(f"\nData format: Each row represents one frequency component\n")
            f.write(f"Features per row: 3 (N-S, E-W, U-D components)\n")
            f.write(f"Frequency range: {dataset_df['frequency_hz'].min():.2f} - {dataset_df['frequency_hz'].max():.2f} Hz\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("SPECIMEN DETAILS\n")
            f.write("="*50 + "\n")
            
            for specimen in sorted(dataset_df['specimen'].unique()):
                specimen_data = dataset_df[dataset_df['specimen'] == specimen]
                damage_level = specimen_data['damage_level'].iloc[0]
                sensors = specimen_data['sensor'].unique()
                
                f.write(f"{specimen:<8} | {damage_level} | ")
                for sensor in sorted(sensors):
                    sensor_data = specimen_data[specimen_data['sensor'] == sensor]
                    freq_count = len(sensor_data)
                    comp_ratio = sensor_data['compression_ratio'].iloc[0]
                    f.write(f"{sensor}: {freq_count} freqs ({comp_ratio:.3f}) | ")
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("="*80 + "\n")
    
    def visualize_preprocessing(self, file_path, save_plot=False):
        """
        Visualize the preprocessing steps for a single file
        """
        # Load and process signal
        signal_data = self.load_signal_file(file_path)
        fft_result, freqs, power_spectrum = self.apply_fft(signal_data)
        selected_indices = self.select_significant_components(power_spectrum)
        feature_matrix = self.build_feature_matrix(fft_result, selected_indices)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Signal Preprocessing Pipeline - {Path(file_path).name}', fontsize=14)
        
        # Row 1: Original signals
        for i, component in enumerate(['N-S', 'E-W', 'U-D']):
            axes[0, i].plot(signal_data[:1000, i])  # Show first 1000 points
            axes[0, i].set_title(f'Time Domain - {component}')
            axes[0, i].set_xlabel('Sample')
            axes[0, i].set_ylabel('Amplitude')
            
        # Row 2: Power spectra and selected components
        for i, component in enumerate(['N-S', 'E-W', 'U-D']):
            axes[1, i].loglog(freqs, power_spectrum[:, i], alpha=0.7, label='All')
            axes[1, i].loglog(freqs[selected_indices], power_spectrum[selected_indices, i], 
                            'ro', markersize=2, label='Selected')
            axes[1, i].set_title(f'Frequency Domain - {component}')
            axes[1, i].set_xlabel('Frequency (Hz)')
            axes[1, i].set_ylabel('Power')
            axes[1, i].legend()
            axes[1, i].grid(True)
            
        plt.tight_layout()
        
        if save_plot:
            output_path = Path(file_path).parent / f'preprocessing_visualization.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
            
        plt.show()
        
        # Print summary statistics
        print(f"\nPreprocessing Summary:")
        print(f"Original signal length: {signal_data.shape[0]} samples")
        print(f"Selected frequencies: {len(selected_indices)} / {len(freqs)}")
        print(f"Compression ratio: {len(selected_indices)/len(freqs):.3f}")
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix


if __name__ == "__main__":
    # Example usage
    preprocessor = SignalPreprocessor(sampling_rate=100, energy_threshold=0.7)
    
    # Process single file for testing
    test_file = "data/Signals_Raw/A1/completo_S1.txt"
    if os.path.exists(test_file):
        feature_matrix, metadata = preprocessor.preprocess_signal_file(test_file)
        print(f"Test processing complete: {metadata}")
        
        # Visualize preprocessing
        preprocessor.visualize_preprocessing(test_file, save_plot=True)