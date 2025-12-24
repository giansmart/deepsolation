"""
Preprocesamiento Metodológicamente Correcto - Experimento 4
========================================================

Implementa el enfoque recomendado por el experto para resolver los problemas
fundamentales de pseudo-replicación en los experimentos anteriores.

PROBLEMA EN EXP1-3:
- Bins de frecuencia tratados como observaciones independientes
- Miles de "observaciones" del mismo dispositivo físico
- Pseudo-replicación severa
- Unidad de observación ≠ Unidad de inferencia

SOLUCIÓN EN EXP4:
- Una observación = Un dispositivo completo
- Características estadísticas agregadas
- Una etiqueta por dispositivo físico
- Alineación correcta de unidades

METODOLOGÍA:
- Análisis temporal por ventanas deslizantes
- Estadísticos espectrales agregados
- Métricas de energía y frecuencia dominante
- Características robustas de distribución espectral
"""

import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class Exp4SignalPreprocessor:
    """
    Preprocesador metodológicamente correcto para Experimento 4
    
    Implementa el enfoque científicamente válido donde:
    - Una observación = Un dispositivo físico completo
    - Características agregadas por dispositivo
    - Sin pseudo-replicación
    """
    
    def __init__(self, sampling_rate: int = 100, window_size: int = 1000, 
                 overlap: float = 0.5, verbose: bool = False):
        """
        Inicializar preprocesador metodológicamente correcto
        
        Args:
            sampling_rate: Frecuencia de muestreo en Hz
            window_size: Tamaño de ventana temporal para análisis estadístico
            overlap: Overlap entre ventanas (0-1)
            verbose: Mostrar información detallada
        """
        self.fs = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.verbose = verbose
        
        # Validación de parámetros
        if not 0 <= overlap < 1:
            raise ValueError("Overlap debe estar entre 0 y 1")
        if window_size <= 0:
            raise ValueError("Window size debe ser positivo")
    
    def load_signal_file(self, file_path: str) -> Optional[np.ndarray]:
        """Cargar señal desde archivo txt"""
        try:
            df = pd.read_csv(file_path, delimiter=' ', skiprows=1)
            # Extraer componentes N-S, E-W, U-D (saltar columnas Fecha y Hora)
            signals = df.iloc[:, [2, 3, 4]].values
            return signals
        except Exception as e:
            if self.verbose:
                print(f"Error cargando {file_path}: {e}")
            return None
    
    def extract_windowed_statistics(self, signal_data: np.ndarray) -> Dict:
        """
        Extraer estadísticos temporales usando ventanas deslizantes
        
        ENFOQUE METODOLÓGICAMENTE CORRECTO:
        - Análisis por ventanas temporales
        - Estadísticos robustos por componente
        - Características globales del dispositivo
        
        Args:
            signal_data: Array (n_samples, 3) con componentes N-S, E-W, U-D
            
        Returns:
            Dict con estadísticos temporales agregados
        """
        n_samples = signal_data.shape[0]
        hop_size = int(self.window_size * (1 - self.overlap))
        
        # Calcular ventanas
        n_windows = (n_samples - self.window_size) // hop_size + 1
        
        if n_windows <= 0:
            # Si la señal es muy corta, usar toda la señal
            window_stats = self._calculate_window_statistics(signal_data)
            return self._aggregate_window_statistics([window_stats])
        
        # Calcular estadísticos por ventana
        window_statistics = []
        for i in range(n_windows):
            start_idx = i * hop_size
            end_idx = start_idx + self.window_size
            window_data = signal_data[start_idx:end_idx, :]
            
            window_stats = self._calculate_window_statistics(window_data)
            window_statistics.append(window_stats)
        
        # Agregar estadísticos a nivel de dispositivo
        device_stats = self._aggregate_window_statistics(window_statistics)
        
        return device_stats
    
    def _calculate_window_statistics(self, window_data: np.ndarray) -> Dict:
        """Calcular estadísticos para una ventana temporal"""
        stats_dict = {}
        
        components = ['NS', 'EW', 'UD']
        for i, comp in enumerate(components):
            data = window_data[:, i]
            
            # Estadísticos temporales básicos
            stats_dict[f'{comp}_mean'] = np.mean(data)
            stats_dict[f'{comp}_std'] = np.std(data)
            stats_dict[f'{comp}_var'] = np.var(data)
            stats_dict[f'{comp}_rms'] = np.sqrt(np.mean(data**2))
            stats_dict[f'{comp}_max'] = np.max(data)
            stats_dict[f'{comp}_min'] = np.min(data)
            stats_dict[f'{comp}_range'] = np.max(data) - np.min(data)
            
            # Estadísticos de forma
            stats_dict[f'{comp}_skewness'] = stats.skew(data)
            stats_dict[f'{comp}_kurtosis'] = stats.kurtosis(data)
            
            # Percentiles
            stats_dict[f'{comp}_p25'] = np.percentile(data, 25)
            stats_dict[f'{comp}_p50'] = np.percentile(data, 50)
            stats_dict[f'{comp}_p75'] = np.percentile(data, 75)
            
            # Medidas de energía
            stats_dict[f'{comp}_energy'] = np.sum(data**2)
            stats_dict[f'{comp}_zero_crossings'] = np.sum(np.diff(np.sign(data)) != 0)
        
        return stats_dict
    
    def _aggregate_window_statistics(self, window_stats_list: List[Dict]) -> Dict:
        """
        Agregar estadísticos de ventanas a nivel de dispositivo
        
        ENFOQUE CORRECTO: Características globales del dispositivo completo
        """
        if not window_stats_list:
            return {}
        
        # Convertir lista de diccionarios a DataFrame para facilitar agregación
        windows_df = pd.DataFrame(window_stats_list)
        
        device_stats = {}
        
        # Para cada métrica temporal, calcular estadísticos agregados
        for column in windows_df.columns:
            values = windows_df[column].values
            
            # Estadísticos de la distribución de la métrica a través de ventanas
            device_stats[f'{column}_global_mean'] = np.mean(values)
            device_stats[f'{column}_global_std'] = np.std(values)
            device_stats[f'{column}_global_min'] = np.min(values)
            device_stats[f'{column}_global_max'] = np.max(values)
            device_stats[f'{column}_global_range'] = np.max(values) - np.min(values)
            device_stats[f'{column}_global_median'] = np.median(values)
        
        # Agregar información de ventanas procesadas
        device_stats['n_windows'] = len(window_stats_list)
        device_stats['total_signal_length'] = len(window_stats_list) * self.window_size
        
        return device_stats
    
    def extract_spectral_statistics(self, signal_data: np.ndarray) -> Dict:
        """
        Extraer características espectrales agregadas del dispositivo
        
        ENFOQUE METODOLÓGICAMENTE CORRECTO:
        - Características espectrales globales
        - Sin bins individuales como observaciones
        - Estadísticos robustos del espectro completo
        
        Args:
            signal_data: Array (n_samples, 3) con componentes N-S, E-W, U-D
            
        Returns:
            Dict con características espectrales agregadas
        """
        n_samples = signal_data.shape[0]
        
        spectral_stats = {}
        components = ['NS', 'EW', 'UD']
        
        for i, comp in enumerate(components):
            data = signal_data[:, i]
            
            # FFT y espectro de potencia
            fft_data = fft(data)
            freqs = fftfreq(n_samples, 1/self.fs)
            
            # Usar solo frecuencias positivas
            positive_mask = freqs >= 0
            fft_data = fft_data[positive_mask]
            freqs = freqs[positive_mask]
            
            # Espectro de potencia
            power_spectrum = np.abs(fft_data)**2
            
            # Características espectrales agregadas
            spectral_stats[f'{comp}_spectral_centroid'] = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
            spectral_stats[f'{comp}_spectral_spread'] = np.sqrt(np.sum((freqs - spectral_stats[f'{comp}_spectral_centroid'])**2 * power_spectrum) / np.sum(power_spectrum))
            spectral_stats[f'{comp}_spectral_rolloff'] = self._spectral_rolloff(freqs, power_spectrum, 0.95)
            spectral_stats[f'{comp}_spectral_flux'] = np.sum(np.diff(power_spectrum)**2)
            
            # Energía en bandas de frecuencia
            spectral_stats[f'{comp}_energy_low'] = np.sum(power_spectrum[freqs <= 1.0])  # 0-1 Hz
            spectral_stats[f'{comp}_energy_mid'] = np.sum(power_spectrum[(freqs > 1.0) & (freqs <= 10.0)])  # 1-10 Hz
            spectral_stats[f'{comp}_energy_high'] = np.sum(power_spectrum[freqs > 10.0])  # >10 Hz
            
            # Frecuencia dominante
            dominant_freq_idx = np.argmax(power_spectrum)
            spectral_stats[f'{comp}_dominant_freq'] = freqs[dominant_freq_idx]
            spectral_stats[f'{comp}_dominant_power'] = power_spectrum[dominant_freq_idx]
            
            # Estadísticos del espectro de potencia
            spectral_stats[f'{comp}_spectral_mean'] = np.mean(power_spectrum)
            spectral_stats[f'{comp}_spectral_std'] = np.std(power_spectrum)
            spectral_stats[f'{comp}_spectral_skew'] = stats.skew(power_spectrum)
            spectral_stats[f'{comp}_spectral_kurt'] = stats.kurtosis(power_spectrum)
            
            # Entropía espectral (medida de concentración de energía)
            normalized_power = power_spectrum / np.sum(power_spectrum)
            normalized_power = normalized_power[normalized_power > 0]  # Evitar log(0)
            spectral_stats[f'{comp}_spectral_entropy'] = -np.sum(normalized_power * np.log2(normalized_power))
        
        return spectral_stats
    
    def _spectral_rolloff(self, freqs: np.ndarray, power_spectrum: np.ndarray, percentile: float = 0.95) -> float:
        """Calcular frecuencia de rolloff espectral"""
        cumulative_power = np.cumsum(power_spectrum)
        total_power = cumulative_power[-1]
        rolloff_idx = np.argmax(cumulative_power >= percentile * total_power)
        return freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]
    
    def preprocess_device_signal(self, file_path: str) -> Optional[Dict]:
        """
        Preprocesar señal completa de un dispositivo
        
        ENFOQUE METODOLÓGICAMENTE CORRECTO:
        - Procesa el dispositivo como una unidad completa
        - Extrae características agregadas
        - Una observación por dispositivo físico
        
        Args:
            file_path: Ruta al archivo de señal
            
        Returns:
            Dict con características agregadas del dispositivo
        """
        # Cargar señal
        signal_data = self.load_signal_file(file_path)
        if signal_data is None:
            return None
        
        # Extraer características temporales agregadas
        temporal_stats = self.extract_windowed_statistics(signal_data)
        
        # Extraer características espectrales agregadas
        spectral_stats = self.extract_spectral_statistics(signal_data)
        
        # Combinar todas las características
        device_features = {
            **temporal_stats,
            **spectral_stats
        }
        
        # Agregar metadata del procesamiento
        device_features.update({
            'file_path': file_path,
            'signal_length': signal_data.shape[0],
            'duration_seconds': signal_data.shape[0] / self.fs,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'sampling_rate': self.fs,
            'n_components': signal_data.shape[1]
        })
        
        return device_features
    
    def process_all_signals_correct(self, signals_dir: str, output_dir: str = None) -> Dict:
        """
        Procesar todas las señales con enfoque metodológicamente correcto
        
        ENFOQUE CORRECTO: Una observación por dispositivo físico
        
        Args:
            signals_dir: Directorio raíz con carpetas de señales (A1, A2, etc.)
            output_dir: Directorio para guardar features procesadas
            
        Returns:
            Dict con características procesadas por dispositivo
        """
        signals_dir = Path(signals_dir)
        processed_devices = {}
        
        # Encontrar todos los archivos de señal
        signal_files = list(signals_dir.glob('*/completo_S*.txt'))
        
        if self.verbose:
            print(f"Encontrados {len(signal_files)} archivos de señal")
        
        for file_path in signal_files:
            # Extraer información del dispositivo
            specimen = file_path.parent.name  # A1, A2, etc.
            sensor = file_path.stem.split('_')[-1]  # S1 o S2
            device_id = f"{specimen}_{sensor}"
            
            if self.verbose:
                print(f"Procesando dispositivo {device_id}...")
            
            # Procesar dispositivo completo
            device_features = self.preprocess_device_signal(file_path)
            
            if device_features is not None:
                processed_devices[device_id] = device_features
                
                if self.verbose:
                    print(f"  ✓ {device_id}: {len(device_features)} características extraídas")
            else:
                if self.verbose:
                    print(f"  ✗ Error procesando {device_id}")
        
        if self.verbose:
            print(f"\n✓ Procesamiento correcto completado: {len(processed_devices)} dispositivos")
        
        return processed_devices
    
    def export_correct_dataset(self, processed_devices: Dict, labels_df: pd.DataFrame, 
                             output_path: str) -> Dict:
        """
        Exportar dataset metodológicamente correcto
        
        ENFOQUE CORRECTO:
        - Una fila por dispositivo físico
        - Características agregadas como columnas
        - Una etiqueta por dispositivo
        
        Args:
            processed_devices: Dict con características procesadas
            labels_df: DataFrame con etiquetas de daño
            output_path: Ruta para el archivo CSV de salida
            
        Returns:
            Dict con resumen del dataset exportado
        """
        if self.verbose:
            print("Exportando dataset metodológicamente correcto...")
            print("🎯 ENFOQUE CORRECTO: Una observación = Un dispositivo físico")
        
        dataset_rows = []
        
        for device_id, features in processed_devices.items():
            # Extraer información del dispositivo
            specimen, sensor = device_id.split('_')
            
            # Obtener etiqueta de daño para este specimen
            damage_level = 'N1'  # Default
            if labels_df is not None:
                matching_rows = labels_df[labels_df['ID'] == specimen]
                if not matching_rows.empty:
                    damage_level = matching_rows['Ndano'].iloc[0]
            
            # Crear fila con información del dispositivo + características + etiqueta
            row = {
                'device_id': device_id,
                'specimen': specimen,
                'sensor': sensor,
                'damage_level': damage_level,  # UNA ETIQUETA POR DISPOSITIVO
                **features  # Todas las características agregadas
            }
            
            dataset_rows.append(row)
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(dataset_rows)
        df.to_csv(output_path, index=False)
        
        # Generar resumen
        summary = {
            'total_devices': len(df),
            'unique_specimens': df['specimen'].nunique(),
            'damage_distribution': dict(df['damage_level'].value_counts()),
            'feature_count': len([col for col in df.columns if col not in ['device_id', 'specimen', 'sensor', 'damage_level']]),
            'total_columns': len(df.columns)
        }
        
        if self.verbose:
            print(f"✓ Dataset metodológicamente correcto exportado: {output_path}")
            print(f"📊 ESTADÍSTICAS CORRECTAS:")
            print(f"  Dispositivos totales: {summary['total_devices']}")
            print(f"  Specimens únicos: {summary['unique_specimens']}")
            print(f"  Distribución de daño: {summary['damage_distribution']}")
            print(f"  Características por dispositivo: {summary['feature_count']}")
            print(f"  Columnas totales: {summary['total_columns']}")
            print()
            print("🎯 ENFOQUE METODOLÓGICAMENTE CORRECTO IMPLEMENTADO:")
            print("   ✅ Una observación = Un dispositivo físico")
            print("   ✅ Una etiqueta por dispositivo")
            print("   ✅ Características agregadas")
            print("   ✅ Sin pseudo-replicación")
        
        return summary


if __name__ == "__main__":
    # Ejemplo de uso del preprocesador metodológicamente correcto
    preprocessor = Exp4SignalPreprocessor(
        sampling_rate=100, 
        window_size=1000, 
        overlap=0.5, 
        verbose=True
    )
    
    # Procesar archivo de prueba
    test_file = "data/Signals_Raw/A1/completo_S1.txt"
    if os.path.exists(test_file):
        device_features = preprocessor.preprocess_device_signal(test_file)
        print(f"✓ Características extraídas para dispositivo A1_S1: {len(device_features)}")
        print("Primeras 10 características:")
        for i, (key, value) in enumerate(list(device_features.items())[:10]):
            print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")