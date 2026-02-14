"""
Utilidad para logging sistemático de métricas de experimentos de clustering.

Este módulo permite registrar métricas de diferentes experimentos de clustering
en un archivo CSV para análisis comparativo posterior.

Fecha: 2026-01-19
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional


class ClusteringMetricsLogger:
    """
    Logger para registrar métricas de experimentos de clustering en CSV.

    Permite comparar diferentes algoritmos, transformaciones y configuraciones
    de forma sistemática para análisis posterior.

    Attributes:
        output_file: Path al archivo CSV de salida
        columns: Lista de columnas esperadas en el CSV

    Examples:
        >>> logger = ClusteringMetricsLogger('../../results/clustering_experiments.csv')
        >>> logger.log_experiment(
        ...     notebook_name="2_clustering_fft_kmeans",
        ...     algorithm="kmeans",
        ...     frequency_transform="fft",
        ...     n_clusters=3,
        ...     silhouette_score=0.32
        ... )
        📊 Experimento registrado: 2_clustering_fft_kmeans | kmeans | fft
    """

    def __init__(self, output_file: str):
        """
        Inicializa el logger.

        Args:
            output_file: Ruta al archivo CSV de salida
        """
        self.output_file = Path(output_file)

        # Crear directorio si no existe
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Columnas esperadas en el CSV
        self.columns = [
            'experiment_id',
            'timestamp',
            'notebook_name',
            'algorithm',
            'frequency_transform',
            'n_clusters',
            'pca_dims',
            'feature_set',
            'inertia',
            'silhouette_score',
            'davies_bouldin_index',
            'calinski_harabasz_score',
            'adjusted_rand_index',
            'normalized_mutual_info',
            'homogeneity',
            'completeness',
            'v_measure',
            'n_samples',
            'n_features',
            'notes'
        ]

    def log_experiment(
        self,
        experiment_id: str,
        notebook_name: str,
        algorithm: str,
        frequency_transform: str,
        n_clusters: Optional[int] = None,
        pca_dims: Optional[int] = None,
        feature_set: str = "unknown",
        inertia: Optional[float] = None,
        silhouette_score: Optional[float] = None,
        davies_bouldin_index: Optional[float] = None,
        calinski_harabasz_score: Optional[float] = None,
        adjusted_rand_index: Optional[float] = None,
        normalized_mutual_info: Optional[float] = None,
        homogeneity: Optional[float] = None,
        completeness: Optional[float] = None,
        v_measure: Optional[float] = None,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        notes: str = ""
    ) -> None:
        """
        Registra un experimento de clustering en el archivo CSV.

        Args:
            experiment_id: ID único del experimento (evita duplicados)
            notebook_name: Nombre del notebook
            algorithm: Algoritmo usado (kmeans, dbscan, agglomerative, hierarchical)
            frequency_transform: Transformación aplicada (fft, wavelet, cwt, none)
            n_clusters: Número de clusters (opcional, para kmeans/agglomerative)
            pca_dims: Dimensiones PCA (2, 3, o None)
            feature_set: Conjunto de features usado (simple_spectral, advanced_spectral, temporal)
            inertia: Inercia WCSS (solo para kmeans)
            silhouette_score: Silhouette Score [-1, 1]
            davies_bouldin_index: Davies-Bouldin Index [0, ∞)
            calinski_harabasz_score: Calinski-Harabasz Score [0, ∞)
            adjusted_rand_index: Adjusted Rand Index [-1, 1]
            normalized_mutual_info: Normalized Mutual Information [0, 1]
            homogeneity: Homogeneity Score [0, 1]
            completeness: Completeness Score [0, 1]
            v_measure: V-Measure Score [0, 1]
            n_samples: Número de muestras
            n_features: Número de características
            notes: Notas adicionales (opcional)

        Returns:
            None

        Examples:
            >>> logger.log_experiment(
            ...     notebook_name="2_clustering_fft_kmeans",
            ...     algorithm="kmeans",
            ...     frequency_transform="fft",
            ...     n_clusters=3,
            ...     pca_dims=2,
            ...     feature_set="simple_spectral",
            ...     inertia=277.67,
            ...     silhouette_score=0.3243,
            ...     notes="Baseline experiment"
            ... )
        """
        # Crear registro
        experiment = {
            'experiment_id': experiment_id,
            'timestamp': self._get_timestamp(),
            'notebook_name': notebook_name,
            'algorithm': algorithm,
            'frequency_transform': frequency_transform,
            'n_clusters': n_clusters,
            'pca_dims': pca_dims,
            'feature_set': feature_set,
            'inertia': inertia,
            'silhouette_score': silhouette_score,
            'davies_bouldin_index': davies_bouldin_index,
            'calinski_harabasz_score': calinski_harabasz_score,
            'adjusted_rand_index': adjusted_rand_index,
            'normalized_mutual_info': normalized_mutual_info,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'v_measure': v_measure,
            'n_samples': n_samples,
            'n_features': n_features,
            'notes': notes
        }

        # Convertir a DataFrame
        df_new = pd.DataFrame([experiment])

        # Verificar si ya existe el experimento (evitar duplicados)
        if self.output_file.exists():
            df_existing = pd.read_csv(self.output_file)
            if experiment_id in df_existing['experiment_id'].values:
                # Actualizar registro existente
                df_existing = df_existing[df_existing['experiment_id'] != experiment_id]
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(self.output_file, mode='w', header=True, index=False)
                print(f"🔄 Experimento actualizado: {notebook_name} | {algorithm} | {frequency_transform} (ID: {experiment_id})")
            else:
                # Agregar nuevo registro
                df_new.to_csv(self.output_file, mode='a', header=False, index=False)
                print(f"📊 Experimento registrado: {notebook_name} | {algorithm} | {frequency_transform} (ID: {experiment_id})")
        else:
            # Crear archivo nuevo
            df_new.to_csv(self.output_file, mode='w', header=True, index=False)
            print(f"📊 Experimento registrado: {notebook_name} | {algorithm} | {frequency_transform} (ID: {experiment_id})")

    def _get_timestamp(self) -> str:
        """
        Retorna timestamp actual en formato ISO.

        Returns:
            Timestamp como string en formato 'YYYY-MM-DD HH:MM:SS'
        """
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    # Código de prueba
    print("🧪 Probando ClusteringMetricsLogger...")

    # Crear logger de prueba
    logger = ClusteringMetricsLogger(output_file="../../results/test_experiments.csv")

    # Registrar experimento de prueba
    logger.log_experiment(
        experiment_id="test_001",
        notebook_name="test_notebook",
        algorithm="kmeans",
        frequency_transform="fft",
        n_clusters=3,
        pca_dims=2,
        feature_set="simple_spectral",
        inertia=277.67,
        silhouette_score=0.3243,
        notes="Test experiment"
    )

    print("✅ Prueba exitosa!")
