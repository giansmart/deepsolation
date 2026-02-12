"""
Módulo de entrenamiento para modelos de deep learning.

Contiene scripts y utilidades para entrenar:
- Autoencoder (ETAPA 1) - aprendizaje no supervisado
- CNN clasificador (ETAPA 2) - aprendizaje supervisado (próximamente)
"""

from .train_autoencoder import train_autoencoder, EarlyStopping

__all__ = [
    'train_autoencoder',
    'EarlyStopping'
]
