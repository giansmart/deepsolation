"""
Modelos de deep learning para clasificación de daño en aisladores sísmicos.

Este módulo contiene implementaciones de PyTorch para:
- Autoencoder no supervisado (ETAPA 1)
- CNN clasificador (ETAPA 2) - próximamente
"""

from .autoencoder import Autoencoder, Encoder, Decoder, create_autoencoder

__all__ = [
    'Autoencoder',
    'Encoder',
    'Decoder',
    'create_autoencoder'
]
