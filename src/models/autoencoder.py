"""
Autoencoder para aprendizaje no supervisado de señales sísmicas (ETAPA 1).

Arquitectura de 4 capas Conv1D para aprender representaciones de 512 dimensiones
a partir de señales S2-S1 concatenadas (6 canales × 60,000 muestras).

Autor: Giancarlo Poémape Lozano
Fecha: 2026-02-07
"""

from typing import Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder del autoencoder: reduce señales (6, 60000) → latent vector (512,).

    Arquitectura:
        - Layer 1: Conv1D(6→64, k=11, s=2) + BN + ReLU + MaxPool(2)
        - Layer 2: Conv1D(64→128, k=7) + BN + ReLU + MaxPool(2)
        - Layer 3: Conv1D(128→256, k=5) + BN + ReLU + MaxPool(2)
        - Layer 4: Conv1D(256→512, k=3) + BN + ReLU + GlobalAvgPool

    Args:
        in_channels: Número de canales de entrada (default: 6)
        latent_dim: Dimensión del espacio latente (default: 512)
    """

    def __init__(self, in_channels: int = 6, latent_dim: int = 512):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Bloque 1: (6, 60000) → (64, 14999) tras MaxPool
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=11,
            stride=2,
            padding=5
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bloque 2: (64, 14999) → (128, 3749) tras MaxPool
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bloque 3: (128, 3749) → (256, 936) tras MaxPool
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bloque 4: (256, 936) → (512, 936) → (512,) tras GlobalAvgPool
        self.conv4 = nn.Conv1d(
            in_channels=256,
            out_channels=latent_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn4 = nn.BatchNorm1d(latent_dim)

        # Global Average Pooling: (512, N) → (512,)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Activación
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del encoder.

        Args:
            x: Tensor de entrada con shape (batch, 6, 60000)

        Returns:
            Latent vector con shape (batch, 512)
        """
        # Bloque 1
        x = self.conv1(x)      # (batch, 64, 30000)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)      # (batch, 64, 15000)

        # Bloque 2
        x = self.conv2(x)      # (batch, 128, 15000)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)      # (batch, 128, 7500)

        # Bloque 3
        x = self.conv3(x)      # (batch, 256, 7500)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)      # (batch, 256, 3750)

        # Bloque 4
        x = self.conv4(x)      # (batch, 512, 3750)
        x = self.bn4(x)
        x = self.relu(x)

        # Global Average Pooling
        x = self.global_pool(x)  # (batch, 512, 1)
        x = x.squeeze(-1)        # (batch, 512)

        return x


class Decoder(nn.Module):
    """
    Decoder del autoencoder: reconstruye señales desde latent vector (512,) → (6, 60000).

    Arquitectura:
        - Layer 1: Linear(512→512×256) + Reshape + ConvTranspose1D(512→256) + BN + ReLU
        - Layer 2: Upsample + ConvTranspose1D(256→128) + BN + ReLU
        - Layer 3: Upsample + ConvTranspose1D(128→64) + BN + ReLU
        - Layer 4: Upsample + Conv1D(64→6, k=11) → Reconstrucción final

    Args:
        latent_dim: Dimensión del espacio latente (default: 512)
        out_channels: Número de canales de salida (default: 6)
    """

    def __init__(self, latent_dim: int = 512, out_channels: int = 6):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.out_channels = out_channels

        # Capa inicial: expandir latent vector a feature map inicial
        # (512,) → (512, 3750) para empezar el upsampling
        self.fc = nn.Linear(latent_dim, latent_dim * 15)  # 512 * 15 = 7680
        self.initial_length = 15

        # Bloque 1: (512, 15) → (256, 60) con upsample ×4
        self.upsample1 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=latent_dim,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(256)

        # Bloque 2: (256, 60) → (128, 3750) con upsample ~×62
        self.upsample2 = nn.Upsample(scale_factor=62.5, mode='linear', align_corners=False)
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=256,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.bn2 = nn.BatchNorm1d(128)

        # Bloque 3: (128, 3750) → (64, 15000) con upsample ×4
        self.upsample3 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.deconv3 = nn.ConvTranspose1d(
            in_channels=128,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.bn3 = nn.BatchNorm1d(64)

        # Bloque 4 (final): (64, 15000) → (6, 60000) con upsample ×4
        self.upsample4 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.final_conv = nn.Conv1d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=11,
            stride=1,
            padding=5
        )

        # Activación
        self.relu = nn.ReLU(inplace=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del decoder.

        Args:
            z: Latent vector con shape (batch, 512)

        Returns:
            Reconstrucción con shape (batch, 6, 60000)
        """
        batch_size = z.shape[0]

        # Expandir latent vector: (batch, 512) → (batch, 512, 15)
        x = self.fc(z)                           # (batch, 7680)
        x = x.view(batch_size, self.latent_dim, self.initial_length)  # (batch, 512, 15)

        # Bloque 1
        x = self.upsample1(x)   # (batch, 512, 60)
        x = self.deconv1(x)     # (batch, 256, 60)
        x = self.bn1(x)
        x = self.relu(x)

        # Bloque 2
        x = self.upsample2(x)   # (batch, 256, 3750)
        x = self.deconv2(x)     # (batch, 128, 3750)
        x = self.bn2(x)
        x = self.relu(x)

        # Bloque 3
        x = self.upsample3(x)   # (batch, 128, 15000)
        x = self.deconv3(x)     # (batch, 64, 15000)
        x = self.bn3(x)
        x = self.relu(x)

        # Bloque 4 (final)
        x = self.upsample4(x)   # (batch, 64, 60000)
        x = self.final_conv(x)  # (batch, 6, 60000)

        return x


class Autoencoder(nn.Module):
    """
    Autoencoder completo para aprendizaje no supervisado de señales sísmicas.

    Combina Encoder y Decoder para reconstrucción (6, 60000) → (512,) → (6, 60000).
    El encoder pre-entrenado se puede extraer para la Etapa 2 (clasificación supervisada).

    Args:
        in_channels: Número de canales de entrada (default: 6)
        latent_dim: Dimensión del espacio latente (default: 512)

    Example:
        >>> autoencoder = Autoencoder(in_channels=6, latent_dim=512)
        >>> x = torch.randn(16, 6, 60000)  # Batch de 16 señales
        >>> reconstruction = autoencoder(x)  # (16, 6, 60000)
        >>>
        >>> # Extraer encoder para clasificación (ETAPA 2)
        >>> encoder = autoencoder.get_encoder()
        >>> features = encoder(x)  # (16, 512)
    """

    def __init__(self, in_channels: int = 6, latent_dim: int = 512):
        super(Autoencoder, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Componentes
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encoder + decoder.

        Args:
            x: Tensor de entrada con shape (batch, 6, 60000)

        Returns:
            Tuple (reconstruction, latent):
                - reconstruction: Tensor (batch, 6, 60000)
                - latent: Tensor (batch, 512)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def get_encoder(self) -> Encoder:
        """
        Extrae el encoder pre-entrenado para usar en ETAPA 2.

        Returns:
            Encoder con pesos entrenados

        Example:
            >>> autoencoder = Autoencoder()
            >>> # ... entrenar autoencoder ...
            >>> encoder = autoencoder.get_encoder()
            >>> # Usar encoder como feature extractor en CNN clasificador
        """
        return self.encoder

    def count_parameters(self) -> int:
        """
        Cuenta parámetros entrenables del modelo.

        Returns:
            Número total de parámetros entrenables
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Función auxiliar para instanciar modelo
def create_autoencoder(
    in_channels: int = 6,
    latent_dim: int = 512,
    device: str = 'cpu'
) -> Autoencoder:
    """
    Factory function para crear autoencoder con configuración estándar.

    Args:
        in_channels: Número de canales de entrada
        latent_dim: Dimensión del espacio latente
        device: Dispositivo PyTorch ('cpu', 'cuda', 'mps')

    Returns:
        Autoencoder inicializado y movido al device especificado

    Example:
        >>> # Mac M2 con MPS
        >>> device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        >>> autoencoder = create_autoencoder(device=device)
        >>> print(f"Parámetros: {autoencoder.count_parameters():,}")
    """
    model = Autoencoder(in_channels=in_channels, latent_dim=latent_dim)
    model = model.to(device)

    # Inicialización de pesos (Xavier/Glorot para Conv1D)
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    return model


if __name__ == '__main__':
    """
    Script de prueba para validar arquitectura del autoencoder.

    Uso:
        python -m deepsolation.src.models.autoencoder
    """
    print("=" * 70)
    print("TEST: Arquitectura del Autoencoder")
    print("=" * 70)

    # Detectar device disponible (Mac M2 → MPS)
    if torch.backends.mps.is_available():
        device = 'mps'
        print("\n✓ Mac M2 detectado: usando MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("\n✓ CUDA detectado: usando GPU")
    else:
        device = 'cpu'
        print("\n✓ Usando CPU")

    print(f"  Device: {device}\n")

    try:
        # Crear autoencoder
        print("1. Creando autoencoder...")
        autoencoder = create_autoencoder(device=device)
        n_params = autoencoder.count_parameters()
        print(f"   ✓ Autoencoder creado: {n_params:,} parámetros")

        # Verificar encoder
        print("\n2. Verificando encoder...")
        encoder = autoencoder.get_encoder()
        print(f"   ✓ Encoder extraído: {sum(p.numel() for p in encoder.parameters()):,} parámetros")

        # Test forward pass con batch pequeño
        print("\n3. Probando forward pass...")
        batch_size = 4
        x_test = torch.randn(batch_size, 6, 60000).to(device)
        print(f"   Input shape: {x_test.shape}")

        # Autoencoder completo
        reconstruction, latent = autoencoder(x_test)
        print(f"   ✓ Latent shape: {latent.shape}")
        print(f"   ✓ Reconstruction shape: {reconstruction.shape}")

        # Solo encoder
        features = encoder(x_test)
        print(f"   ✓ Encoder features shape: {features.shape}")

        # Verificar dimensiones esperadas
        assert latent.shape == (batch_size, 512), f"Latent shape incorrecta: {latent.shape}"
        assert reconstruction.shape == (batch_size, 6, 60000), f"Reconstruction shape incorrecta: {reconstruction.shape}"
        assert features.shape == (batch_size, 512), f"Features shape incorrecta: {features.shape}"

        print("\n4. Verificando reconstrucción...")
        mse_loss = nn.MSELoss()
        loss = mse_loss(reconstruction, x_test)
        print(f"   MSE Loss (sin entrenar): {loss.item():.6f}")

        print("\n" + "=" * 70)
        print("✅ TEST EXITOSO: Arquitectura funcionando correctamente")
        print("=" * 70 + "\n")

        # Resumen de arquitectura
        print("RESUMEN DE ARQUITECTURA:")
        print(f"  - Input:          (batch, 6, 60000)")
        print(f"  - Latent:         (batch, 512)")
        print(f"  - Reconstruction: (batch, 6, 60000)")
        print(f"  - Parámetros:     {n_params:,}")
        print(f"  - Device:         {device}")
        print()

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
