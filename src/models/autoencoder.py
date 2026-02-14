"""
Autoencoder para aprendizaje no supervisado de señales sísmicas (ETAPA 1).

Arquitectura de 4 capas Conv1D para aprender representaciones latentes
a partir de señales S2-S1 concatenadas (6 canales).

El Encoder usa GlobalAvgPool y acepta cualquier largo de señal.
El Decoder reconstruye al target_length especificado.

Autor: Giancarlo Poémape Lozano
Fecha: 2026-02-07
"""

from typing import Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder del autoencoder: reduce señales (6, L) → latent vector (latent_dim,).

    Acepta señales de cualquier largo L gracias a GlobalAvgPool.

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

        # Bloque 1: (6, L) → (64, L//4)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=11,
            stride=2,
            padding=5
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bloque 2: (64, L//4) → (128, L//8)
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bloque 3: (128, L//8) → (256, L//16)
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bloque 4: (256, L//16) → (512,) tras GlobalAvgPool
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
            x: Tensor de entrada con shape (batch, in_channels, signal_length)

        Returns:
            Latent vector con shape (batch, latent_dim)
        """
        # Bloque 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Bloque 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Bloque 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # Bloque 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # Global Average Pooling
        x = self.global_pool(x)  # (batch, latent_dim, 1)
        x = x.squeeze(-1)        # (batch, latent_dim)

        return x


class Decoder(nn.Module):
    """
    Decoder del autoencoder: reconstruye señales desde latent vector.

    Los tamaños intermedios de upsample se calculan dinámicamente
    a partir de target_length, permitiendo reconstruir a cualquier largo.

    Arquitectura:
        - Linear(latent_dim → latent_dim × initial) + Reshape
        - Layer 1: Upsample + ConvTranspose1D(512→256) + BN + ReLU + Dropout
        - Layer 2: Upsample + ConvTranspose1D(256→128) + BN + ReLU + Dropout
        - Layer 3: Upsample + ConvTranspose1D(128→64) + BN + ReLU + Dropout
        - Layer 4: Upsample + Conv1D(64→6) → Reconstrucción final

    Args:
        latent_dim: Dimensión del espacio latente (default: 512)
        out_channels: Número de canales de salida (default: 6)
        target_length: Largo de la señal a reconstruir (default: 60000)
    """

    def __init__(
        self,
        latent_dim: int = 512,
        out_channels: int = 6,
        target_length: int = 60000
    ):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.target_length = target_length

        # Calcular tamaños intermedios basados en target_length
        self.initial_length = max(target_length // 4000, 2)
        self._len_stage1 = self.initial_length * 4
        self._len_stage2 = target_length // 16
        self._len_stage3 = target_length // 4
        self._len_stage4 = target_length

        # Capa inicial: expandir latent vector a feature map
        self.fc = nn.Linear(latent_dim, latent_dim * self.initial_length)

        # Bloque 1: → (256, len_stage1)
        self.upsample1 = nn.Upsample(
            size=self._len_stage1, mode='linear', align_corners=False
        )
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=latent_dim,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(256)

        # Bloque 2: → (128, len_stage2)
        self.upsample2 = nn.Upsample(
            size=self._len_stage2, mode='linear', align_corners=False
        )
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=256,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.bn2 = nn.BatchNorm1d(128)

        # Bloque 3: → (64, len_stage3)
        self.upsample3 = nn.Upsample(
            size=self._len_stage3, mode='linear', align_corners=False
        )
        self.deconv3 = nn.ConvTranspose1d(
            in_channels=128,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.bn3 = nn.BatchNorm1d(64)

        # Bloque 4 (final): → (out_channels, target_length)
        self.upsample4 = nn.Upsample(
            size=self._len_stage4, mode='linear', align_corners=False
        )
        self.final_conv = nn.Conv1d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=11,
            stride=1,
            padding=5
        )

        # Activación y regularización
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del decoder.

        Args:
            z: Latent vector con shape (batch, latent_dim)

        Returns:
            Reconstrucción con shape (batch, out_channels, target_length)
        """
        batch_size = z.shape[0]

        # Expandir latent vector
        x = self.fc(z)
        x = x.view(batch_size, self.latent_dim, self.initial_length)

        # Bloque 1
        x = self.upsample1(x)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Bloque 2
        x = self.upsample2(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Bloque 3
        x = self.upsample3(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Bloque 4 (final)
        x = self.upsample4(x)
        x = self.final_conv(x)

        return x


class Autoencoder(nn.Module):
    """
    Autoencoder completo para aprendizaje no supervisado de señales sísmicas.

    Combina Encoder y Decoder para reconstrucción.
    El encoder pre-entrenado se puede extraer para la Etapa 2 (clasificación).

    Args:
        in_channels: Número de canales de entrada (default: 6)
        latent_dim: Dimensión del espacio latente (default: 512)
        target_length: Largo de señal para reconstrucción (default: 60000)
    """

    def __init__(
        self,
        in_channels: int = 6,
        latent_dim: int = 512,
        target_length: int = 60000
    ):
        super(Autoencoder, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.target_length = target_length

        # Componentes
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            target_length=target_length
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encoder + decoder.

        Args:
            x: Tensor de entrada con shape (batch, in_channels, signal_length)

        Returns:
            Tuple (reconstruction, latent):
                - reconstruction: Tensor (batch, in_channels, target_length)
                - latent: Tensor (batch, latent_dim)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def get_encoder(self) -> Encoder:
        """
        Extrae el encoder pre-entrenado para usar en ETAPA 2.

        Returns:
            Encoder con pesos entrenados
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
    target_length: int = 60000,
    device: str = 'cpu'
) -> Autoencoder:
    """
    Factory function para crear autoencoder con configuración estándar.

    Args:
        in_channels: Número de canales de entrada
        latent_dim: Dimensión del espacio latente
        target_length: Largo de señal para reconstrucción
        device: Dispositivo PyTorch ('cpu', 'cuda', 'mps')

    Returns:
        Autoencoder inicializado y movido al device especificado
    """
    model = Autoencoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
        target_length=target_length
    )
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

    batch_size = 4

    try:
        # --- Test 1: target_length=60000 (original) ---
        print("1. Test con target_length=60000 (señal completa)...")
        ae_60k = create_autoencoder(target_length=60000, device=device)
        x_60k = torch.randn(batch_size, 6, 60000).to(device)
        recon_60k, latent_60k = ae_60k(x_60k)

        assert latent_60k.shape == (batch_size, 512)
        assert recon_60k.shape == (batch_size, 6, 60000)
        print(f"   ✓ Input:  {x_60k.shape}")
        print(f"   ✓ Latent: {latent_60k.shape}")
        print(f"   ✓ Recon:  {recon_60k.shape}")
        print(f"   ✓ Params: {ae_60k.count_parameters():,}")

        # Decoder intermediate sizes
        dec = ae_60k.decoder
        print(f"   ✓ Decoder stages: {dec.initial_length} → {dec._len_stage1}"
              f" → {dec._len_stage2} → {dec._len_stage3} → {dec._len_stage4}")

        # --- Test 2: target_length=10000 (windowed) ---
        print("\n2. Test con target_length=10000 (ventanas de 100s)...")
        ae_10k = create_autoencoder(target_length=10000, device=device)
        x_10k = torch.randn(batch_size, 6, 10000).to(device)
        recon_10k, latent_10k = ae_10k(x_10k)

        assert latent_10k.shape == (batch_size, 512)
        assert recon_10k.shape == (batch_size, 6, 10000)
        print(f"   ✓ Input:  {x_10k.shape}")
        print(f"   ✓ Latent: {latent_10k.shape}")
        print(f"   ✓ Recon:  {recon_10k.shape}")
        print(f"   ✓ Params: {ae_10k.count_parameters():,}")

        dec10 = ae_10k.decoder
        print(f"   ✓ Decoder stages: {dec10.initial_length} → {dec10._len_stage1}"
              f" → {dec10._len_stage2} → {dec10._len_stage3} → {dec10._len_stage4}")

        # --- Test 3: Encoder acepta ambos largos ---
        print("\n3. Verificando que encoder produce mismo latent_dim...")
        encoder = ae_60k.get_encoder()
        feat_60k = encoder(x_60k)
        feat_10k = encoder(x_10k)
        print(f"   ✓ Encoder(60000) → {feat_60k.shape}")
        print(f"   ✓ Encoder(10000) → {feat_10k.shape}")
        assert feat_60k.shape == feat_10k.shape

        print("\n" + "=" * 70)
        print("✅ TEST EXITOSO: Autoencoder adaptable a target_length")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
