"""Decoder for reconstructing mel spectrogram from disentangled representations.

Combines content, prosody, and speaker embeddings to reconstruct the original
mel spectrogram, ensuring all information is captured in the representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConvBlock(nn.Module):
    """Convolutional block with residual connection."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, 
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        residual = x
        
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm1(x + residual)
        
        return x


class Decoder(nn.Module):
    """Decoder that reconstructs mel spectrogram from disentangled representations.
    
    Architecture:
    1. Combine content, prosody, and speaker embeddings
    2. Transformer decoder layers for temporal modeling
    3. Conv layers for local patterns
    4. Output projection to mel spectrogram
    """
    
    def __init__(
        self,
        content_size: int = 256,
        prosody_size: int = 64,
        speaker_size: int = 128,
        hidden_size: int = 512,
        output_size: int = 80,  # n_mels
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 2000
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input projection: combine all embeddings
        total_input = content_size + prosody_size + speaker_size
        self.input_proj = nn.Linear(total_input, hidden_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, max_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers)
        
        # Conv blocks for local patterns
        self.conv_blocks = nn.ModuleList([
            ConvBlock(hidden_size, kernel_size=5, dropout=dropout)
            for _ in range(2)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        # Optional: predict variance for probabilistic reconstruction
        self.predict_variance = False
        if self.predict_variance:
            self.variance_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size),
                nn.Softplus()
            )
    
    def forward(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor,
        speaker: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            content: Content embeddings [B, T, content_size]
            prosody: Prosody embeddings [B, T, prosody_size]
            speaker: Speaker embedding [B, speaker_size]
            lengths: Sequence lengths [B]
            
        Returns:
            mel: Reconstructed mel spectrogram [B, n_mels, T]
            variance: Optional variance [B, n_mels, T]
        """
        B, T, _ = content.shape
        
        # Expand speaker embedding to all time steps
        speaker_expanded = speaker.unsqueeze(1).expand(-1, T, -1)  # [B, T, speaker_size]
        
        # Concatenate all embeddings
        combined = torch.cat([content, prosody, speaker_expanded], dim=-1)
        
        # Project to hidden size
        x = self.input_proj(combined)  # [B, T, hidden]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create mask for padding
        mask = None
        if lengths is not None:
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Output projection
        mel = self.output_proj(x)  # [B, T, n_mels]
        mel = mel.transpose(1, 2)  # [B, n_mels, T]
        
        # Variance if needed
        variance = None
        if self.predict_variance:
            variance = self.variance_proj(x).transpose(1, 2)
        
        return mel, variance
    
    def inference(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor,
        speaker: torch.Tensor
    ) -> torch.Tensor:
        """Inference mode without variance."""
        mel, _ = self.forward(content, prosody, speaker)
        return mel


class VariationalDecoder(nn.Module):
    """VAE-style decoder with latent sampling.
    
    Adds stochasticity for better generation quality.
    """
    
    def __init__(
        self,
        content_size: int = 256,
        prosody_size: int = 64,
        speaker_size: int = 128,
        latent_size: int = 64,
        hidden_size: int = 512,
        output_size: int = 80,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        total_input = content_size + prosody_size + speaker_size
        
        # Encoder to latent
        self.to_latent = nn.Sequential(
            nn.Linear(total_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2)  # mean and logvar
        )
        
        # Decoder from latent
        self.from_latent = nn.Linear(latent_size + speaker_size, hidden_size)
        
        # Main decoder
        self.decoder = Decoder(
            content_size=hidden_size,
            prosody_size=0,  # Already included
            speaker_size=0,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.latent_size = latent_size
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor,
        speaker: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mel: Reconstructed mel
            mu: Latent mean
            logvar: Latent log variance
        """
        B, T, _ = content.shape
        
        # Combine inputs
        combined = torch.cat([content, prosody, speaker.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        
        # Encode to latent
        latent_params = self.to_latent(combined)
        mu, logvar = latent_params.chunk(2, dim=-1)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        speaker_expanded = speaker.unsqueeze(1).expand(-1, T, -1)
        decoder_input = self.from_latent(torch.cat([z, speaker_expanded], dim=-1))
        
        mel, _ = self.decoder(
            decoder_input,
            torch.zeros(B, T, 0, device=content.device),
            torch.zeros(B, 0, device=content.device),
            lengths
        )
        
        return mel, mu, logvar
