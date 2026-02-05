"""Content Encoder based on HuBERT with VQ-VAE discretization.

The content encoder learns "what is being said" while removing prosodic 
information (pitch, rhythm) using Instance Normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from transformers import HubertModel, HubertConfig


class InstanceNorm1d(nn.Module):
    """Instance Normalization for removing style information."""
    
    def __init__(self, num_features: int, affine: bool = False):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=affine)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T] -> norm -> [B, T, C]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA updates.
    
    Discretizes continuous representations into a fixed codebook,
    which helps remove speaker/prosody specific information.
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # EMA parameters (not trained)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())
        
    def forward(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input tensor [B, T, D]
            
        Returns:
            quantized: Quantized output [B, T, D]
            loss: VQ loss (commitment + codebook loss)
            encoding_indices: Indices of nearest embeddings [B, T]
        """
        B, T, D = z.shape
        
        # Flatten
        flat_z = z.reshape(-1, D)  # [B*T, D]
        
        # Calculate distances
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * flat_z @ self.embedding.weight.t()
        )  # [B*T, num_embeddings]
        
        # Find nearest embeddings
        encoding_indices = distances.argmin(dim=1)  # [B*T]
        
        # Quantize
        quantized_flat = self.embedding(encoding_indices)  # [B*T, D]
        quantized = quantized_flat.view(B, T, D)
        
        # EMA updates (training only)
        if self.training:
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            
            # Update cluster sizes
            self.ema_cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            
            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size.data.add_(self.epsilon).div_(
                n + self.num_embeddings * self.epsilon
            ).mul_(n)
            
            # Update embeddings
            dw = encodings.t() @ flat_z
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            self.embedding.weight.data.copy_(
                self.ema_w / self.ema_cluster_size.unsqueeze(1)
            )
        
        # Loss: commitment loss (encoder output should stay close to codebook)
        commitment_loss = F.mse_loss(z, quantized.detach())
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        loss = self.commitment_cost * commitment_loss
        
        return quantized, loss, encoding_indices.view(B, T)
    
    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        """Calculate percentage of codebook entries used."""
        unique = torch.unique(indices)
        return len(unique) / self.num_embeddings


class ContentEncoder(nn.Module):
    """Content encoder using HuBERT backbone with VQ discretization.
    
    Architecture:
    1. HuBERT extracts contextualized speech representations
    2. Instance Normalization removes prosodic style
    3. Projection layer maps to target dimension
    4. VQ-VAE discretizes the content
    """
    
    def __init__(
        self,
        pretrained_hubert: str = "facebook/hubert-base-ls960",
        freeze_hubert: bool = False,
        hubert_hidden_size: int = 768,
        output_size: int = 256,
        use_instance_norm: bool = True,
        vq_num_embeddings: int = 512,
        vq_commitment_cost: float = 0.25,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.use_instance_norm = use_instance_norm
        self.output_size = output_size
        
        # Load pretrained HuBERT
        try:
            self.hubert = HubertModel.from_pretrained(pretrained_hubert)
        except Exception:
            # Fallback: initialize from config
            config = HubertConfig()
            self.hubert = HubertModel(config)
            print("Warning: Could not load pretrained HuBERT, using random init")
        
        if freeze_hubert:
            for param in self.hubert.parameters():
                param.requires_grad = False
        
        # Instance normalization for style removal
        if use_instance_norm:
            self.instance_norm = InstanceNorm1d(hubert_hidden_size)
        
        # Projection to content space
        self.projection = nn.Sequential(
            nn.Linear(hubert_hidden_size, output_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 2, output_size),
            nn.LayerNorm(output_size)
        )
        
        # Vector quantizer
        self.vq = VectorQuantizer(
            num_embeddings=vq_num_embeddings,
            embedding_dim=output_size,
            commitment_cost=vq_commitment_cost
        )
        
    def forward(
        self, 
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_pre_vq: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform: Raw audio waveform [B, T_audio]
            attention_mask: Mask for padded positions [B, T_audio]
            return_pre_vq: Whether to also return pre-VQ representations
            
        Returns:
            content: Quantized content embeddings [B, T, D]
            vq_loss: Vector quantization loss
            indices: Codebook indices [B, T]
        """
        # Extract HuBERT features
        outputs = self.hubert(
            waveform, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state
        hidden = outputs.last_hidden_state  # [B, T, 768]
        
        # Instance normalization to remove style
        if self.use_instance_norm:
            hidden = self.instance_norm(hidden)
        
        # Project to content space
        content_pre_vq = self.projection(hidden)  # [B, T, output_size]
        
        # Vector quantization
        content, vq_loss, indices = self.vq(content_pre_vq)
        
        if return_pre_vq:
            return content, vq_loss, indices, content_pre_vq
        
        return content, vq_loss, indices
    
    def encode(
        self, 
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio to content representation (inference mode)."""
        content, _, indices = self.forward(waveform, attention_mask)
        return content, indices
    
    def get_codebook_indices(
        self, 
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get only the codebook indices (for ASR or analysis)."""
        _, _, indices = self.forward(waveform, attention_mask)
        return indices


class ContentEncoderMel(nn.Module):
    """Alternative content encoder using mel spectrogram input.
    
    For cases where HuBERT is too heavy or when working with 
    preprocessed mel spectrograms.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        hidden_size: int = 512,
        output_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        vq_num_embeddings: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Instance normalization
        self.instance_norm = nn.InstanceNorm1d(hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size)
        )
        
        # Vector quantizer
        self.vq = VectorQuantizer(
            num_embeddings=vq_num_embeddings,
            embedding_dim=output_size
        )
    
    def forward(
        self, 
        mel: torch.Tensor,
        mel_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: Mel spectrogram [B, n_mels, T]
            mel_lengths: Length of each mel sequence
            
        Returns:
            content: Quantized content [B, T, D]
            vq_loss: VQ loss
            indices: Codebook indices [B, T]
        """
        # Conv layers: [B, n_mels, T] -> [B, hidden, T]
        x = self.input_proj(mel)
        
        # Instance norm for style removal
        x = self.instance_norm(x)
        
        # [B, hidden, T] -> [B, T, hidden]
        x = x.transpose(1, 2)
        
        # Create mask if lengths provided
        mask = None
        if mel_lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] >= mel_lengths[:, None]
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Project to content space
        content_pre_vq = self.output_proj(x)
        
        # Vector quantization
        content, vq_loss, indices = self.vq(content_pre_vq)
        
        return content, vq_loss, indices
