"""Prosody Encoder for extracting pitch and rhythm information.

Specifically designed to capture tonal patterns in Lhasa/Kham Tibetan
and rhythmic patterns in Amdo Tibetan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ProsodyEncoder(nn.Module):
    """Prosody encoder for F0 and energy features.
    
    Architecture:
    1. Separate encoders for F0 and energy
    2. BiLSTM to capture temporal patterns
    3. Attention mechanism for important prosodic events
    4. Fusion layer to combine F0 and energy
    
    Key design for Tibetan:
    - Handles voiced/unvoiced (important for Amdo consonant clusters)
    - Captures both local (tone) and global (intonation) patterns
    """
    
    def __init__(
        self,
        f0_input_size: int = 1,
        energy_input_size: int = 1,
        hidden_size: int = 256,
        output_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_attention = use_attention
        
        lstm_hidden = hidden_size // 2 if bidirectional else hidden_size
        
        # F0 encoder
        self.f0_conv = nn.Sequential(
            nn.Conv1d(f0_input_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 2, hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.f0_lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Energy encoder
        self.energy_conv = nn.Sequential(
            nn.Conv1d(energy_input_size, hidden_size // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 4, hidden_size // 4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.energy_lstm = nn.LSTM(
            input_size=hidden_size // 4,
            hidden_size=lstm_hidden // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Voiced/unvoiced embedding
        self.voiced_embedding = nn.Embedding(2, hidden_size // 4)
        
        # Attention for prosody
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size + hidden_size // 2,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Fusion layer
        fusion_input_size = hidden_size + hidden_size // 2 + hidden_size // 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size)
        )
        
        # Global prosody pooling (for utterance-level features)
        self.global_pool = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.Tanh()
        )
        
    def forward(
        self,
        f0: torch.Tensor,
        energy: torch.Tensor,
        voiced: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f0: Normalized F0 contour [B, T]
            energy: Energy contour [B, T]
            voiced: Voiced/unvoiced flags [B, T]
            lengths: Sequence lengths [B]
            
        Returns:
            prosody: Frame-level prosody embeddings [B, T, output_size]
            prosody_global: Utterance-level prosody [B, output_size]
        """
        B, T = f0.shape
        
        # Add channel dimension: [B, T] -> [B, 1, T]
        f0_input = f0.unsqueeze(1)
        energy_input = energy.unsqueeze(1)
        
        # F0 encoding
        f0_conv = self.f0_conv(f0_input)  # [B, hidden//2, T]
        f0_conv = f0_conv.transpose(1, 2)  # [B, T, hidden//2]
        f0_encoded, _ = self.f0_lstm(f0_conv)  # [B, T, hidden]
        
        # Energy encoding
        energy_conv = self.energy_conv(energy_input)  # [B, hidden//4, T]
        energy_conv = energy_conv.transpose(1, 2)  # [B, T, hidden//4]
        energy_encoded, _ = self.energy_lstm(energy_conv)  # [B, T, hidden//2]
        
        # Voiced embedding
        if voiced is None:
            voiced = (f0.abs() > 0).long()
        voiced_emb = self.voiced_embedding(voiced.long())  # [B, T, hidden//4]
        
        # Concatenate all features
        combined = torch.cat([f0_encoded, energy_encoded, voiced_emb], dim=-1)
        # [B, T, hidden + hidden//2 + hidden//4]
        
        # Self-attention for prosodic patterns
        if self.use_attention:
            # Create attention mask for padding
            attn_mask = None
            if lengths is not None:
                attn_mask = torch.arange(T, device=f0.device)[None, :] >= lengths[:, None]
            
            combined_attn, _ = self.attention(
                combined, combined, combined,
                key_padding_mask=attn_mask
            )
            combined = combined + combined_attn  # Residual
        
        # Fusion to prosody embeddings
        prosody = self.fusion(combined)  # [B, T, output_size]
        
        # Global prosody (mean pooling with length masking)
        if lengths is not None:
            mask = torch.arange(T, device=f0.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            prosody_sum = (prosody * mask).sum(dim=1)
            prosody_global = prosody_sum / lengths.unsqueeze(-1).float()
        else:
            prosody_global = prosody.mean(dim=1)
        
        prosody_global = self.global_pool(prosody_global)  # [B, output_size]
        
        return prosody, prosody_global
    
    def encode_f0_only(self, f0: torch.Tensor) -> torch.Tensor:
        """Encode only F0 (for tone analysis)."""
        f0_input = f0.unsqueeze(1)
        f0_conv = self.f0_conv(f0_input)
        f0_conv = f0_conv.transpose(1, 2)
        f0_encoded, _ = self.f0_lstm(f0_conv)
        return f0_encoded


class ToneClassifier(nn.Module):
    """Classifier for Tibetan lexical tones (for Lhasa/Kham dialects).
    
    Tibetan tones:
    - High falling (typical in Lhasa)
    - Low rising
    - etc.
    """
    
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 128,
        num_tones: int = 4,  # Tibetan typically has 2-4 tones
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tones)
        )
        
    def forward(self, prosody_global: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prosody_global: Global prosody embedding [B, input_size]
            
        Returns:
            tone_logits: Tone class logits [B, num_tones]
        """
        return self.classifier(prosody_global)


class DialectProsodyAdapter(nn.Module):
    """Adapter for dialect-specific prosody handling.
    
    Handles the fundamental difference:
    - Lhasa/Kham: Lexical tone (pitch matters for meaning)
    - Amdo: No lexical tone (pitch is only for intonation)
    """
    
    def __init__(
        self,
        prosody_size: int = 64,
        num_dialects: int = 3
    ):
        super().__init__()
        
        # Dialect-specific prosody transformations
        self.dialect_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prosody_size, prosody_size),
                nn.ReLU(),
                nn.Linear(prosody_size, prosody_size)
            )
            for _ in range(num_dialects)
        ])
        
        # Shared output normalization
        self.output_norm = nn.LayerNorm(prosody_size)
        
    def forward(
        self,
        prosody: torch.Tensor,
        dialect_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            prosody: Prosody embeddings [B, T, prosody_size]
            dialect_id: Dialect indices [B]
            
        Returns:
            adapted_prosody: Dialect-adapted prosody [B, T, prosody_size]
        """
        B, T, D = prosody.shape
        output = torch.zeros_like(prosody)
        
        for dialect_idx in range(len(self.dialect_transforms)):
            mask = (dialect_id == dialect_idx).unsqueeze(1).unsqueeze(2)
            transformed = self.dialect_transforms[dialect_idx](prosody)
            output = output + mask.float() * transformed
        
        return self.output_norm(output + prosody)  # Residual connection
