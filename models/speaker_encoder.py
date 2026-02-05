"""Speaker Encoder for extracting speaker identity embeddings.

Extracts speaker characteristics (timbre, voice quality) independent
of content and prosody.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SpeakerEncoder(nn.Module):
    """Speaker encoder using mel spectrogram input.
    
    Based on the d-vector approach with improvements:
    1. Conv + LSTM architecture
    2. Attentive statistics pooling
    3. Speaker embedding projection
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        hidden_size: int = 256,
        output_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Conv frontend
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attentive statistics pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, output_size)
        )
        
        # L2 normalization for speaker embedding
        self.normalize = True
        
    def forward(
        self,
        mel: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            mel: Mel spectrogram [B, n_mels, T]
            lengths: Sequence lengths [B]
            
        Returns:
            speaker_emb: Speaker embedding [B, output_size]
        """
        B, _, T = mel.shape
        
        # Conv frontend: [B, n_mels, T] -> [B, hidden, T]
        x = self.conv(mel)
        
        # Transpose for LSTM: [B, hidden, T] -> [B, T, hidden]
        x = x.transpose(1, 2)
        
        # Pack for variable length if needed
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM: [B, T, hidden] -> [B, T, hidden*2]
        x, _ = self.lstm(x)
        
        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        # Attentive statistics pooling
        attn_weights = self.attention(x)  # [B, T, 1]
        
        # Mask padding positions
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=1)  # [B, T, 1]
        
        # Weighted mean
        mean = (x * attn_weights).sum(dim=1)  # [B, hidden*2]
        
        # Weighted std
        std = torch.sqrt(
            ((x - mean.unsqueeze(1)) ** 2 * attn_weights).sum(dim=1) + 1e-8
        )  # [B, hidden*2]
        
        # Concatenate mean and std
        pooled = torch.cat([mean, std], dim=-1)  # [B, hidden*4]
        
        # Project to speaker embedding
        speaker_emb = self.projection(pooled)  # [B, output_size]
        
        # L2 normalize
        if self.normalize:
            speaker_emb = F.normalize(speaker_emb, p=2, dim=-1)
        
        return speaker_emb
    
    def get_frame_embeddings(
        self,
        mel: torch.Tensor
    ) -> torch.Tensor:
        """Get frame-level speaker features (before pooling)."""
        x = self.conv(mel)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return x


class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN speaker encoder.
    
    State-of-the-art speaker verification model with:
    - SE-Res2Block
    - Multi-layer feature aggregation
    - Channel and context dependent statistics pooling
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        channels: int = 512,
        output_size: int = 128
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_mels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        
        # SE-Res2Blocks
        self.layer1 = SERes2Block(channels, channels, kernel_size=3, dilation=2)
        self.layer2 = SERes2Block(channels, channels, kernel_size=3, dilation=3)
        self.layer3 = SERes2Block(channels, channels, kernel_size=3, dilation=4)
        
        # Multi-layer feature aggregation
        self.mfa = nn.Conv1d(channels * 3, channels * 3, kernel_size=1)
        
        # Attentive stats pooling
        self.asp = AttentiveStatsPool(channels * 3, 128)
        
        # Final layers
        self.bn2 = nn.BatchNorm1d(channels * 6)
        self.fc = nn.Linear(channels * 6, output_size)
        self.bn3 = nn.BatchNorm1d(output_size)
        
    def forward(
        self,
        mel: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            mel: Mel spectrogram [B, n_mels, T]
            
        Returns:
            speaker_emb: Speaker embedding [B, output_size]
        """
        x = F.relu(self.bn1(self.conv1(mel)))
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        # Multi-layer aggregation
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.mfa(x))
        
        # Attentive stats pooling
        x = self.asp(x, lengths)
        
        # Final projection
        x = self.bn2(x)
        x = self.fc(x)
        x = self.bn3(x)
        
        return F.normalize(x, p=2, dim=-1)


class SERes2Block(nn.Module):
    """SE-Res2Block for ECAPA-TDNN."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8
    ):
        super().__init__()
        
        self.scale = scale
        width = in_channels // scale
        
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size, padding=dilation, dilation=dilation)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(width) for _ in range(scale - 1)
        ])
        
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # SE block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 8),
            nn.ReLU(),
            nn.Linear(out_channels // 8, out_channels),
            nn.Sigmoid()
        )
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Split and process with Res2Net style
        xs = torch.chunk(x, self.scale, dim=1)
        ys = []
        for i in range(self.scale):
            if i == 0:
                ys.append(xs[i])
            elif i == 1:
                ys.append(F.relu(self.bns[i-1](self.convs[i-1](xs[i]))))
            else:
                ys.append(F.relu(self.bns[i-1](self.convs[i-1](xs[i] + ys[-1]))))
        
        x = torch.cat(ys, dim=1)
        x = self.bn3(self.conv3(x))
        
        # SE attention
        se = self.se(x).unsqueeze(-1)
        x = x * se
        
        return F.relu(x + residual)


class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling."""
    
    def __init__(self, in_channels: int, attention_channels: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(attention_channels, in_channels, kernel_size=1),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Features [B, C, T]
            lengths: Sequence lengths [B]
            
        Returns:
            pooled: [B, C*2]
        """
        attn = self.attention(x)  # [B, C, T]
        
        # Mask if needed
        if lengths is not None:
            B, C, T = x.shape
            mask = torch.arange(T, device=x.device)[None, None, :] >= lengths[:, None, None]
            attn = attn.masked_fill(mask, 0)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted mean and std
        mean = (x * attn).sum(dim=-1)
        std = torch.sqrt(((x - mean.unsqueeze(-1)) ** 2 * attn).sum(dim=-1) + 1e-8)
        
        return torch.cat([mean, std], dim=-1)


class SpeakerClassifier(nn.Module):
    """Speaker classification head for training with speaker labels."""
    
    def __init__(
        self,
        input_size: int = 128,
        num_speakers: int = 100,
        use_aam_softmax: bool = True,
        scale: float = 30.0,
        margin: float = 0.2
    ):
        super().__init__()
        
        self.use_aam_softmax = use_aam_softmax
        self.scale = scale
        self.margin = margin
        
        if use_aam_softmax:
            self.weight = nn.Parameter(torch.randn(num_speakers, input_size))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.fc = nn.Linear(input_size, num_speakers)
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Speaker embeddings [B, input_size]
            labels: Speaker labels [B] (required for AAM-softmax training)
            
        Returns:
            logits: Speaker logits [B, num_speakers]
        """
        if self.use_aam_softmax:
            # Normalize
            x = F.normalize(x, p=2, dim=-1)
            w = F.normalize(self.weight, p=2, dim=-1)
            
            # Cosine similarity
            cosine = x @ w.t()  # [B, num_speakers]
            
            if labels is not None and self.training:
                # AAM-softmax margin
                one_hot = F.one_hot(labels, num_classes=w.size(0)).float()
                cosine = cosine - one_hot * self.margin
            
            return cosine * self.scale
        else:
            return self.fc(x)
