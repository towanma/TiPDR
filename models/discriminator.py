"""Discriminators and Gradient Reversal for adversarial disentanglement.

Key components:
1. GradientReversalLayer: Enables adversarial training
2. ContentDiscriminator: Tries to predict prosody from content (should fail)
3. ProsodyDiscriminator: Tries to predict content from prosody (should fail)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer function.
    
    In forward pass: identity
    In backward pass: multiply gradient by -lambda
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for adversarial training.
    
    When training:
    - Forward: passes through unchanged
    - Backward: reverses gradients, making encoder try to fool discriminator
    """
    
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float):
        """Adjust the gradient scaling factor."""
        self.lambda_ = lambda_


class ContentDiscriminator(nn.Module):
    """Discriminator that tries to predict prosody/dialect from content.
    
    Goal: Content encoder should produce representations where this
    discriminator CANNOT predict prosody features (adversarial training).
    """
    
    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,  # 3 dialects
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Gradient reversal
        self.grl = GradientReversalLayer()
        
        # Frame-level processing
        layers = []
        in_size = input_size
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_size = hidden_size
        
        self.frame_encoder = nn.Sequential(*layers)
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(
        self,
        content: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        apply_grl: bool = True
    ) -> torch.Tensor:
        """
        Args:
            content: Content embeddings [B, T, input_size]
            lengths: Sequence lengths [B]
            apply_grl: Whether to apply gradient reversal
            
        Returns:
            logits: Dialect/prosody class logits [B, num_classes]
        """
        B, T, _ = content.shape
        
        # Apply gradient reversal
        if apply_grl:
            content = self.grl(content)
        
        # Encode frames
        x = self.frame_encoder(content)  # [B, T, hidden]
        
        # Attention pooling
        attn = self.attention(x)  # [B, T, 1]
        
        if lengths is not None:
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
            attn = attn.masked_fill(mask.unsqueeze(-1), -1e9)
        
        attn = F.softmax(attn, dim=1)
        pooled = (x * attn).sum(dim=1)  # [B, hidden]
        
        # Classify
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits
    
    def set_grl_lambda(self, lambda_: float):
        """Adjust GRL strength during training."""
        self.grl.set_lambda(lambda_)


class ProsodyDiscriminator(nn.Module):
    """Discriminator that tries to predict content/words from prosody.
    
    Goal: Prosody encoder should produce representations where this
    discriminator CANNOT predict content features.
    """
    
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_content_classes: int = 512,  # VQ codebook size
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Gradient reversal
        self.grl = GradientReversalLayer()
        
        # Frame-level classifier
        layers = []
        in_size = input_size
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_size = hidden_size
        
        self.encoder = nn.Sequential(*layers)
        
        # Per-frame content prediction
        self.classifier = nn.Linear(hidden_size, num_content_classes)
        
    def forward(
        self,
        prosody: torch.Tensor,
        apply_grl: bool = True
    ) -> torch.Tensor:
        """
        Args:
            prosody: Prosody embeddings [B, T, input_size]
            apply_grl: Whether to apply gradient reversal
            
        Returns:
            logits: Content class logits [B, T, num_content_classes]
        """
        if apply_grl:
            prosody = self.grl(prosody)
        
        x = self.encoder(prosody)
        logits = self.classifier(x)
        
        return logits
    
    def set_grl_lambda(self, lambda_: float):
        self.grl.set_lambda(lambda_)


class SpeakerDiscriminator(nn.Module):
    """Discriminator that tries to predict speaker from content/prosody.
    
    Ensures content and prosody encoders don't leak speaker information.
    """
    
    def __init__(
        self,
        content_size: int = 256,
        prosody_size: int = 64,
        hidden_size: int = 256,
        num_speakers: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.grl = GradientReversalLayer()
        
        input_size = content_size + prosody_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.classifier = nn.Linear(hidden_size, num_speakers)
        
    def forward(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        apply_grl: bool = True
    ) -> torch.Tensor:
        """
        Args:
            content: Content embeddings [B, T, content_size]
            prosody: Prosody embeddings [B, T, prosody_size]
            
        Returns:
            logits: Speaker logits [B, num_speakers]
        """
        B, T, _ = content.shape
        
        # Concatenate
        x = torch.cat([content, prosody], dim=-1)
        
        if apply_grl:
            x = self.grl(x)
        
        x = self.encoder(x)
        
        # Attention pooling
        attn = self.attention(x)
        if lengths is not None:
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
            attn = attn.masked_fill(mask.unsqueeze(-1), -1e9)
        attn = F.softmax(attn, dim=1)
        
        pooled = (x * attn).sum(dim=1)
        
        return self.classifier(pooled)
    
    def set_grl_lambda(self, lambda_: float):
        self.grl.set_lambda(lambda_)


class MutualInformationEstimator(nn.Module):
    """Neural network based mutual information estimator (MINE-style).
    
    Estimates MI between content and prosody to minimize it during training.
    Lower MI means better disentanglement.
    """
    
    def __init__(
        self,
        content_size: int = 256,
        prosody_size: int = 64,
        hidden_size: int = 256
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(content_size + prosody_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate mutual information using MINE.
        
        Args:
            content: Content embeddings [B, T, content_size]
            prosody: Prosody embeddings [B, T, prosody_size]
            
        Returns:
            mi_estimate: Estimated MI (scalar)
        """
        B, T, _ = content.shape
        
        # Joint samples (content_i, prosody_i)
        joint = torch.cat([content, prosody], dim=-1)  # [B, T, C+P]
        joint_scores = self.network(joint).mean()
        
        # Marginal samples (content_i, prosody_j) - shuffle prosody
        prosody_shuffled = prosody[torch.randperm(B)]
        marginal = torch.cat([content, prosody_shuffled], dim=-1)
        marginal_scores = self.network(marginal)
        
        # MINE lower bound: E[T(x,y)] - log(E[exp(T(x,y'))])
        mi_estimate = joint_scores - torch.logsumexp(
            marginal_scores.view(-1), dim=0
        ) + torch.log(torch.tensor(B * T, dtype=torch.float, device=content.device))
        
        return mi_estimate


class ContrastiveDisentangler(nn.Module):
    """Contrastive learning based disentanglement.
    
    Uses InfoNCE-style loss to push apart content and prosody representations.
    """
    
    def __init__(
        self,
        content_size: int = 256,
        prosody_size: int = 64,
        proj_size: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Project to same dimension
        self.content_proj = nn.Sequential(
            nn.Linear(content_size, proj_size),
            nn.ReLU(),
            nn.Linear(proj_size, proj_size)
        )
        
        self.prosody_proj = nn.Sequential(
            nn.Linear(prosody_size, proj_size),
            nn.ReLU(),
            nn.Linear(proj_size, proj_size)
        )
    
    def forward(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive disentanglement loss.
        
        We want content and prosody to be DISSIMILAR (negative pairs).
        """
        # Global pooling
        content_global = content.mean(dim=1)  # [B, content_size]
        prosody_global = prosody.mean(dim=1)  # [B, prosody_size]
        
        # Project
        content_proj = F.normalize(self.content_proj(content_global), dim=-1)
        prosody_proj = F.normalize(self.prosody_proj(prosody_global), dim=-1)
        
        # Compute similarity matrix
        sim = content_proj @ prosody_proj.t() / self.temperature  # [B, B]
        
        # We want LOW similarity between content and prosody
        # So we maximize the negative of InfoNCE (or minimize cosine similarity)
        disentangle_loss = sim.diag().mean()  # Minimize diagonal similarities
        
        return disentangle_loss
