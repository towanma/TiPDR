"""Loss functions for prosody-disentangled speech representation learning.

Includes:
1. Reconstruction loss: Ensure all information is captured
2. VQ loss: Discrete content representation
3. Adversarial loss: Disentangle prosody from content
4. MI minimization: Further enforce disentanglement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ReconstructionLoss(nn.Module):
    """Mel spectrogram reconstruction loss.
    
    Combines multiple reconstruction objectives:
    1. L1 loss (primary)
    2. L2 loss (smoothness)
    3. SSIM-like loss (structural)
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.5,
        use_spectral_convergence: bool = True
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.use_spectral_convergence = use_spectral_convergence
        
    def forward(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_mel: Predicted mel [B, n_mels, T]
            target_mel: Target mel [B, n_mels, T]
            lengths: Sequence lengths [B]
            
        Returns:
            Dictionary with loss components
        """
        # Create mask for variable length
        if lengths is not None:
            B, _, T = pred_mel.shape
            mask = torch.arange(T, device=pred_mel.device)[None, None, :] < lengths[:, None, None]
            mask = mask.float()
            
            # Masked losses
            diff = (pred_mel - target_mel) * mask
            
            l1_loss = diff.abs().sum() / mask.sum()
            l2_loss = (diff ** 2).sum() / mask.sum()
        else:
            l1_loss = F.l1_loss(pred_mel, target_mel)
            l2_loss = F.mse_loss(pred_mel, target_mel)
        
        # Spectral convergence loss
        sc_loss = torch.tensor(0.0, device=pred_mel.device)
        if self.use_spectral_convergence:
            if lengths is not None:
                norm_target = (target_mel * mask).norm(dim=(1, 2))
                norm_diff = (diff).norm(dim=(1, 2))
                sc_loss = (norm_diff / (norm_target + 1e-8)).mean()
            else:
                sc_loss = (pred_mel - target_mel).norm() / (target_mel.norm() + 1e-8)
        
        total = self.l1_weight * l1_loss + self.l2_weight * l2_loss + 0.5 * sc_loss
        
        return {
            "reconstruction_total": total,
            "reconstruction_l1": l1_loss,
            "reconstruction_l2": l2_loss,
            "spectral_convergence": sc_loss
        }


class VQLoss(nn.Module):
    """Vector quantization related losses.
    
    1. Commitment loss: Encoder commits to codebook
    2. Codebook diversity: Encourage full codebook usage
    """
    
    def __init__(
        self,
        commitment_weight: float = 0.25,
        diversity_weight: float = 0.1
    ):
        super().__init__()
        
        self.commitment_weight = commitment_weight
        self.diversity_weight = diversity_weight
        
    def forward(
        self,
        vq_loss: torch.Tensor,
        indices: torch.Tensor,
        num_embeddings: int
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vq_loss: VQ commitment loss from quantizer
            indices: Codebook indices [B, T]
            num_embeddings: Total codebook size
            
        Returns:
            Dictionary with loss components
        """
        # Commitment loss (already computed in VQ layer)
        commitment = self.commitment_weight * vq_loss
        
        # Codebook usage diversity loss
        # Encourage uniform usage of codebook entries
        B, T = indices.shape
        one_hot = F.one_hot(indices.view(-1), num_classes=num_embeddings).float()
        usage = one_hot.mean(dim=0)  # [num_embeddings]
        
        # Entropy of usage distribution (higher is better)
        entropy = -(usage * torch.log(usage + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(num_embeddings, dtype=torch.float, device=indices.device))
        
        # Loss: minimize negative entropy
        diversity_loss = -self.diversity_weight * (entropy / max_entropy)
        
        # Codebook utilization metric
        utilization = (usage > 1e-6).float().mean()
        
        total = commitment + diversity_loss
        
        return {
            "vq_total": total,
            "vq_commitment": commitment,
            "vq_diversity": diversity_loss,
            "codebook_utilization": utilization
        }


class AdversarialLoss(nn.Module):
    """Adversarial losses for disentanglement.
    
    1. Content discriminator: Content should not predict prosody/dialect
    2. Prosody discriminator: Prosody should not predict content
    3. Speaker discriminator: Neither should predict speaker
    """
    
    def __init__(
        self,
        content_weight: float = 0.1,
        prosody_weight: float = 0.1,
        speaker_weight: float = 0.1,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.content_weight = content_weight
        self.prosody_weight = prosody_weight
        self.speaker_weight = speaker_weight
        self.label_smoothing = label_smoothing
        
    def forward(
        self,
        content_disc_logits: Optional[torch.Tensor] = None,
        dialect_labels: Optional[torch.Tensor] = None,
        prosody_disc_logits: Optional[torch.Tensor] = None,
        content_indices: Optional[torch.Tensor] = None,
        speaker_disc_logits: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adversarial losses.
        
        For the encoders (through GRL), we want discriminators to FAIL.
        For the discriminators themselves, we want them to succeed.
        
        The GRL handles this automatically - encoder gets reversed gradients.
        """
        losses = {}
        total = torch.tensor(0.0, device=content_disc_logits.device if content_disc_logits is not None else "cpu")
        
        # Content discriminator (predicting dialect from content)
        if content_disc_logits is not None and dialect_labels is not None:
            content_adv_loss = F.cross_entropy(
                content_disc_logits,
                dialect_labels,
                label_smoothing=self.label_smoothing
            )
            losses["content_adv_loss"] = content_adv_loss
            total = total + self.content_weight * content_adv_loss
        
        # Prosody discriminator (predicting content from prosody)
        if prosody_disc_logits is not None and content_indices is not None:
            # Handle potential length mismatch
            B, T_prosody, C = prosody_disc_logits.shape
            T_content = content_indices.shape[1]
            
            # Align to shorter length
            T_min = min(T_prosody, T_content)
            prosody_disc_aligned = prosody_disc_logits[:, :T_min, :]
            indices_aligned = content_indices[:, :T_min]
            
            # Flatten: [B, T, C] -> [B*T, C], [B, T] -> [B*T]
            prosody_disc_flat = prosody_disc_aligned.reshape(-1, C)
            indices_flat = indices_aligned.reshape(-1)
            
            prosody_adv_loss = F.cross_entropy(
                prosody_disc_flat,
                indices_flat,
                label_smoothing=self.label_smoothing
            )
            losses["prosody_adv_loss"] = prosody_adv_loss
            total = total + self.prosody_weight * prosody_adv_loss
        
        # Speaker discriminator
        if speaker_disc_logits is not None and speaker_labels is not None:
            speaker_adv_loss = F.cross_entropy(
                speaker_disc_logits,
                speaker_labels,
                label_smoothing=self.label_smoothing
            )
            losses["speaker_adv_loss"] = speaker_adv_loss
            total = total + self.speaker_weight * speaker_adv_loss
        
        losses["adversarial_total"] = total
        
        return losses


class MutualInformationLoss(nn.Module):
    """Mutual information minimization between content and prosody.
    
    Uses variational bounds or contrastive estimation.
    """
    
    def __init__(
        self,
        method: str = "contrastive",  # "mine" or "contrastive"
        weight: float = 0.05,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.method = method
        self.weight = weight
        self.temperature = temperature
        
    def forward(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor,
        mi_estimator: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            content: Content embeddings [B, T, D_c]
            prosody: Prosody embeddings [B, T, D_p]
            mi_estimator: Optional neural MI estimator
            
        Returns:
            Dictionary with MI loss
        """
        if self.method == "mine" and mi_estimator is not None:
            # Use MINE estimator
            mi_estimate = mi_estimator(content, prosody)
            mi_loss = self.weight * mi_estimate
            
        elif self.method == "contrastive":
            # Contrastive MI bound
            # Pool to get global representations
            content_global = content.mean(dim=1)  # [B, D_c]
            prosody_global = prosody.mean(dim=1)  # [B, D_p]
            
            # Normalize
            content_norm = F.normalize(content_global, dim=-1)
            prosody_norm = F.normalize(prosody_global, dim=-1)
            
            # Compute similarity matrix
            # We want content and prosody to be INDEPENDENT (low similarity)
            B = content.size(0)
            
            # Similarity between matched pairs (should be low)
            pos_sim = (content_norm * prosody_norm).sum(dim=-1)  # [B]
            
            # MI loss: minimize positive similarity
            mi_loss = self.weight * pos_sim.mean()
            mi_estimate = pos_sim.mean()
            
        else:
            # Simple correlation-based
            content_flat = content.reshape(content.size(0), -1)
            prosody_flat = prosody.reshape(prosody.size(0), -1)
            
            # Compute correlation
            content_centered = content_flat - content_flat.mean(dim=0, keepdim=True)
            prosody_centered = prosody_flat - prosody_flat.mean(dim=0, keepdim=True)
            
            corr = (content_centered * prosody_centered).mean()
            mi_loss = self.weight * corr.abs()
            mi_estimate = corr
        
        return {
            "mi_loss": mi_loss,
            "mi_estimate": mi_estimate
        }


class DisentanglementLoss(nn.Module):
    """Combined loss for representation disentanglement.
    
    Ensures:
    1. Content is prosody-independent
    2. Prosody is content-independent
    3. Both are speaker-independent
    """
    
    def __init__(
        self,
        orthogonality_weight: float = 0.1,
        variance_weight: float = 0.05
    ):
        super().__init__()
        
        self.orthogonality_weight = orthogonality_weight
        self.variance_weight = variance_weight
        
    def forward(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor,
        speaker: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            content: [B, T, D_c]
            prosody: [B, T, D_p]
            speaker: [B, D_s] (optional)
        """
        losses = {}
        
        # Orthogonality loss: content and prosody should span different subspaces
        # Project to same dimension and compute gram matrix
        content_pooled = content.mean(dim=1)  # [B, D_c]
        prosody_pooled = prosody.mean(dim=1)  # [B, D_p]
        
        # Normalize
        content_norm = F.normalize(content_pooled, dim=-1)
        prosody_norm = F.normalize(prosody_pooled, dim=-1)
        
        # Cross-correlation should be zero for orthogonal representations
        cross_corr = (content_norm.unsqueeze(2) * prosody_norm.unsqueeze(1)).mean(dim=0)
        orthogonality_loss = self.orthogonality_weight * cross_corr.pow(2).mean()
        losses["orthogonality_loss"] = orthogonality_loss
        
        # Variance loss: each dimension should have non-trivial variance
        content_var = content_pooled.var(dim=0).mean()
        prosody_var = prosody_pooled.var(dim=0).mean()
        
        # We want variance to be high (avoid collapse)
        variance_loss = self.variance_weight * (
            F.relu(1.0 - content_var) + F.relu(1.0 - prosody_var)
        )
        losses["variance_loss"] = variance_loss
        
        # Total
        losses["disentanglement_total"] = orthogonality_loss + variance_loss
        
        return losses


class TotalLoss(nn.Module):
    """Combined loss function for the entire model."""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        vq_weight: float = 1.0,
        adversarial_weight: float = 0.1,
        mi_weight: float = 0.05,
        disentanglement_weight: float = 0.1,
        speaker_clf_weight: float = 0.1
    ):
        super().__init__()
        
        self.reconstruction_loss = ReconstructionLoss()
        self.vq_loss = VQLoss()
        self.adversarial_loss = AdversarialLoss()
        self.mi_loss = MutualInformationLoss()
        self.disentanglement_loss = DisentanglementLoss()
        
        self.weights = {
            "reconstruction": reconstruction_weight,
            "vq": vq_weight,
            "adversarial": adversarial_weight,
            "mi": mi_weight,
            "disentanglement": disentanglement_weight,
            "speaker_clf": speaker_clf_weight
        }
        
    def forward(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        content: torch.Tensor,
        prosody: torch.Tensor,
        speaker_emb: torch.Tensor,
        vq_loss: torch.Tensor,
        vq_indices: torch.Tensor,
        num_embeddings: int,
        content_disc_logits: Optional[torch.Tensor] = None,
        dialect_labels: Optional[torch.Tensor] = None,
        prosody_disc_logits: Optional[torch.Tensor] = None,
        speaker_disc_logits: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
        speaker_clf_logits: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        mi_estimator: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute all losses and return total with breakdown.
        """
        all_losses = {}
        total = torch.tensor(0.0, device=pred_mel.device)
        
        # Reconstruction loss
        recon_losses = self.reconstruction_loss(pred_mel, target_mel, lengths)
        all_losses.update(recon_losses)
        total = total + self.weights["reconstruction"] * recon_losses["reconstruction_total"]
        
        # VQ loss
        vq_losses = self.vq_loss(vq_loss, vq_indices, num_embeddings)
        all_losses.update(vq_losses)
        total = total + self.weights["vq"] * vq_losses["vq_total"]
        
        # Adversarial loss
        if content_disc_logits is not None:
            adv_losses = self.adversarial_loss(
                content_disc_logits=content_disc_logits,
                dialect_labels=dialect_labels,
                prosody_disc_logits=prosody_disc_logits,
                content_indices=vq_indices,
                speaker_disc_logits=speaker_disc_logits,
                speaker_labels=speaker_labels
            )
            all_losses.update(adv_losses)
            total = total + self.weights["adversarial"] * adv_losses["adversarial_total"]
        
        # MI loss
        mi_losses = self.mi_loss(content, prosody, mi_estimator)
        all_losses.update(mi_losses)
        total = total + self.weights["mi"] * mi_losses["mi_loss"]
        
        # Disentanglement loss
        dis_losses = self.disentanglement_loss(content, prosody, speaker_emb)
        all_losses.update(dis_losses)
        total = total + self.weights["disentanglement"] * dis_losses["disentanglement_total"]
        
        # Speaker classification loss (auxiliary)
        if speaker_clf_logits is not None and speaker_labels is not None:
            speaker_clf_loss = F.cross_entropy(speaker_clf_logits, speaker_labels)
            all_losses["speaker_clf_loss"] = speaker_clf_loss
            total = total + self.weights["speaker_clf"] * speaker_clf_loss
        
        all_losses["total_loss"] = total
        
        return total, all_losses
