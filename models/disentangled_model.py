"""Main Prosody-Disentangled Model for Tibetan Speech.

Integrates all components:
- Content Encoder (HuBERT + VQ-VAE)
- Prosody Encoder (F0 + Energy)
- Speaker Encoder
- Decoder
- Discriminators
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

from .content_encoder import ContentEncoder, ContentEncoderMel
from .prosody_encoder import ProsodyEncoder, DialectProsodyAdapter
from .speaker_encoder import SpeakerEncoder, ECAPA_TDNN, SpeakerClassifier
from .decoder import Decoder
from .discriminator import (
    ContentDiscriminator, 
    ProsodyDiscriminator,
    SpeakerDiscriminator,
    MutualInformationEstimator,
    ContrastiveDisentangler
)


class ProsodyDisentangledModel(nn.Module):
    """Prosody-Disentangled Speech Representation Model for Tibetan.
    
    This model learns three separate representations:
    1. Content: What is being said (phonemes, words) - dialect-invariant
    2. Prosody: How it's said (pitch, rhythm) - captures tonal patterns
    3. Speaker: Who is saying it (voice characteristics)
    
    Key innovations for Tibetan:
    - Handles tonal (Lhasa/Kham) vs non-tonal (Amdo) dialect difference
    - Uses adversarial training to enforce disentanglement
    - VQ discretization for robust content representation
    """
    
    def __init__(
        self,
        # Content encoder config
        pretrained_hubert: str = "facebook/hubert-base-ls960",
        freeze_hubert: bool = False,
        content_size: int = 256,
        vq_num_embeddings: int = 512,
        use_mel_content_encoder: bool = False,
        
        # Prosody encoder config
        prosody_hidden_size: int = 256,
        prosody_size: int = 64,
        
        # Speaker encoder config
        speaker_hidden_size: int = 256,
        speaker_size: int = 128,
        num_speakers: int = 100,
        use_ecapa: bool = False,
        
        # Decoder config
        decoder_hidden_size: int = 512,
        decoder_num_layers: int = 4,
        n_mels: int = 80,
        
        # Discriminator config
        num_dialects: int = 3,
        use_mi_estimator: bool = True,
        
        # General
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.content_size = content_size
        self.prosody_size = prosody_size
        self.speaker_size = speaker_size
        self.vq_num_embeddings = vq_num_embeddings
        self.n_mels = n_mels
        self.use_mel_content_encoder = use_mel_content_encoder
        
        # Content Encoder
        if use_mel_content_encoder:
            self.content_encoder = ContentEncoderMel(
                n_mels=n_mels,
                hidden_size=512,
                output_size=content_size,
                vq_num_embeddings=vq_num_embeddings,
                dropout=dropout
            )
        else:
            self.content_encoder = ContentEncoder(
                pretrained_hubert=pretrained_hubert,
                freeze_hubert=freeze_hubert,
                output_size=content_size,
                vq_num_embeddings=vq_num_embeddings,
                dropout=dropout
            )
        
        # Prosody Encoder
        self.prosody_encoder = ProsodyEncoder(
            hidden_size=prosody_hidden_size,
            output_size=prosody_size,
            dropout=dropout
        )
        
        # Dialect-specific prosody adaptation
        self.prosody_adapter = DialectProsodyAdapter(
            prosody_size=prosody_size,
            num_dialects=num_dialects
        )
        
        # Speaker Encoder
        if use_ecapa:
            self.speaker_encoder = ECAPA_TDNN(
                n_mels=n_mels,
                output_size=speaker_size
            )
        else:
            self.speaker_encoder = SpeakerEncoder(
                n_mels=n_mels,
                hidden_size=speaker_hidden_size,
                output_size=speaker_size,
                dropout=dropout
            )
        
        # Speaker classifier (auxiliary loss)
        self.speaker_classifier = SpeakerClassifier(
            input_size=speaker_size,
            num_speakers=num_speakers
        )
        
        # Decoder
        self.decoder = Decoder(
            content_size=content_size,
            prosody_size=prosody_size,
            speaker_size=speaker_size,
            hidden_size=decoder_hidden_size,
            output_size=n_mels,
            num_layers=decoder_num_layers,
            dropout=dropout
        )
        
        # Discriminators for adversarial disentanglement
        self.content_discriminator = ContentDiscriminator(
            input_size=content_size,
            num_classes=num_dialects,
            dropout=dropout
        )
        
        self.prosody_discriminator = ProsodyDiscriminator(
            input_size=prosody_size,
            num_content_classes=vq_num_embeddings,
            dropout=dropout
        )
        
        self.speaker_discriminator = SpeakerDiscriminator(
            content_size=content_size,
            prosody_size=prosody_size,
            num_speakers=num_speakers,
            dropout=dropout
        )
        
        # Mutual information estimator
        self.use_mi_estimator = use_mi_estimator
        if use_mi_estimator:
            self.mi_estimator = MutualInformationEstimator(
                content_size=content_size,
                prosody_size=prosody_size
            )
        
        # Contrastive disentangler
        self.contrastive_disentangler = ContrastiveDisentangler(
            content_size=content_size,
            prosody_size=prosody_size
        )
        
    def forward(
        self,
        waveform: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        voiced: Optional[torch.Tensor] = None,
        dialect_id: Optional[torch.Tensor] = None,
        speaker_id: Optional[torch.Tensor] = None,
        mel_lengths: Optional[torch.Tensor] = None,
        waveform_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all model components.
        
        Args:
            waveform: Raw audio [B, T_audio] (for HuBERT content encoder)
            mel: Mel spectrogram [B, n_mels, T_mel]
            f0: F0 contour [B, T_mel]
            energy: Energy contour [B, T_mel]
            voiced: Voiced flags [B, T_mel]
            dialect_id: Dialect labels [B]
            speaker_id: Speaker labels [B]
            mel_lengths: Mel sequence lengths [B]
            waveform_lengths: Waveform lengths [B]
            
        Returns:
            Dictionary containing:
            - pred_mel: Reconstructed mel spectrogram
            - content: Content embeddings
            - prosody: Prosody embeddings
            - speaker_emb: Speaker embedding
            - vq_loss: VQ commitment loss
            - vq_indices: Codebook indices
            - Various discriminator outputs
        """
        outputs = {}
        
        # === Content Encoding ===
        if self.use_mel_content_encoder:
            content, vq_loss, vq_indices = self.content_encoder(mel, mel_lengths)
        else:
            assert waveform is not None, "Waveform required for HuBERT encoder"
            content, vq_loss, vq_indices = self.content_encoder(
                waveform, 
                attention_mask=self._create_attention_mask(waveform, waveform_lengths)
            )
        
        outputs["content"] = content
        outputs["vq_loss"] = vq_loss
        outputs["vq_indices"] = vq_indices
        
        # === Prosody Encoding ===
        assert f0 is not None and energy is not None, "F0 and energy required"
        
        prosody, prosody_global = self.prosody_encoder(
            f0, energy, voiced, mel_lengths
        )
        
        # Apply dialect-specific adaptation
        if dialect_id is not None:
            prosody = self.prosody_adapter(prosody, dialect_id)
        
        outputs["prosody"] = prosody
        outputs["prosody_global"] = prosody_global
        
        # === Speaker Encoding ===
        speaker_emb = self.speaker_encoder(mel, mel_lengths)
        outputs["speaker_emb"] = speaker_emb
        
        # Speaker classification (auxiliary)
        if speaker_id is not None:
            speaker_logits = self.speaker_classifier(speaker_emb, speaker_id)
            outputs["speaker_logits"] = speaker_logits
        
        # === Alignment ===
        # Align content and prosody lengths (they may differ due to different encoders)
        content, prosody = self._align_sequences(content, prosody, mel_lengths)
        outputs["content_aligned"] = content
        outputs["prosody_aligned"] = prosody
        
        # === Decoding ===
        pred_mel, _ = self.decoder(content, prosody, speaker_emb, mel_lengths)
        outputs["pred_mel"] = pred_mel
        
        # === Discriminators ===
        # Content discriminator (should NOT predict dialect)
        content_disc_logits = self.content_discriminator(
            content, mel_lengths, apply_grl=True
        )
        outputs["content_disc_logits"] = content_disc_logits
        
        # Prosody discriminator (should NOT predict content)
        prosody_disc_logits = self.prosody_discriminator(
            prosody, apply_grl=True
        )
        outputs["prosody_disc_logits"] = prosody_disc_logits
        
        # Speaker discriminator
        speaker_disc_logits = self.speaker_discriminator(
            content, prosody, mel_lengths, apply_grl=True
        )
        outputs["speaker_disc_logits"] = speaker_disc_logits
        
        # MI estimation
        if self.use_mi_estimator:
            mi_estimate = self.mi_estimator(content, prosody)
            outputs["mi_estimate"] = mi_estimate
        
        # Contrastive disentanglement
        disentangle_loss = self.contrastive_disentangler(content, prosody)
        outputs["disentangle_loss"] = disentangle_loss
        
        return outputs
    
    def _create_attention_mask(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Create attention mask for HuBERT."""
        if lengths is None:
            return None
        
        B, T = waveform.shape
        mask = torch.arange(T, device=waveform.device)[None, :] < lengths[:, None]
        return mask.long()
    
    def _align_sequences(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align content and prosody sequences to same length."""
        T_content = content.size(1)
        T_prosody = prosody.size(1)
        
        if T_content == T_prosody:
            return content, prosody
        
        # Interpolate to match lengths
        if T_content > T_prosody:
            prosody = F.interpolate(
                prosody.transpose(1, 2),
                size=T_content,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            content = F.interpolate(
                content.transpose(1, 2),
                size=T_prosody,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        return content, prosody
    
    def encode_content(
        self,
        waveform: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract content representation only."""
        if self.use_mel_content_encoder:
            content, _, indices = self.content_encoder(mel)
        else:
            content, _, indices = self.content_encoder(waveform)
        return content, indices
    
    def encode_prosody(
        self,
        f0: torch.Tensor,
        energy: torch.Tensor,
        voiced: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract prosody representation only."""
        prosody, _ = self.prosody_encoder(f0, energy, voiced)
        return prosody
    
    def encode_speaker(
        self,
        mel: torch.Tensor
    ) -> torch.Tensor:
        """Extract speaker embedding only."""
        return self.speaker_encoder(mel)
    
    def convert_dialect(
        self,
        waveform: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        f0_source: torch.Tensor = None,
        energy_source: torch.Tensor = None,
        f0_target: torch.Tensor = None,
        energy_target: torch.Tensor = None,
        speaker_mel: torch.Tensor = None
    ) -> torch.Tensor:
        """Convert speech from one dialect's prosody to another.
        
        Example: Amdo content + Lhasa prosody = Lhasa-accented speech
        """
        # Get content from source
        content, _ = self.encode_content(waveform, mel)
        
        # Get prosody from target
        prosody = self.encode_prosody(f0_target, energy_target)
        
        # Get speaker from reference
        speaker_emb = self.encode_speaker(speaker_mel)
        
        # Align
        content, prosody = self._align_sequences(content, prosody)
        
        # Decode
        pred_mel = self.decoder.inference(content, prosody, speaker_emb)
        
        return pred_mel
    
    def set_discriminator_grl_lambda(self, lambda_: float):
        """Adjust GRL strength during training curriculum."""
        self.content_discriminator.set_grl_lambda(lambda_)
        self.prosody_discriminator.set_grl_lambda(lambda_)
        self.speaker_discriminator.set_grl_lambda(lambda_)
    
    def freeze_encoder(self, encoder_name: str):
        """Freeze specific encoder for staged training."""
        if encoder_name == "content":
            for param in self.content_encoder.parameters():
                param.requires_grad = False
        elif encoder_name == "prosody":
            for param in self.prosody_encoder.parameters():
                param.requires_grad = False
        elif encoder_name == "speaker":
            for param in self.speaker_encoder.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder(self, encoder_name: str):
        """Unfreeze specific encoder."""
        if encoder_name == "content":
            for param in self.content_encoder.parameters():
                param.requires_grad = True
        elif encoder_name == "prosody":
            for param in self.prosody_encoder.parameters():
                param.requires_grad = True
        elif encoder_name == "speaker":
            for param in self.speaker_encoder.parameters():
                param.requires_grad = True


def create_model(config: Dict[str, Any]) -> ProsodyDisentangledModel:
    """Create model from config dictionary."""
    model_config = config.get("model", {})
    
    return ProsodyDisentangledModel(
        pretrained_hubert=model_config.get("content_encoder", {}).get(
            "pretrained_hubert", "facebook/hubert-base-ls960"
        ),
        freeze_hubert=model_config.get("content_encoder", {}).get("freeze_hubert", False),
        content_size=model_config.get("content_encoder", {}).get("output_size", 256),
        vq_num_embeddings=model_config.get("vq", {}).get("num_embeddings", 512),
        prosody_hidden_size=model_config.get("prosody_encoder", {}).get("hidden_size", 256),
        prosody_size=model_config.get("prosody_encoder", {}).get("output_size", 64),
        speaker_hidden_size=model_config.get("speaker_encoder", {}).get("hidden_size", 256),
        speaker_size=model_config.get("speaker_encoder", {}).get("output_size", 128),
        decoder_hidden_size=model_config.get("decoder", {}).get("hidden_size", 512),
        decoder_num_layers=model_config.get("decoder", {}).get("num_layers", 4),
        n_mels=config.get("data", {}).get("n_mels", 80),
        dropout=model_config.get("decoder", {}).get("dropout", 0.1)
    )
